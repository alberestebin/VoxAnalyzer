import time
import math
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, windows, resample_poly
from scipy.linalg import solve_toeplitz
from scipy.io import wavfile
import sounddevice as sd
import threading
import tempfile
import os
import winsound

# ============================================================
# 1. CONFIGURACIÓN Y MOTOR DSP/IA
# ============================================================
FS = 16000
DURACION_CAPTURA = 3
CANALES = 1
F_CORTE = 70
ORDEN_FILTRO = 4
DURACION_VENTANA_ESPECTRO = 0.70
DURACION_FRAME = 0.030
ORDEN_LPC = 16
F_MIN = 90.0
F_MAX = 5000.0
ANCHO_BANDA_MAX = 400.0
VOCALES = ["a", "e", "i", "o", "u"]

CARPETA_CAPTURAS = Path("capturas_vocales")
CARPETA_CAPTURAS.mkdir(parents=True, exist_ok=True)

def grabar_audio(duracion, fs):
    fs_rec = fs
    try:
        sd.check_input_settings(samplerate=fs, channels=CANALES)
    except Exception:
        fs_rec = int(sd.query_devices(kind="input")["default_samplerate"])

    bloques = []
    grabando = [False]

    def callback(indata, frames, tiempo, status):
        if grabando[0]: bloques.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=fs_rec, channels=CANALES, dtype="float32", latency="high", callback=callback):
        sd.sleep(1000)
        grabando[0] = True
        sd.sleep(int(duracion * 1000))
        grabando[0] = False

    if not bloques: raise RuntimeError("No se recibieron muestras.")
    x = np.concatenate(bloques).astype(np.float64)[:int(fs_rec * duracion)]

    if fs_rec != fs:
        g = math.gcd(int(fs), int(fs_rec))
        x = resample_poly(x, fs // g, fs_rec // g)
    return x

def detectar_segmento(x, fs, dur=DURACION_VENTANA_ESPECTRO, vent=0.03, paso=0.01):
    n, h = int(vent * fs), int(paso * fs)
    if len(x) <= n: return 0.0, len(x) / fs
    rms = np.array([np.sqrt(np.mean(x[i:i + n] ** 2)) for i in range(0, len(x) - n, h)])
    n_seg = max(1, int(dur / paso))
    medias = np.convolve(rms, np.ones(n_seg) / n_seg, mode="valid")
    k = int(np.argmax(medias))
    return float(k * paso), float(min(k * paso + dur, len(x) / fs))

def ventana_guardada(d, dur_total):
    if "inicio_voz" in d.files and "fin_voz" in d.files:
        ini, fin = float(d["inicio_voz"]), float(d["fin_voz"])
    else:
        ini, fin = 0.50, 1.20
    return ini, min(fin, dur_total)

def calcular_lpc(senal, orden):
    r = np.correlate(senal, senal, mode="full")[len(senal) - 1:][: orden + 1]
    coefs = solve_toeplitz(r[:orden], r[1:orden + 1])
    return np.concatenate(([1.0], -coefs))

def estimar_formantes(coefs, fs, max_candidatos=6):
    raices = np.roots(coefs)
    candidatos = []
    for raiz in raices:
        if np.imag(raiz) <= 0: continue
        f = np.angle(raiz) * fs / (2 * np.pi)
        radio = np.abs(raiz)
        if radio <= 0 or radio >= 1: continue
        bw = -fs / np.pi * np.log(radio)
        if (F_MIN <= f <= min(F_MAX, fs / 2) and 0 < bw <= ANCHO_BANDA_MAX):
            candidatos.append((float(f), float(bw)))
    candidatos.sort(key=lambda x: x[0])
    return candidatos[:max_candidatos]

def procesar_archivo_formantes(archivo):
    with np.load(archivo, allow_pickle=False) as d:
        fs = int(d["fs"])
        senal = np.asarray(d["normalizada"], dtype=np.float64)
        vocal, hablante, rep = str(d["vocal"].item()), str(d["hablante"].item()), int(d["repeticion"].item())
        ini_v, fin_v = ventana_guardada(d, len(senal) / fs)

    dur = len(senal) / fs
    ini_s = min(max(0.0, ((ini_v + fin_v) / 2) - (DURACION_FRAME / 2)), max(0.0, dur - DURACION_FRAME))
    i0 = int(ini_s * fs)
    long = int(DURACION_FRAME * fs)

    frag = senal[i0:i0+long].copy()
    frag -= np.mean(frag)
    frag = frag * windows.hamming(len(frag), sym=False)

    orden = min(ORDEN_LPC, len(frag) // 2 - 1)
    coefs = calcular_lpc(frag, orden)
    cands = estimar_formantes(coefs, fs)
    if not cands: raise ValueError("Sin candidatos a formantes.")

    def g(idx, campo): return cands[idx][campo] if idx < len(cands) else 0

    return {"vocal": vocal, "F1": g(0, 0), "F2": g(1, 0), "F3": g(2, 0)}

def entrenar_y_predecir_fiabilidad(archivo_actual):
    base_datos = {
        "a": [(730, 1090), (750, 1150), (700, 1200)],
        "e": [(530, 1840), (550, 1900), (500, 1750)],
        "i": [(270, 2290), (300, 2200), (280, 2400)],
        "o": [(570, 840),  (600, 900),  (550, 800)],
        "u": [(300, 870),  (320, 850),  (290, 900)]
    }

    archivos = sorted(CARPETA_CAPTURAS.glob("*.npz"))
    for a in archivos:
        if a == archivo_actual: continue
        try:
            r = procesar_archivo_formantes(a)
            if r["vocal"] in base_datos and r["F1"] > 0 and r["F2"] > 0:
                base_datos[r["vocal"]].append((r["F1"], r["F2"]))
        except: pass

    r_actual = procesar_archivo_formantes(archivo_actual)
    f1_new, f2_new = r_actual["F1"], r_actual["F2"]
    probabilidades = {}

    for v, puntos in base_datos.items():
        if not puntos:
            probabilidades[v] = 1e-10
            continue
        pts = np.array(puntos)
        mu_f1, mu_f2 = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        var_f1 = np.var(pts[:, 0]) + 100.0
        var_f2 = np.var(pts[:, 1]) + 100.0
        p_f1 = np.exp(-0.5 * ((f1_new - mu_f1)**2) / var_f1)
        p_f2 = np.exp(-0.5 * ((f2_new - mu_f2)**2) / var_f2)
        probabilidades[v] = p_f1 * p_f2

    suma_probs = sum(probabilidades.values())
    if suma_probs > 0:
        probs_pct = {v: (p / suma_probs) * 100 for v, p in probabilidades.items()}
    else:
        probs_pct = {v: 20.0 for v in VOCALES}

    max_prob = max(probabilidades.values())
    aviso = max_prob < 1e-6
    return r_actual, probs_pct, aviso

# ============================================================
# 2. INTERFAZ GRÁFICA
# ============================================================
def crear_boton_redondeado(parent, text, command, bg, fg="#ffffff", hover_bg=None,
                            font=("Segoe UI", 10, "bold"), width=200, height=42, radius=10):
    if hover_bg is None: hover_bg = bg
    canvas = tk.Canvas(parent, width=width, height=height, bg=parent["bg"], highlightthickness=0)

    def dibujar(color):
        canvas.delete("all")
        canvas.create_rounded_rectangle(2, 3, width-2, height-2, radius=radius, fill="#000000", outline="", stipple="gray25")
        canvas.create_rounded_rectangle(0, 0, width-4, height-3, radius=radius, fill=color, outline="")
        canvas.create_text(width//2 - 2, height//2 - 2, text=text, fill=fg, font=font)

    canvas.bind("<Enter>", lambda e: dibujar(hover_bg))
    canvas.bind("<Leave>", lambda e: dibujar(bg))
    canvas.bind("<Button-1>", lambda e: command())
    dibujar(bg)
    return canvas

def _create_rounded_rectangle(self, x1, y1, x2, y2, radius=25, **kwargs):
    points = [
        x1+radius, y1, x1+radius, y1, x2-radius, y1, x2-radius, y1, x2, y1, x2, y1+radius,
        x2, y1+radius, x2, y2-radius, x2, y2-radius, x2, y2, x2-radius, y2, x2-radius, y2,
        x1+radius, y2, x1+radius, y2, x1, y2, x1, y2-radius, x1, y2-radius, x1, y1+radius,
        x1, y1+radius, x1, y1
    ]
    return self.create_polygon(points, **kwargs, smooth=True)

tk.Canvas.create_rounded_rectangle = _create_rounded_rectangle

# ============================================================
# FUNCIÓN DE REPRODUCCIÓN (WINSOUND NATIVO)
# ============================================================
def reproducir_audio_seguro(ruta_archivo):
    temp_wav = None
    try:
        with np.load(ruta_archivo, allow_pickle=False) as d:
            fs = int(d["fs"])
            senal = np.asarray(d["normalizada"], dtype=np.float64)

        if np.max(np.abs(senal)) > 0:
            audio_int16 = (senal * 32767 / np.max(np.abs(senal))).astype(np.int16)
        else:
            audio_int16 = (senal * 32767).astype(np.int16)

        # Nombre único para evitar choques si se pulsa "reproducir" varias veces
        temp_wav = os.path.join(tempfile.gettempdir(), f"temp_audio_vox_{time.time_ns()}.wav")
        wavfile.write(temp_wav, fs, audio_int16)

        winsound.PlaySound(temp_wav, winsound.SND_FILENAME)

    except Exception as e:
        print(f"Error al reproducir: {e}")
    finally:
        if temp_wav:
            try: os.remove(temp_wav)
            except: pass

class AppVocales:
    def __init__(self, root):
        self.root = root
        self.root.title("VoxAnalyzer · Laboratorio de Análisis Acústico")
        self.root.geometry("1500x950")
        self.root.minsize(1200, 800)

        self.colors = {
            "bg_app": "#0a0e1a", "bg_sidebar": "#111827", "bg_card": "#1a2332",
            "bg_hover": "#243044", "bg_input": "#0f1623", "border": "#1f2937",
            "border_light": "#374151", "text_primary": "#f1f5f9", "text_secondary": "#94a3b8",
            "text_muted": "#64748b", "accent": "#06b6d4", "accent_hover": "#0891b2",
            "primary": "#3b82f6", "primary_hover": "#2563eb", "danger": "#ef4444",
            "danger_hover": "#dc2626", "success": "#10b981", "warning": "#f59e0b",
            "purple": "#8b5cf6", "pink": "#ec4899"
        }

        self.colores_vocales = {
            "a": "#3b82f6", "e": "#f59e0b", "i": "#10b981", "o": "#ef4444", "u": "#8b5cf6"
        }

        self.root.configure(bg=self.colors["bg_app"])
        self.root.bind("<Escape>", lambda e: self.mostrar_panel_inicio())

        self.animando_audio = False
        self.animando_grabacion = False
        self._vocal_play_sel = "a"
        self._archivos_play = []

        self.setup_fonts()
        self.setup_styles()
        self.build_ui()
        self.mostrar_panel_inicio()

    def setup_fonts(self):
        self.font_title = ("Segoe UI", 24, "bold")
        self.font_subtitle = ("Segoe UI", 13, "bold")
        self.font_body = ("Segoe UI", 10)
        self.font_small = ("Segoe UI", 9)
        self.font_mono = ("Consolas", 10)

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=self.font_body, background=self.colors["bg_app"], foreground=self.colors["text_primary"])
        style.configure("Sidebar.TButton", background=self.colors["bg_sidebar"], foreground=self.colors["text_secondary"],
                        font=("Segoe UI", 10, "bold"), borderwidth=0, padding=(20, 14), anchor="w")
        style.map("Sidebar.TButton", background=[("active", self.colors["bg_hover"])], foreground=[("active", self.colors["text_primary"])])
        style.configure("TEntry", fieldbackground=self.colors["bg_input"], foreground=self.colors["text_primary"],
                        borderwidth=1, relief="flat", padding=(12, 10))
        style.configure("TCombobox", fieldbackground=self.colors["bg_app"], background=self.colors["bg_app"],
                        foreground=self.colors["text_primary"], borderwidth=1, relief="flat", padding=(8, 5),
                        arrowcolor=self.colors["text_secondary"])
        style.map("TCombobox", fieldbackground=[("readonly", self.colors["bg_app"])], foreground=[("readonly", self.colors["text_primary"])])

    def build_ui(self):
        main = tk.Frame(self.root, bg=self.colors["bg_app"])
        main.pack(fill="both", expand=True)

        sidebar = tk.Frame(main, bg=self.colors["bg_sidebar"], width=320)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        logo_frame = tk.Frame(sidebar, bg=self.colors["bg_sidebar"], pady=30)
        logo_frame.pack(fill="x")
        logo_canvas = tk.Canvas(logo_frame, width=60, height=60, bg=self.colors["bg_sidebar"], highlightthickness=0)
        logo_canvas.pack()
        logo_canvas.create_oval(5, 5, 55, 55, fill=self.colors["accent"], outline="")
        logo_canvas.create_text(30, 30, text="V", fill="#ffffff", font=("Segoe UI", 24, "bold"))
        tk.Label(logo_frame, text="VOX ANALYZER", bg=self.colors["bg_sidebar"], fg=self.colors["text_primary"],
                 font=("Segoe UI", 14, "bold")).pack(pady=(10, 0))
        tk.Label(logo_frame, text="Análisis Acústico Profesional", bg=self.colors["bg_sidebar"],
                 fg=self.colors["text_muted"], font=self.font_small).pack()

        tk.Frame(sidebar, bg=self.colors["border"], height=1).pack(fill="x", padx=25, pady=15)

        tk.Label(sidebar, text="CONFIGURACIÓN", bg=self.colors["bg_sidebar"], fg=self.colors["text_muted"],
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=30, pady=(10, 8))

        form = tk.Frame(sidebar, bg=self.colors["bg_sidebar"])
        form.pack(fill="x", padx=25)

        def add_field(label, var, widget_type="entry"):
            tk.Label(form, text=label, bg=self.colors["bg_sidebar"], fg=self.colors["text_secondary"],
                     font=self.font_small).pack(anchor="w", pady=(8, 4))
            if widget_type == "entry":
                ttk.Entry(form, textvariable=var).pack(fill="x", pady=(0, 8))
            else:
                ttk.Combobox(form, textvariable=var, values=VOCALES, state="readonly").pack(fill="x", pady=(0, 8))

        self.var_hablante = tk.StringVar(value="H01")
        add_field("Identificador del Hablante", self.var_hablante)
        self.var_vocal = tk.StringVar(value="a")
        add_field("Vocal Objetivo", self.var_vocal, "combo")
        self.var_rep = tk.StringVar(value="1")
        add_field("Repetición", self.var_rep)

        tk.Label(sidebar, text="ACCIONES", bg=self.colors["bg_sidebar"], fg=self.colors["text_muted"],
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=30, pady=(20, 8))

        btn_grabar = crear_boton_redondeado(sidebar, "🔴  Iniciar Grabación", self.accion_grabar,
                                             bg=self.colors["danger"], hover_bg=self.colors["danger_hover"],
                                             width=270, height=48, radius=12)
        btn_grabar.pack(padx=25, pady=8)

        btn_espectro = crear_boton_redondeado(sidebar, "📊  Análisis de Espectro", self.accion_espectro,
                                               bg=self.colors["accent"], hover_bg=self.colors["accent_hover"],
                                               width=270, height=44, radius=10)
        btn_espectro.pack(padx=25, pady=6)

        btn_formantes = crear_boton_redondeado(sidebar, "📈  Análisis de Formantes", self.accion_formantes,
                                                bg=self.colors["primary"], hover_bg=self.colors["primary_hover"],
                                                width=270, height=44, radius=10)
        btn_formantes.pack(padx=25, pady=6)

        tk.Frame(sidebar, bg=self.colors["border"], height=1).pack(fill="x", padx=25, pady=20)

        tk.Label(sidebar, text="HERRAMIENTAS", bg=self.colors["bg_sidebar"], fg=self.colors["text_muted"],
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=30, pady=(0, 8))

        ttk.Button(sidebar, text="🗂️   Explorador Completo", style="Sidebar.TButton",
                   command=self.cargar_lista_reproductor).pack(fill="x", padx=15, pady=3)
        ttk.Button(sidebar, text="🩺   Diagnóstico de Entrada", style="Sidebar.TButton",
                   command=self.accion_diagnostico).pack(fill="x", padx=15, pady=3)

        tk.Label(sidebar, text="v2.6 · VoxAnalyzer", bg=self.colors["bg_sidebar"],
                 fg=self.colors["text_muted"], font=self.font_small).pack(side="bottom", pady=20)

        right_container = tk.Frame(main, bg=self.colors["bg_app"])
        right_container.pack(side="right", fill="both", expand=True)

        header = tk.Frame(right_container, bg=self.colors["bg_app"], height=90)
        header.pack(fill="x", padx=40, pady=(25, 15))
        header.pack_propagate(False)

        title_frame = tk.Frame(header, bg=self.colors["bg_app"])
        title_frame.pack(side="left", fill="y")
        tk.Label(title_frame, text="Laboratorio de Voz", bg=self.colors["bg_app"],
                 fg=self.colors["text_primary"], font=self.font_title).pack(anchor="w")
        tk.Label(title_frame, text="Procesamiento digital y análisis de formantes vocálicos",
                 bg=self.colors["bg_app"], fg=self.colors["text_muted"], font=self.font_small).pack(anchor="w")

        status_frame = tk.Frame(header, bg=self.colors["bg_app"])
        status_frame.pack(side="right", fill="y")
        self.status_dot = tk.Canvas(status_frame, width=12, height=12, bg=self.colors["bg_app"], highlightthickness=0)
        self.status_dot.pack(side="left", padx=(0, 8), pady=20)
        self.status_dot.create_oval(2, 2, 10, 10, fill=self.colors["success"], outline="")

        self.status_var = tk.StringVar(value="Listo para operar")
        tk.Label(status_frame, textvariable=self.status_var, bg=self.colors["bg_app"],
                 fg=self.colors["text_secondary"], font=self.font_small).pack(side="left", pady=20)

        tk.Frame(right_container, bg=self.colors["border"], height=1).pack(fill="x", padx=40)

        self.right = tk.Frame(right_container, bg=self.colors["bg_card"],
                              highlightbackground=self.colors["border"], highlightthickness=1)
        self.right.pack(fill="both", expand=True, padx=40, pady=(20, 30))

        footer = tk.Frame(right_container, bg=self.colors["bg_sidebar"], height=30)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        tk.Label(footer, text="©️ 2024 VoxAnalyzer · Todos los derechos reservados",
                 bg=self.colors["bg_sidebar"], fg=self.colors["text_muted"],
                 font=("Segoe UI", 8)).pack(side="left", padx=40, pady=8)
        self.info_footer = tk.Label(footer, text="FS: 16000 Hz | Filtro: Highpass 70 Hz",
                                     bg=self.colors["bg_sidebar"], fg=self.colors["text_muted"],
                                     font=("Segoe UI", 8))
        self.info_footer.pack(side="right", padx=40, pady=8)

    def set_status(self, msg, color=None):
        self.status_var.set(msg)
        if color:
            self.status_dot.delete("all")
            self.status_dot.create_oval(2, 2, 10, 10, fill=color, outline="")
        self.root.update()

    def detener_audio(self):
        """Detiene cualquier reproducción en curso (sounddevice y winsound)."""
        sd.stop()
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass
        self.animando_audio = False

    def limpiar_panel(self):
        self.detener_audio()
        self.animando_grabacion = False
        for widget in self.right.winfo_children():
            widget.destroy()

    def agregar_boton_volver(self, parent):
        btn_volver = crear_boton_redondeado(parent, "⬅  Volver al Inicio", self.mostrar_panel_inicio,
                                             bg=self.colors["bg_hover"], hover_bg=self.colors["border_light"],
                                             width=160, height=36, radius=8)
        btn_volver.place(relx=1.0, rely=0.0, anchor="ne", x=-20, y=20)

    def crear_medidor_visual(self, parent, color_fondo="#0f1623", color_barra="#10b981"):
        canvas = tk.Canvas(parent, width=270, height=10, bg=parent["bg"], highlightthickness=0)
        canvas.pack(pady=5)
        canvas.create_rectangle(0, 0, 270, 10, fill=color_fondo, outline="", tags="fondo")
        barra = canvas.create_rectangle(0, 0, 0, 10, fill=color_barra, outline="", tags="barra")
        return canvas, barra

    def animar_grabacion(self, canvas, barra):
        if not self.animando_grabacion: return
        import random
        nivel = random.uniform(0.1, 0.9)
        if random.random() > 0.8: nivel = random.uniform(0.5, 1.0)
        canvas.coords(barra, 0, 0, 270 * nivel, 10)
        if nivel > 0.8: canvas.itemconfig(barra, fill=self.colors["danger"])
        elif nivel > 0.5: canvas.itemconfig(barra, fill=self.colors["warning"])
        else: canvas.itemconfig(barra, fill=self.colors["success"])
        self.root.after(50, lambda: self.animar_grabacion(canvas, barra))

    def animar_reproduccion(self, canvas, barra, duracion):
        if not self.animando_audio: return
        import random
        nivel = random.uniform(0.2, 0.8)
        if random.random() > 0.7: nivel = random.uniform(0.6, 1.0)
        canvas.coords(barra, 0, 0, 270 * nivel, 10)
        canvas.itemconfig(barra, fill=self.colors["accent"])
        self.root.after(50, lambda: self.animar_reproduccion(canvas, barra, duracion))

    # ============================================================
    # ELIMINACIÓN DE GRABACIONES
    # ============================================================
    def eliminar_archivos(self, rutas, callback=None):
        """Pide confirmación, borra los .npz y ejecuta callback para refrescar la vista."""
        if not rutas:
            return
        if len(rutas) == 1:
            msg = f"¿Eliminar definitivamente esta grabación?\n\n{rutas[0].name}"
        else:
            msg = f"¿Eliminar definitivamente {len(rutas)} grabaciones?"
        if not messagebox.askyesno("Confirmar eliminación", msg, icon="warning"):
            return

        self.detener_audio()
        errores = []
        borrados = 0
        for r in rutas:
            try:
                r.unlink()
                borrados += 1
            except Exception as e:
                errores.append(f"{r.name}: {e}")

        if errores:
            messagebox.showerror("Error al eliminar", "\n".join(errores))
        self.set_status(f"{borrados} grabación(es) eliminada(s)", self.colors["warning"])
        if callback:
            callback()

    # ============================================================
    # PANEL DE INICIO
    # ============================================================
    def mostrar_panel_inicio(self):
        self.limpiar_panel()

        header_frame = tk.Frame(self.right, bg=self.colors["bg_card"])
        header_frame.pack(fill="x", padx=50, pady=(30, 10))

        tk.Label(header_frame, text="🎙️ Biblioteca de Reproducción", bg=self.colors["bg_card"],
                 fg=self.colors["text_primary"], font=("Segoe UI", 22, "bold")).pack(anchor="w")
        tk.Label(header_frame, text="Selecciona la vocal y la grabación en el panel derecho para escuchar o eliminar.",
                 bg=self.colors["bg_card"], fg=self.colors["text_muted"], font=self.font_body).pack(anchor="w", pady=(5, 0))

        tk.Frame(self.right, bg=self.colors["border"], height=1).pack(fill="x", padx=50, pady=20)

        content_frame = tk.Frame(self.right, bg=self.colors["bg_card"])
        content_frame.pack(fill="both", expand=True, padx=40, pady=10)

        left_cards = tk.Frame(content_frame, bg=self.colors["bg_card"])
        left_cards.pack(side="left", fill="both", expand=True)

        right_player = tk.Frame(content_frame, bg=self.colors["bg_app"], width=380,
                                highlightbackground=self.colors["border"], highlightthickness=1)
        right_player.pack(side="right", fill="y", padx=(20, 0))
        right_player.pack_propagate(False)

        # --- PANEL IZQUIERDO: TARJETAS DE VOCALES ---
        left_cards.columnconfigure(0, weight=1)
        left_cards.columnconfigure(1, weight=1)
        left_cards.columnconfigure(2, weight=1)

        for idx, vocal in enumerate(VOCALES):
            archivos = sorted(CARPETA_CAPTURAS.glob(f"*_{vocal}_rep*.npz"), key=lambda x: x.stat().st_mtime, reverse=True)

            card = tk.Frame(left_cards, bg=self.colors["bg_app"], padx=20, pady=20,
                            highlightbackground=self.colores_vocales[vocal], highlightthickness=2)
            row = idx // 3
            col = idx % 3
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

            letra_canvas = tk.Canvas(card, width=80, height=80, bg=self.colors["bg_app"], highlightthickness=0)
            letra_canvas.pack(pady=(5, 5))
            color_vocal = self.colores_vocales[vocal]
            letra_canvas.create_oval(5, 5, 75, 75, fill=self.colors["bg_card"], outline=color_vocal, width=2)
            letra_canvas.create_text(40, 40, text=vocal.upper(), fill=color_vocal, font=("Segoe UI", 35, "bold"))

            cant = len(archivos)
            tk.Label(card, text=f"{cant} grabación{'es' if cant != 1 else ''} disponible{'s' if cant != 1 else ''}",
                     bg=self.colors["bg_app"], fg=self.colors["text_secondary"], font=self.font_small).pack(pady=(0, 10))

        # --- PANEL DERECHO: REPRODUCTOR ---
        tk.Label(right_player, text="REPRODUCTOR DE AUDIO", bg=self.colors["bg_app"],
                 fg=self.colors["accent"], font=("Segoe UI", 10, "bold")).pack(pady=(30, 20))

        tk.Label(right_player, text="1. Elige la vocal:", bg=self.colors["bg_app"],
                 fg=self.colors["text_secondary"], font=self.font_body).pack(anchor="w", padx=30)

        self.var_vocal_play = tk.StringVar(value=self._vocal_play_sel)
        combo_vocal_play = ttk.Combobox(right_player, textvariable=self.var_vocal_play, values=VOCALES, state="readonly")
        combo_vocal_play.pack(fill="x", padx=30, pady=5)

        tk.Label(right_player, text="2. Elige la grabación:", bg=self.colors["bg_app"],
                 fg=self.colors["text_secondary"], font=self.font_body).pack(anchor="w", padx=30, pady=(15, 0))

        self.var_archivo_play = tk.StringVar()
        self.combo_archivo_play = ttk.Combobox(right_player, textvariable=self.var_archivo_play, state="readonly")
        self.combo_archivo_play.pack(fill="x", padx=30, pady=5)

        def actualizar_archivos(event=None):
            vocal_sel = self.var_vocal_play.get()
            self._vocal_play_sel = vocal_sel
            self._archivos_play = sorted(CARPETA_CAPTURAS.glob(f"*_{vocal_sel}_rep*.npz"),
                                         key=lambda x: x.stat().st_mtime, reverse=True)
            opciones = []
            for a in self._archivos_play:
                try:
                    partes = a.stem.split("_")
                    fecha_dt = datetime.strptime(partes[-2] + "_" + partes[-1], "%Y%m%d_%H%M%S")
                    opciones.append(f"{partes[0]} · {fecha_dt.strftime('%d/%m/%Y - %H:%M:%S')}")
                except Exception:
                    opciones.append(a.name)

            self.combo_archivo_play["values"] = opciones
            if opciones:
                self.combo_archivo_play.current(0)
            else:
                self.var_archivo_play.set("")

        combo_vocal_play.bind("<<ComboboxSelected>>", actualizar_archivos)
        actualizar_archivos()

        def archivo_seleccionado():
            idx = self.combo_archivo_play.current()
            if idx < 0 or idx >= len(self._archivos_play):
                messagebox.showinfo("Sin datos", "No hay grabaciones para esta vocal.")
                return None
            return self._archivos_play[idx]

        def play_seleccionado():
            a = archivo_seleccionado()
            if a:
                self._reproducir_archivo(a, right_player)

        def eliminar_seleccionado():
            a = archivo_seleccionado()
            if a:
                # Al terminar se reconstruye el panel (actualiza contadores de las tarjetas)
                self.eliminar_archivos([a], callback=self.mostrar_panel_inicio)

        btn_bg = tk.Frame(right_player, bg=self.colors["success"], padx=2, pady=2)
        btn_bg.pack(fill="x", padx=30, pady=(30, 10))
        tk.Button(btn_bg, text="▶️ REPRODUCIR AUDIO", command=play_seleccionado,
                  bg="#ffffff", fg=self.colors["bg_app"], font=("Segoe UI", 11, "bold"),
                  relief="flat", cursor="hand2", borderwidth=0).pack(fill="x", expand=True, ipady=6)

        btn_del_bg = tk.Frame(right_player, bg=self.colors["danger"], padx=2, pady=2)
        btn_del_bg.pack(fill="x", padx=30, pady=(0, 10))
        tk.Button(btn_del_bg, text="🗑 ELIMINAR AUDIO", command=eliminar_seleccionado,
                  bg=self.colors["danger"], fg="#ffffff", activebackground=self.colors["danger_hover"],
                  activeforeground="#ffffff", font=("Segoe UI", 11, "bold"),
                  relief="flat", cursor="hand2", borderwidth=0).pack(fill="x", expand=True, ipady=6)

    def _reproducir_archivo(self, ruta_archivo, parent_widget=None):
        self.detener_audio()
        self.set_status(f"Reproduciendo: {ruta_archivo.name}", self.colors["success"])

        hilo = threading.Thread(target=reproducir_audio_seguro, args=(ruta_archivo,))
        hilo.daemon = True
        hilo.start()

        if parent_widget:
            for w in parent_widget.winfo_children():
                if isinstance(w, tk.Canvas) and w.winfo_height() == 10:
                    w.destroy()

            canvas, barra = self.crear_medidor_visual(parent_widget, color_barra=self.colors["accent"])
            self.animando_audio = True
            self.animar_reproduccion(canvas, barra, 3.0)

    def mostrar_interfaz_resultado(self, letra, f1, f2, f3, probs, aviso=False, ruta_audio=None, vocal_detectada=None):
        self.limpiar_panel()

        main_frame = tk.Frame(self.right, bg=self.colors["bg_card"])
        main_frame.pack(fill="both", expand=True, padx=50, pady=30)

        self.agregar_boton_volver(main_frame)

        tk.Label(main_frame, text="RESULTADO DEL ANÁLISIS", bg=self.colors["bg_card"],
                 fg=self.colors["accent"], font=("Segoe UI", 10, "bold")).pack(pady=(0, 5))

        letra_canvas = tk.Canvas(main_frame, width=200, height=200, bg=self.colors["bg_card"], highlightthickness=0)
        letra_canvas.pack(pady=(10, 15))

        color = self.colores_vocales.get(letra, self.colors["accent"])
        letra_canvas.create_oval(15, 15, 185, 185, fill=self.colors["bg_app"], outline=color, width=3)
        letra_canvas.create_text(100, 100, text=letra.upper(), fill=color, font=("Segoe UI", 90, "bold"))

        form_frame = tk.Frame(main_frame, bg=self.colors["bg_card"])
        form_frame.pack(pady=15)

        def add_f_card(label, val, color_accent):
            card = tk.Frame(form_frame, bg=self.colors["bg_app"], padx=25, pady=15)
            card.pack(side="left", padx=10)
            tk.Frame(card, bg=color_accent, height=3).place(x=0, y=0, relwidth=1)
            tk.Label(card, text=label, bg=self.colors["bg_app"], fg=self.colors["text_muted"], font=("Segoe UI", 9, "bold")).pack(pady=(8, 2))
            tk.Label(card, text=f"{val:.0f}" if val > 0 else "---", bg=self.colors["bg_app"],
                     fg=self.colors["text_primary"], font=("Segoe UI", 24, "bold")).pack()
            tk.Label(card, text="Hz", bg=self.colors["bg_app"], fg=self.colors["text_muted"], font=("Segoe UI", 9)).pack()

        add_f_card("FORMANTE 1", f1, self.colors["primary"])
        add_f_card("FORMANTE 2", f2, self.colors["accent"])
        add_f_card("FORMANTE 3", f3, self.colors["purple"])

        if aviso:
            aviso_frame = tk.Frame(main_frame, bg="#422006", highlightbackground="#b45309", highlightthickness=1)
            aviso_frame.pack(fill="x", pady=15, padx=60)
            tk.Label(aviso_frame, text="⚠️  Los formantes detectados se alejan de los patrones de entrenamiento.\nLa estimación puede ser poco fiable.",
                     bg="#422006", fg="#fbbf24", font=("Segoe UI", 9), justify="center").pack(padx=15, pady=10)

        # Aviso de vocal incorrecta (la pronunciada no coincide con la seleccionada)
        if vocal_detectada:
            err_frame = tk.Frame(main_frame, bg="#450a0a", highlightbackground="#dc2626", highlightthickness=1)
            err_frame.pack(fill="x", pady=10, padx=60)
            tk.Label(err_frame,
                     text=f"❌  Seleccionaste '{letra.upper()}' pero parece que pronunciaste "
                          f"'{vocal_detectada.upper()}' ({probs[vocal_detectada]:.0f}%).",
                     bg="#450a0a", fg="#fca5a5", font=("Segoe UI", 10, "bold"),
                     justify="center").pack(padx=15, pady=10)

        btn_frame = tk.Frame(main_frame, bg=self.colors["bg_card"])
        btn_frame.pack(pady=20)

        btn_nueva = crear_boton_redondeado(btn_frame, "🔴  Nueva Grabación", self.accion_grabar,
                                            bg=self.colors["danger"], hover_bg=self.colors["danger_hover"],
                                            width=250, height=48, radius=12)
        btn_nueva.pack(side="left", padx=10)

        if ruta_audio:
            def play_con_medidor():
                self._reproducir_archivo(ruta_audio, parent_widget=btn_frame)

            btn_escuchar = crear_boton_redondeado(btn_frame, "▶️  Escuchar esta Grabación", play_con_medidor,
                                                   bg=self.colors["success"], hover_bg="#059669",
                                                   width=250, height=48, radius=12)
            btn_escuchar.pack(side="left", padx=10)

            def eliminar_esta():
                self.eliminar_archivos([ruta_audio], callback=self.mostrar_panel_inicio)

            btn_eliminar = crear_boton_redondeado(btn_frame, "🗑  Eliminar", eliminar_esta,
                                                   bg=self.colors["bg_hover"], hover_bg=self.colors["danger"],
                                                   width=160, height=48, radius=12)
            btn_eliminar.pack(side="left", padx=10)

        tk.Frame(main_frame, bg=self.colors["border"], height=1).pack(fill="x", padx=60, pady=15)

        tk.Label(main_frame, text="PROBABILIDAD ESTIMADA", bg=self.colors["bg_card"],
                 fg=self.colors["text_primary"], font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=60, pady=(0, 15))

        probs_frame = tk.Frame(main_frame, bg=self.colors["bg_card"])
        probs_frame.pack(fill="x", padx=60, pady=(0, 10))

        max_prob = max(probs.values()) if probs else 0
        for v in VOCALES:
            pct = probs.get(v, 0)
            fila = tk.Frame(probs_frame, bg=self.colors["bg_card"], pady=6)
            fila.pack(fill="x")
            tk.Label(fila, text=v.upper(), bg=self.colors["bg_card"], fg=self.colores_vocales[v],
                     font=("Segoe UI", 12, "bold"), width=3, anchor="w").pack(side="left")
            bar_bg = tk.Frame(fila, bg=self.colors["bg_app"], height=18)
            bar_bg.pack(side="left", fill="x", expand=True, padx=15)
            bar_bg.pack_propagate(False)
            if pct > 0:
                color_barra = self.colores_vocales[v] if pct == max_prob else self.colors["text_muted"]
                bar_fill = tk.Frame(bar_bg, bg=color_barra, height=18)
                bar_fill.place(x=0, y=0, relwidth=pct/100.0, relheight=1)
            tk.Label(fila, text=f"{int(pct)}%", bg=self.colors["bg_card"], fg=self.colors["text_primary"],
                     font=("Segoe UI", 11, "bold"), width=5, anchor="e").pack(side="right")

    def accion_grabar(self):
        self.limpiar_panel()
        self.set_status("Grabando audio...", self.colors["danger"])

        frame = tk.Frame(self.right, bg=self.colors["bg_card"])
        frame.pack(expand=True)

        canvas = tk.Canvas(frame, width=150, height=150, bg=self.colors["bg_card"], highlightthickness=0)
        canvas.pack(pady=(20, 10))
        canvas.create_oval(15, 15, 135, 135, fill=self.colors["danger"], outline="", width=0)
        canvas.create_text(75, 75, text="🎙️", font=("Segoe UI", 50))

        tk.Label(frame, text="Nivel de entrada:", bg=self.colors["bg_card"], fg=self.colors["text_muted"], font=self.font_small).pack()
        medidor_canvas, medidor_barra = self.crear_medidor_visual(frame)
        self.animando_grabacion = True
        self.animar_grabacion(medidor_canvas, medidor_barra)

        tk.Label(frame, text="Grabando señal acústica...", bg=self.colors["bg_card"],
                 fg=self.colors["text_primary"], font=("Segoe UI", 20, "bold")).pack(pady=(10, 5))
        tk.Label(frame, text=f"Pronuncie la vocal '{self.var_vocal.get().upper()}' claramente",
                 bg=self.colors["bg_card"], fg=self.colors["text_muted"], font=self.font_body).pack(pady=5)
        self.root.update()

        try:
            x = grabar_audio(DURACION_CAPTURA, FS)
        except Exception as e:
            self.animando_grabacion = False
            return messagebox.showerror("Error de Hardware", str(e))

        self.animando_grabacion = False

        x_dc = x - np.mean(x)
        sos = butter(ORDEN_FILTRO, F_CORTE, btype="highpass", fs=FS, output="sos")
        x_filtrada = sosfiltfilt(sos, x_dc)
        pico_f = float(np.max(np.abs(x_filtrada)))
        x_norm = x_filtrada / pico_f if pico_f > 1e-8 else x_filtrada
        ini_v, fin_v = detectar_segmento(x_norm, FS)

        vocal = self.var_vocal.get().lower()
        hablante = self.var_hablante.get().upper() or "H01"
        marca = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta = CARPETA_CAPTURAS / f"{hablante}_{vocal}_rep01_{marca}.npz"

        np.savez(ruta, fs=FS, vocal=vocal, hablante=hablante, repeticion=1,
                 original=x, filtrada=x_filtrada, normalizada=x_norm,
                 inicio_voz=ini_v, fin_voz=fin_v)

        r_actual, probs, aviso = entrenar_y_predecir_fiabilidad(ruta)

        # Vocal que realmente parece haberse pronunciado
        vocal_detectada = max(probs, key=probs.get)
        no_coincide = vocal_detectada != vocal

        self.set_status("Análisis completado", self.colors["success"])
        self.mostrar_interfaz_resultado(vocal, r_actual["F1"], r_actual["F2"], r_actual["F3"],
                                         probs, aviso=aviso, ruta_audio=ruta,
                                         vocal_detectada=vocal_detectada if no_coincide else None)

        if no_coincide:
            self.set_status("La vocal no coincide con la seleccionada", self.colors["warning"])
            messagebox.showwarning(
                "Vocal incorrecta",
                f"Seleccionaste la vocal '{vocal.upper()}', pero parece que pronunciaste "
                f"la '{vocal_detectada.upper()}' ({probs[vocal_detectada]:.0f}% de probabilidad).\n\n"
                f"Puedes eliminar esta grabación y repetirla."
            )

    def accion_espectro(self):
        self.limpiar_panel()
        hablante = self.var_hablante.get().upper() or "H01"
        vocal = self.var_vocal.get().lower()
        archivos = sorted(CARPETA_CAPTURAS.glob(f"{hablante}_{vocal}_rep01_*.npz"))
        if not archivos:
            self.mostrar_panel_inicio()
            return messagebox.showwarning("Sin datos", "No se encontró grabación previa.")

        with np.load(archivos[-1], allow_pickle=False) as d:
            fs = int(d["fs"])
            senal = np.asarray(d["normalizada"], dtype=np.float64)
            ini, fin = ventana_guardada(d, len(senal) / fs)

        frag = senal[int(ini * fs):int(fin * fs)].copy()
        frag -= np.mean(frag)
        ventana = windows.hann(len(frag), sym=False)
        n_fft = 2 ** int(np.ceil(np.log2(len(frag))))
        X = np.fft.rfft(frag * ventana, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        mag_db = 20 * np.log10(np.maximum(np.abs(X) / np.max(np.abs(X)), 1e-8))

        self.set_status("Espectro FFT generado", self.colors["accent"])
        self.dibujar_espectro(freqs, mag_db, fs)

    def dibujar_espectro(self, freqs, mag_db, fs):
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=self.colors["bg_card"])
        ax.set_facecolor(self.colors["bg_app"])
        ax.fill_between(freqs, mag_db, -80, color=self.colors["accent"], alpha=0.15)
        ax.plot(freqs, mag_db, color=self.colors["accent"], linewidth=1.8)
        ax.set_xlim(0, min(4000.0, fs / 2.0))
        ax.set_ylim(-80, 5)
        ax.set_title("Espectro de Frecuencias (FFT)", color=self.colors["text_primary"],
                     pad=20, fontweight="bold", fontsize=13)
        ax.set_xlabel("Frecuencia (Hz)", color=self.colors["text_muted"], fontweight="bold", fontsize=10)
        ax.set_ylabel("Magnitud (dB)", color=self.colors["text_muted"], fontweight="bold", fontsize=10)
        ax.tick_params(colors=self.colors["text_secondary"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(self.colors["border_light"])
        ax.grid(True, color=self.colors["border_light"], linestyle="--", alpha=0.3)
        fig.tight_layout()

        frame_grafica = tk.Frame(self.right, bg=self.colors["bg_card"])
        frame_grafica.pack(fill="both", expand=True, padx=30, pady=30)

        self.agregar_boton_volver(frame_grafica)

        canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def accion_formantes(self):
        self.limpiar_panel()
        archivos = sorted(CARPETA_CAPTURAS.glob("*.npz"))
        if not archivos:
            self.mostrar_panel_inicio()
            return messagebox.showwarning("Sin datos", "El repositorio está vacío.")

        datos_por_vocal = {v: [] for v in VOCALES}
        for a in archivos:
            try:
                r = procesar_archivo_formantes(a)
                if r["vocal"] in datos_por_vocal and r["F1"] > 0 and r["F2"] > 0:
                    datos_por_vocal[r["vocal"]].append((r["F1"], r["F2"]))
            except:
                pass

        self.set_status("Análisis de dispersión completado", self.colors["success"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=self.colors["bg_card"])
        fig.subplots_adjust(wspace=0.3, left=0.08, right=0.95)

        ax1.set_facecolor(self.colors["bg_app"])
        etiquetas = ["a", "e", "i", "o", "u"]
        centroides = {v: np.mean(pts, axis=0) for v, pts in datos_por_vocal.items() if pts}
        matriz = np.zeros((5, 5), dtype=int)
        if centroides:
            for v_real in VOCALES:
                for f1, f2 in datos_por_vocal[v_real]:
                    mejor_v = min(centroides.keys(), key=lambda v_pred: np.sqrt((f1 - centroides[v_pred][0])**2 + (f2 - centroides[v_pred][1])**2))
                    matriz[etiquetas.index(v_real), etiquetas.index(mejor_v)] += 1

        cmap_custom = LinearSegmentedColormap.from_list("custom", ["#0a0e1a", "#1e3a8a", "#3b82f6", "#06b6d4"])
        ax1.imshow(matriz, cmap=cmap_custom)
        ax1.set_xticks(np.arange(5)); ax1.set_yticks(np.arange(5))
        ax1.set_xticklabels(etiquetas); ax1.set_yticklabels(etiquetas)
        ax1.set_xlabel("Predicción", color=self.colors["text_muted"], fontweight="bold")
        ax1.set_ylabel("Etiqueta Real", color=self.colors["text_muted"], fontweight="bold")
        ax1.set_title("Matriz de Confusión", color=self.colors["text_primary"], pad=15, fontweight="bold", fontsize=12)
        for i in range(5):
            for j in range(5):
                if matriz[i, j] > 0:
                    color = "#ffffff" if matriz[i, j] > matriz.max() / 2 else self.colors["text_primary"]
                    ax1.text(j, i, str(matriz[i, j]), ha="center", va="center", color=color, fontsize=13, fontweight="bold")

        ax2.set_facecolor(self.colors["bg_app"])
        hay_puntos = False
        for v in VOCALES:
            pts = np.array(datos_por_vocal[v])
            if len(pts) > 0:
                hay_puntos = True
                ax2.scatter(pts[:, 1], pts[:, 0], c=self.colores_vocales[v], label=v.upper(),
                           s=100, alpha=0.9, edgecolors=self.colors["bg_app"], linewidth=1.5)
        ax2.set_title("Espacio Vocal (F1 vs F2)", color=self.colors["text_primary"], pad=15, fontweight="bold", fontsize=12)
        ax2.set_xlabel("F2 (Hz)", color=self.colors["text_muted"], fontweight="bold")
        ax2.set_ylabel("F1 (Hz)", color=self.colors["text_muted"], fontweight="bold")
        ax2.invert_xaxis(); ax2.invert_yaxis()
        if hay_puntos:
            legend = ax2.legend(framealpha=1, facecolor=self.colors["bg_app"], edgecolor=self.colors["border_light"])
            for text in legend.get_texts():
                text.set_color(self.colors["text_primary"])
        ax2.grid(True, color=self.colors["border_light"], linestyle="--", alpha=0.3)
        for ax in (ax1, ax2):
            ax.tick_params(colors=self.colors["text_secondary"])
            for spine in ax.spines.values():
                spine.set_color(self.colors["border_light"])
        fig.tight_layout()

        frame_grafica = tk.Frame(self.right, bg=self.colors["bg_card"])
        frame_grafica.pack(fill="both", expand=True, padx=30, pady=30)

        self.agregar_boton_volver(frame_grafica)

        canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ============================================================
    # EXPLORADOR COMPLETO (selección múltiple + eliminar)
    # ============================================================
    def cargar_lista_reproductor(self):
        self.limpiar_panel()
        archivos = sorted(CARPETA_CAPTURAS.glob("*.npz"))
        if not archivos:
            self.mostrar_panel_inicio()
            return messagebox.showwarning("Sin datos", "No hay grabaciones guardadas.")

        main_frame = tk.Frame(self.right, bg=self.colors["bg_card"])
        main_frame.pack(fill="both", expand=True, padx=40, pady=20)

        self.agregar_boton_volver(main_frame)

        tk.Label(main_frame, text="🗂️ Explorador Completo", bg=self.colors["bg_card"],
                 fg=self.colors["text_primary"], font=("Segoe UI", 18, "bold")).pack(pady=(0, 5))
        tk.Label(main_frame, text=f"{len(archivos)} archivos encontrados en el repositorio  ·  Ctrl/Shift para seleccionar varios",
                 bg=self.colors["bg_card"], fg=self.colors["text_muted"], font=self.font_small).pack(pady=(0, 20))

        list_frame = tk.Frame(main_frame, bg=self.colors["bg_card"])
        list_frame.pack(fill="both", expand=True, padx=20, pady=10)

        scrollbar = tk.Scrollbar(list_frame, bg=self.colors["bg_sidebar"], troughcolor=self.colors["bg_app"])
        scrollbar.pack(side="right", fill="y")

        listbox = tk.Listbox(list_frame, selectmode="extended", bg=self.colors["bg_app"], fg=self.colors["text_primary"],
                             font=("Consolas", 10), selectbackground=self.colors["accent"],
                             selectforeground="#ffffff", borderwidth=0, highlightthickness=1,
                             highlightbackground=self.colors["border"], yscrollcommand=scrollbar.set,
                             exportselection=False)
        listbox.pack(fill="both", expand=True)
        scrollbar.config(command=listbox.yview)

        for a in archivos:
            listbox.insert("end", f"  📄  {a.name}")

        def seleccion_unica():
            sel = listbox.curselection()
            if not sel:
                messagebox.showinfo("Selecciona un archivo", "Primero selecciona un archivo de la lista.")
                return None
            return archivos[sel[0]]

        def abrir():
            a = seleccion_unica()
            if a: self.reproductor_grafico(a)

        def escuchar_rapido():
            a = seleccion_unica()
            if a: self._reproducir_archivo(a, parent_widget=main_frame)

        def eliminar_sel():
            sel = listbox.curselection()
            if not sel:
                return messagebox.showinfo("Selecciona un archivo", "Primero selecciona uno o más archivos.")
            rutas = [archivos[i] for i in sel]
            self.eliminar_archivos(rutas, callback=self.cargar_lista_reproductor)

        btn_frame = tk.Frame(main_frame, bg=self.colors["bg_card"])
        btn_frame.pack(pady=20)

        btn_abrir = crear_boton_redondeado(btn_frame, "▶️   Ver Forma de Onda", abrir,
                                            bg=self.colors["accent"], hover_bg=self.colors["accent_hover"],
                                            width=240, height=48, radius=12)
        btn_abrir.pack(side="left", padx=10)

        btn_escuchar = crear_boton_redondeado(btn_frame, "🎧   Escuchar Rápido", escuchar_rapido,
                                               bg=self.colors["success"], hover_bg="#059669",
                                               width=220, height=48, radius=12)
        btn_escuchar.pack(side="left", padx=10)

        btn_eliminar = crear_boton_redondeado(btn_frame, "🗑   Eliminar", eliminar_sel,
                                               bg=self.colors["danger"], hover_bg=self.colors["danger_hover"],
                                               width=180, height=48, radius=12)
        btn_eliminar.pack(side="left", padx=10)

    def reproductor_grafico(self, archivo):
        self.limpiar_panel()

        with np.load(archivo, allow_pickle=False) as d:
            fs = int(d["fs"])
            senal = np.asarray(d["normalizada"], dtype=np.float64)
            vocal = str(d["vocal"].item()).upper()
            ini, fin = ventana_guardada(d, len(senal) / fs)

        main_frame = tk.Frame(self.right, bg=self.colors["bg_card"])
        main_frame.pack(fill="both", expand=True, padx=30, pady=20)

        self.agregar_boton_volver(main_frame)

        tk.Label(main_frame, text=f"Análisis de la vocal '{vocal}'", bg=self.colors["bg_card"],
                 fg=self.colors["text_primary"], font=("Segoe UI", 18, "bold")).pack(pady=(10, 5))
        tk.Label(main_frame, text=archivo.name, bg=self.colors["bg_card"],
                 fg=self.colors["text_muted"], font=self.font_small).pack(pady=(0, 15))

        fig, ax = plt.subplots(figsize=(10, 4), facecolor=self.colors["bg_card"])
        ax.set_facecolor(self.colors["bg_app"])
        t = np.arange(len(senal)) / fs
        ax.plot(t, senal, color=self.colors["accent"], linewidth=1)
        ax.axvspan(ini, fin, color=self.colors["warning"], alpha=0.2, label="Ventana de Análisis")
        ax.set_title("Forma de Onda", color=self.colors["text_primary"], pad=15, fontweight="bold", fontsize=12)
        ax.tick_params(colors=self.colors["text_secondary"])
        legend = ax.legend(loc="upper right", facecolor=self.colors["bg_app"], edgecolor=self.colors["border_light"])
        for text in legend.get_texts():
            text.set_color(self.colors["text_primary"])
        for spine in ax.spines.values():
            spine.set_color(self.colors["border_light"])
        ax.grid(True, color=self.colors["border_light"], linestyle="--", alpha=0.3)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        ctrl = tk.Frame(main_frame, bg=self.colors["bg_card"], pady=15)
        ctrl.pack()

        def play():
            self._reproducir_archivo(archivo, parent_widget=main_frame)

        def stop():
            self.detener_audio()

        def eliminar():
            self.eliminar_archivos([archivo], callback=self.cargar_lista_reproductor)

        btn_play = crear_boton_redondeado(ctrl, "▶️  Reproducir", play,
                                            bg=self.colors["success"], hover_bg="#059669",
                                            width=160, height=44, radius=10)
        btn_play.pack(side="left", padx=10)

        btn_stop = crear_boton_redondeado(ctrl, "⏹  Detener", stop,
                                           bg=self.colors["bg_hover"], hover_bg=self.colors["border_light"],
                                           width=140, height=44, radius=10)
        btn_stop.pack(side="left", padx=10)

        btn_del = crear_boton_redondeado(ctrl, "🗑  Eliminar", eliminar,
                                          bg=self.colors["danger"], hover_bg=self.colors["danger_hover"],
                                          width=150, height=44, radius=10)
        btn_del.pack(side="left", padx=10)

    def accion_diagnostico(self):
        self.limpiar_panel()
        self.set_status("Evaluando hardware...", self.colors["warning"])

        main_frame = tk.Frame(self.right, bg=self.colors["bg_card"])
        main_frame.pack(fill="both", expand=True, padx=30, pady=20)

        self.agregar_boton_volver(main_frame)

        tk.Label(main_frame, text="🩺  Diagnóstico de Micrófono", bg=self.colors["bg_card"],
                 fg=self.colors["text_primary"], font=("Segoe UI", 18, "bold")).pack(pady=(10, 5))
        tk.Label(main_frame, text="Capturando 4 segundos de audio para verificar la señal de entrada...",
                 bg=self.colors["bg_card"], fg=self.colors["text_muted"], font=self.font_body).pack(pady=(0, 20))
        self.root.update()

        try:
            x = grabar_audio(4, FS)
        except Exception as e:
            return messagebox.showerror("Error de Hardware", str(e))

        fig, ax = plt.subplots(figsize=(10, 4), facecolor=self.colors["bg_card"])
        ax.set_facecolor(self.colors["bg_app"])
        ax.plot(np.arange(len(x))/FS, x, color=self.colors["text_secondary"], linewidth=0.8)
        ax.set_title("Señal de Entrada (4 segundos)", color=self.colors["text_primary"],
                     pad=15, fontweight="bold", fontsize=12)
        ax.set_xlabel("Tiempo (s)", color=self.colors["text_muted"], fontweight="bold")
        ax.set_ylabel("Amplitud", color=self.colors["text_muted"], fontweight="bold")
        ax.tick_params(colors=self.colors["text_secondary"])
        for spine in ax.spines.values():
            spine.set_color(self.colors["border_light"])
        ax.grid(True, color=self.colors["border_light"], linestyle="--", alpha=0.3)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = AppVocales(root)
    root.mainloop()