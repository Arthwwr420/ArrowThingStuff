"""
╔══════════════════════════════════════════════════════════════╗
║  INFERENCIA EN TIEMPO REAL — Cámara                         ║
║  Requiere: modelo_flechas.joblib (generado en entrenamiento) ║
╠══════════════════════════════════════════════════════════════╣
║  pip install opencv-python scikit-image joblib numpy        ║
║                                                              ║
║  Controles:                                                  ║
║    Q / ESC  → salir                                         ║
║    S        → guardar captura actual como PNG               ║
║    P        → pausar / reanudar                             ║
║    H        → mostrar/ocultar visualización HOG             ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import joblib
import numpy as np
from skimage.filters  import threshold_otsu
from skimage.feature  import hog
from skimage          import morphology, exposure
from datetime         import datetime


# ══════════════════════════════════════════════════════════════
# CARGA DEL MODELO
# ══════════════════════════════════════════════════════════════
print(" Cargando modelo...")
datos = joblib.load('modelo_flechas.joblib')

pipeline    = datos['pipeline']
HOG_PARAMS  = datos['hog_params']
IMG_SIZE    = datos['img_size']
CLASES      = datos['clases']
metricas    = datos['metricas']

print(f"    Modelo cargado")
print(f"   Accuracy entrenamiento : {metricas['accuracy']}")
print(f"   F1-score               : {metricas['f1']}")
print(f"   ROC AUC                : {metricas['roc_auc']}")


# ══════════════════════════════════════════════════════════════
# PREPROCESAMIENTO (igual que en entrenamiento)
# ══════════════════════════════════════════════════════════════
def preprocess_otsu(img_gray_float):
    """img_gray_float: float32 [0,1] de tamaño IMG_SIZE×IMG_SIZE"""
    img_eq = exposure.equalize_adapthist(img_gray_float, clip_limit=0.03)
    thresh = threshold_otsu(img_eq)
    binary = (img_eq > thresh).astype(np.float32)
    if binary.mean() > 0.5:
        binary = 1.0 - binary
    bool_mask = binary.astype(bool)
    cleaned   = morphology.remove_small_objects(bool_mask, min_size=20)
    cleaned   = morphology.remove_small_holes(cleaned, area_threshold=50)
    return cleaned.astype(np.float32)


def extract_features(img_proc):
    """Extrae HOG + asimetría de una imagen ya preprocesada."""
    # HOG
    hog_f, _ = hog(img_proc, **{**HOG_PARAMS, 'visualize': True})

    # Asimetría manual (misma lógica que entrenamiento)
    dark     = img_proc.copy()
    col_proj = dark.mean(axis=0)
    half     = IMG_SIZE // 2
    mass_L   = dark[:, :half].mean()
    mass_R   = dark[:, half:].mean()
    lr_diff  = mass_L - mass_R
    lr_ratio = mass_L / (mass_R + 1e-9)
    cols     = np.arange(IMG_SIZE, dtype=float)
    total    = dark.sum() + 1e-9
    cx       = (dark.sum(axis=0) @ cols) / total / IMG_SIZE
    proj_asym = col_proj[:half].mean() - col_proj[half:].mean()
    grad_asym = (np.gradient(col_proj)[:half].mean() -
                 np.gradient(col_proj)[half:].mean())
    tl = dark[:half, :half].mean(); tr = dark[:half, half:].mean()
    bl = dark[half:, :half].mean(); br = dark[half:, half:].mean()
    quad_asym = (tl + bl) - (tr + br)

    asym_f = np.array([lr_diff, lr_ratio, cx, proj_asym,
                       grad_asym, quad_asym, tl, tr, bl, br],
                      dtype=np.float32)
    return np.concatenate([hog_f, asym_f])


def inferir_frame(frame_bgr):
    """
    Recibe un frame BGR de OpenCV.
    Devuelve: clase (str), confianza (float), img_proc (np.array), hog_vis (np.array)
    """
    # Convertir a escala de grises y redimensionar
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray  = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray_f = gray.astype(np.float32) / 255.0

    # Preprocesar
    img_proc = preprocess_otsu(gray_f)

    # HOG visual
    _, hog_vis = hog(img_proc, **{**HOG_PARAMS, 'visualize': True})
    hog_eq     = exposure.rescale_intensity(hog_vis, in_range=(0, 0.08))

    # Extraer features y predecir
    feat    = extract_features(img_proc).reshape(1, -1)
    pred    = pipeline.predict(feat)[0]
    proba   = pipeline.predict_proba(feat)[0]
    clase   = CLASES[pred]
    conf    = proba[pred]

    return clase, conf, img_proc, hog_eq


# ══════════════════════════════════════════════════════════════
# UTILIDADES DE OVERLAY EN OPENCV
# ══════════════════════════════════════════════════════════════

# Colores BGR
C_BLUE   = (203, 163, 31)    # azul
C_ORANGE = (38, 124, 224)    # naranja
C_WHITE  = (255, 255, 255)
C_BLACK  = (0, 0, 0)
C_GREEN  = (87, 139, 46)
C_GRAY   = (140, 140, 140)
C_RED    = (43, 57, 192)
C_DARK   = (74, 59, 0)


def barra_confianza(frame, conf, clase, x, y, ancho=220, alto=22):
    """Dibuja una barra de confianza en el frame."""
    # Fondo
    cv2.rectangle(frame, (x, y), (x+ancho, y+alto), C_GRAY, -1)
    # Relleno proporcional
    color_barra = C_ORANGE if clase == 'Derecha' else C_BLUE
    fill = int(ancho * conf)
    cv2.rectangle(frame, (x, y), (x+fill, y+alto), color_barra, -1)
    # Borde
    cv2.rectangle(frame, (x, y), (x+ancho, y+alto), C_WHITE, 1)
    # Texto
    txt = f'{conf*100:.1f}%'
    cv2.putText(frame, txt, (x + ancho//2 - 20, y + alto - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHITE, 1, cv2.LINE_AA)


def flecha_overlay(frame, clase, cx, cy, radio=45):
    """Dibuja una flecha indicadora encima del frame."""
    color = C_ORANGE if clase == 'Derecha' else C_BLUE
    grosor = 3
    if clase == 'Derecha':
        # →
        cv2.arrowedLine(frame,
                        (cx - radio, cy),
                        (cx + radio, cy),
                        color, grosor, tipLength=0.4)
    else:
        # ←
        cv2.arrowedLine(frame,
                        (cx + radio, cy),
                        (cx - radio, cy),
                        color, grosor, tipLength=0.4)


def panel_info(frame, clase, conf, pausado, show_hog, capturas):
    """Panel semitransparente con info en la esquina superior izquierda."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (290, 200), C_BLACK, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    color_clase = C_ORANGE if clase == 'Derecha' else C_BLUE
    simbolo     = '>>' if clase == 'Derecha' else '<<'

    cv2.putText(frame, f'{simbolo}  {clase}',
                (12, 38), cv2.FONT_HERSHEY_DUPLEX, 1.1, color_clase, 2, cv2.LINE_AA)

    barra_confianza(frame, conf, clase, 12, 52)

    estado = 'PAUSADO' if pausado else 'EN VIVO'
    cv2.putText(frame, estado,
                (12, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                C_RED if pausado else C_GREEN, 1, cv2.LINE_AA)

    cv2.putText(frame, f'HOG: {"ON" if show_hog else "OFF"}',
                (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHITE, 1, cv2.LINE_AA)

    cv2.putText(frame, f'Capturas: {capturas}',
                (180, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHITE, 1, cv2.LINE_AA)

    cv2.putText(frame, 'Q/ESC salir  S guardar',
                (12, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1)
    cv2.putText(frame, 'P pausar  H HOG visual',
                (12, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1)

    # Flecha decorativa grande
    flecha_overlay(frame, clase, w - 80, 60)


def insertar_miniatura(frame, img_small, x, y, size=96, titulo=''):
    """
    Inserta una miniatura en escala de grises o mapa de color
    en la esquina indicada del frame principal.
    """
    h, w = frame.shape[:2]

    # Redimensionar la miniatura
    if img_small.dtype != np.uint8:
        img_disp = (img_small * 255).clip(0, 255).astype(np.uint8)
    else:
        img_disp = img_small.copy()

    img_disp = cv2.resize(img_disp, (size, size))

    # Convertir a BGR si es gris
    if img_disp.ndim == 2:
        img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)

    # Pegar
    x2, y2 = x + size, y + size
    if y2 <= h and x2 <= w:
        frame[y:y2, x:x2] = img_disp
        cv2.rectangle(frame, (x, y), (x2, y2), C_WHITE, 1)
        if titulo:
            cv2.putText(frame, titulo, (x+2, y2+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_WHITE, 1)


def hog_a_bgr(hog_eq):
    """Convierte la imagen HOG float [0,1] a BGR coloreada con magma."""
    hog_uint8 = (hog_eq * 255).clip(0, 255).astype(np.uint8)
    hog_color = cv2.applyColorMap(hog_uint8, cv2.COLORMAP_MAGMA)
    return hog_color


# ══════════════════════════════════════════════════════════════
# BUCLE PRINCIPAL DE CÁMARA
# ══════════════════════════════════════════════════════════════
def main():
    # Intentar abrir cámara (índice 0 = cámara por defecto)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" No se pudo abrir la cámara.")
        print("   Intenta cambiar el índice: cv2.VideoCapture(1)")
        return

    # Ajustar resolución (opcional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"\n📷  Cámara abierta  ({int(cap.get(3))}×{int(cap.get(4))} @ {fps:.0f} fps)")
    print("   Controles: Q/ESC salir | S guardar | P pausar | H HOG\n")

    # Estado
    pausado    = False
    show_hog   = False
    capturas   = 0
    clase      = '---'
    conf       = 0.0
    img_proc   = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    hog_vis    = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # Región de interés: cuadrado central de detección
    ROI_FRAC = 0.55   # 55% del tamaño del frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠  No se pudo leer el frame.")
            break

        h, w = frame.shape[:2]

        # ── Definir ROI central ───────────────────────────────
        roi_w = int(w * ROI_FRAC)
        roi_h = int(h * ROI_FRAC)
        rx1   = (w - roi_w) // 2
        ry1   = (h - roi_h) // 2
        rx2   = rx1 + roi_w
        ry2   = ry1 + roi_h

        # ── Inferencia (solo si no está pausado) ─────────────
        if not pausado:
            roi_frame = frame[ry1:ry2, rx1:rx2]
            clase, conf, img_proc, hog_vis = inferir_frame(roi_frame)

        # ── Color del borde según predicción ─────────────────
        color_borde = C_ORANGE if clase == 'Derecha' else C_BLUE
        grosor_ok   = 3 if conf >= 0.75 else 1

        # ── Dibujar ROI ───────────────────────────────────────
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color_borde, grosor_ok)

        # Línea central vertical de referencia (dentro del ROI)
        cx_roi = (rx1 + rx2) // 2
        cv2.line(frame, (cx_roi, ry1+4), (cx_roi, ry2-4),
                 (180, 180, 180), 1, cv2.LINE_AA)

        # ── Panel de info ─────────────────────────────────────
        panel_info(frame, clase, conf, pausado, show_hog, capturas)

        # ── Miniaturas (esquina inferior derecha) ─────────────
        mini_size = 96
        margin    = 8
        x_mini    = w - mini_size - margin
        y_mini    = h - mini_size - margin - 16

        insertar_miniatura(frame, img_proc, x_mini, y_mini,
                           size=mini_size, titulo='Binarizado')

        if show_hog:
            hog_bgr = hog_a_bgr(hog_vis)
            insertar_miniatura(frame, hog_bgr,
                               x_mini - mini_size - margin, y_mini,
                               size=mini_size, titulo='HOG')

        # ── Etiqueta de confianza sobre el ROI ───────────────
        etiqueta = f'{clase}  {conf*100:.0f}%'
        (tw, th), _ = cv2.getTextSize(etiqueta,
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        tx = rx1 + (roi_w - tw) // 2
        ty = ry2 + 22
        cv2.putText(frame, etiqueta, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_borde, 2, cv2.LINE_AA)

        # ── Mostrar ───────────────────────────────────────────
        cv2.imshow('¿Izquierda o Derecha? — Q para salir', frame)

        # ── Teclado ───────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):        # Q o ESC → salir
            break

        elif key in (ord('s'), ord('S')):           # S → guardar
            ts   = datetime.now().strftime('%H%M%S_%f')[:10]
            name = f'captura_{clase[0]}_{ts}.png'
            cv2.imwrite(name, frame)
            capturas += 1
            print(f"    Guardado: {name}  ({clase} {conf*100:.1f}%)")

        elif key in (ord('p'), ord('P')):           # P → pausar
            pausado = not pausado
            print(f"   {'⏸  Pausado' if pausado else '▶  Reanudado'}")

        elif key in (ord('h'), ord('H')):           # H → HOG visual
            show_hog = not show_hog

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n Sesión terminada.  Capturas guardadas: {capturas}")


if __name__ == '__main__':
    main()