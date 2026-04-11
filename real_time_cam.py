"""
╔══════════════════════════════════════════════════════════════╗
║  INFERENCIA EN TIEMPO REAL — Cámara  (v3)                   ║
║  Fix: votación de polaridad + modo captura para reentrenar  ║
╠══════════════════════════════════════════════════════════════╣
║  Controles:                                                  ║
║    Q / ESC  → salir                                         ║
║    S        → guardar captura (sin etiquetar)               ║
║    L        → capturar como IZQUIERDA (para reentrenar)     ║
║    R        → capturar como DERECHA   (para reentrenar)     ║
║    P        → pausar / reanudar                             ║
║    D        → debug (ventana de preprocesamiento)           ║
║    H        → HOG visual                                    ║
║    +  /  -  → agrandar / achicar el ROI                     ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import joblib
import numpy as np
import os
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.feature import hog
from skimage         import morphology, exposure
from datetime        import datetime


# ══════════════════════════════════════════════════════════════
# CARGA DEL MODELO
# ══════════════════════════════════════════════════════════════
print(" Cargando modelo...")
datos      = joblib.load('modelo_flechas.joblib')
pipeline   = datos['pipeline']
HOG_PARAMS = datos['hog_params']
IMG_SIZE   = datos['img_size']
CLASES     = datos['clases']           # {0: 'Izquierda', 1: 'Derecha'}
metricas   = datos['metricas']

assert list(pipeline.classes_) == [0, 1], \
    f"Orden inesperado de clases: {pipeline.classes_}"

print(f"    Modelo listo  |  Accuracy: {metricas['accuracy']}  F1: {metricas['f1']}")

os.makedirs(os.path.join('cam_data', 'Left'),  exist_ok=True)
os.makedirs(os.path.join('cam_data', 'Right'), exist_ok=True)


# ══════════════════════════════════════════════════════════════
# PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════
def binarizar(img_float: np.ndarray) -> np.ndarray:
    """
    Binariza con Otsu + Sauvola combinados.
    NO normaliza polaridad (eso lo hace polarity_vote).
    """
    img_u8  = (img_float * 255).clip(0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_u8, (3, 3), 0).astype(np.float32) / 255.0

    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    eq      = clahe.apply((blurred * 255).astype(np.uint8)).astype(np.float32) / 255.0

    bin_o   = (eq > threshold_otsu(eq)).astype(np.float32)
    bin_s   = (eq > threshold_sauvola(eq, window_size=15)).astype(np.float32)
    combined = ((bin_o + bin_s) > 0.5).astype(np.float32)

    mask    = morphology.remove_small_objects(combined.astype(bool), min_size=15)
    mask    = morphology.remove_small_holes(mask, area_threshold=40)
    return mask.astype(np.float32)


def extract_features(img_proc: np.ndarray) -> np.ndarray:
    """HOG + features de asimetría (idéntico al entrenamiento)."""
    hog_f, _ = hog(img_proc, **{**HOG_PARAMS, 'visualize': True})

    dark      = img_proc.copy()
    col_proj  = dark.mean(axis=0)
    half      = IMG_SIZE // 2
    mass_L    = dark[:, :half].mean()
    mass_R    = dark[:, half:].mean()
    lr_diff   = mass_L - mass_R
    lr_ratio  = mass_L / (mass_R + 1e-9)
    cols      = np.arange(IMG_SIZE, dtype=float)
    total     = dark.sum() + 1e-9
    cx        = (dark.sum(axis=0) @ cols) / total / IMG_SIZE
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


def polarity_vote(img_bin: np.ndarray):
    """
    FIX PRINCIPAL: prueba polaridad normal e invertida.
    Gana la que produce mayor confianza en el modelo.

    Resuelve el problema de que Otsu invierte la flecha cuando
    la pantalla tiene fondo claro o contraste inesperado.

    Returns: (prob_derecha, nombre_polaridad, confianza, img_ganadora)
    """
    mejor_conf = -1.0
    mejor_prob = 0.5
    mejor_pol  = 'normal'
    mejor_img  = img_bin

    for nombre, candidato in [('normal', img_bin), ('invertida', 1.0 - img_bin)]:
        media = candidato.mean()
        if media < 0.02 or media > 0.98:   # imagen casi vacía o llena → skip
            continue
        feat = extract_features(candidato).reshape(1, -1)
        prob = pipeline.predict_proba(feat)[0]   # [P(Izq), P(Der)]
        conf = float(np.max(prob))

        if conf > mejor_conf:
            mejor_conf = conf
            mejor_prob = float(prob[1])          # P(Derecha)
            mejor_pol  = nombre
            mejor_img  = candidato

    return mejor_prob, mejor_pol, mejor_conf, mejor_img


# ══════════════════════════════════════════════════════════════
# SUAVIZADOR TEMPORAL
# ══════════════════════════════════════════════════════════════
class Suavizador:
    def __init__(self, ventana=10):
        self.ventana = ventana
        self.buf     = []

    def push(self, p):
        self.buf.append(p)
        if len(self.buf) > self.ventana:
            self.buf.pop(0)

    @property
    def prob(self):
        return float(np.mean(self.buf)) if self.buf else 0.5

    @property
    def clase(self):
        return CLASES[1 if self.prob >= 0.5 else 0]

    @property
    def conf(self):
        p = self.prob
        return p if p >= 0.5 else 1 - p

    @property
    def estab(self):
        if len(self.buf) < 3:
            return 0.0
        return float(np.clip(1.0 - np.std(self.buf) * 5, 0, 1))


# ══════════════════════════════════════════════════════════════
# INFERENCIA
# ══════════════════════════════════════════════════════════════
def inferir(roi_bgr, suav):
    gray_f  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray_f  = cv2.resize(gray_f, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    img_bin = binarizar(gray_f)

    prob_raw, pol, conf_raw, img_proc = polarity_vote(img_bin)

    if img_proc.mean() < 0.02 or img_proc.mean() > 0.98:
        img_proc = gray_f

    suav.push(prob_raw)

    _, hog_vis = hog(img_proc, **{**HOG_PARAMS, 'visualize': True})
    hog_eq     = exposure.rescale_intensity(hog_vis, in_range=(0, 0.08))

    return suav.clase, suav.conf, prob_raw, pol, conf_raw, img_bin, img_proc, hog_eq


# ══════════════════════════════════════════════════════════════
# OVERLAY
# ══════════════════════════════════════════════════════════════
CB  = (203, 163,  31)
CO  = ( 38, 124, 224)
CW  = (255, 255, 255)
CK  = (  0,   0,   0)
CGR = (140, 140, 140)
CR  = ( 43,  57, 192)
CY  = (  0, 220, 220)
CGN = ( 87, 139,  46)


def draw_barra_prob(frame, prob_der, x, y, w=240, h=18):
    mid = x + w // 2
    cv2.rectangle(frame, (x, y), (x+w, y+h), (55, 55, 55), -1)
    cv2.rectangle(frame, (mid - int((1-prob_der)*w//2), y), (mid, y+h), CB, -1)
    cv2.rectangle(frame, (mid, y), (mid + int(prob_der*w//2), y+h), CO, -1)
    cv2.line(frame, (mid, y), (mid, y+h), CW, 1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), CGR, 1)
    cv2.putText(frame, f'IZQ {(1-prob_der)*100:.0f}%',
                (x+2, y+h-4), cv2.FONT_HERSHEY_SIMPLEX, 0.31, CW, 1)
    cv2.putText(frame, f'{prob_der*100:.0f}% DER',
                (x+w-62, y+h-4), cv2.FONT_HERSHEY_SIMPLEX, 0.31, CW, 1)


def draw_panel(frame, clase, conf, prob_raw, pol, estab,
               pausa, hog_on, dbg, nL, nR):
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (310, 205), CK, -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)

    col = CO if clase == 'Derecha' else CB
    sim = '>>' if clase == 'Derecha' else '<<'
    fw  = frame.shape[1]

    if conf >= 0.65:
        cv2.putText(frame, f'{sim}  {clase}',
                    (12, 36), cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, '?  Baja confianza',
                    (12, 36), cv2.FONT_HERSHEY_DUPLEX, 0.8, CGR, 2, cv2.LINE_AA)

    cv2.putText(frame, f'Conf: {conf*100:.1f}%   Polaridad: {pol}',
                (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.37, col, 1)

    draw_barra_prob(frame, prob_raw, 12, 66)

    # Barra de estabilidad
    cv2.rectangle(frame, (12, 98), (252, 108), (40, 40, 40), -1)
    ec = CGN if estab > 0.6 else (CY if estab > 0.3 else CR)
    cv2.rectangle(frame, (12, 98), (12+int(estab*240), 108), ec, -1)
    cv2.rectangle(frame, (12, 98), (252, 108), CGR, 1)
    cv2.putText(frame, f'Estab {estab*100:.0f}%',
                (12, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.30, CGR, 1)

    cv2.putText(frame, f'L:{nL} izq guardadas   R:{nR} der guardadas',
                (12, 127), cv2.FONT_HERSHEY_SIMPLEX, 0.37, CY, 1)

    estado = ('PAUSADO ' if pausa else '') + ('HOG ' if hog_on else '') + ('DBG' if dbg else '')
    cv2.putText(frame, estado or 'EN VIVO',
                (12, 147), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                CR if pausa else CGN, 1)

    cv2.putText(frame, 'L=guardar izq  R=guardar der  D=debug  H=hog  +/-=roi  Q=salir',
                (12, 167), cv2.FONT_HERSHEY_SIMPLEX, 0.28, CGR, 1)
    cv2.putText(frame, 'P=pausar  S=captura  | retrain.py cuando L>=20 y R>=20',
                (12, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.28, CGR, 1)

    if conf >= 0.65:
        ax = fw - 70
        pts = ((ax-40, 50), (ax+40, 50)) if clase == 'Derecha' else ((ax+40, 50), (ax-40, 50))
        cv2.arrowedLine(frame, pts[0], pts[1], col, 4, tipLength=0.4)


def insertar_mini(frame, img, x, y, size=90, label='', cmap=None):
    if img.dtype != np.uint8:
        disp = (img * 255).clip(0, 255).astype(np.uint8)
    else:
        disp = img.copy()
    disp = cv2.resize(disp, (size, size))
    if cmap is not None:
        disp = cv2.applyColorMap(disp, cmap)
    elif disp.ndim == 2:
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    fh, fw = frame.shape[:2]
    if y+size <= fh and x+size <= fw:
        frame[y:y+size, x:x+size] = disp
        cv2.rectangle(frame, (x, y), (x+size, y+size), CW, 1)
        if label:
            cv2.putText(frame, label, (x+2, y+size+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, CW, 1)


def draw_debug(roi_bgr, img_bin, img_proc):
    s  = 150
    c  = np.zeros((s+30, s*3+20, 3), np.uint8)
    g  = cv2.resize(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY), (s, s))
    c[0:s, 0:s]          = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    b  = cv2.resize((img_bin*255).astype(np.uint8), (s, s))
    c[0:s, s+10:2*s+10]  = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    p  = cv2.resize((img_proc*255).astype(np.uint8), (s, s))
    c[0:s, 2*s+20:3*s+20]= cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
    for i, txt in enumerate(['ROI gris', 'Binarizado', 'Polarity vote']):
        cv2.putText(c, txt, (i*(s+10)+2, s+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    CY if txt == 'Polarity vote' else CW, 1)
    cv2.imshow('DEBUG', c)


# ══════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ══════════════════════════════════════════════════════════════
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" No se pudo abrir la cámara.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"\n📷  Cámara lista  {int(cap.get(3))}×{int(cap.get(4))}")
    print("   L = guardar frame como IZQUIERDA")
    print("   R = guardar frame como DERECHA")
    print("   Junta ≥20 de cada clase → ejecuta retrain.py\n")

    suav     = Suavizador(ventana=10)
    pausa    = False;  hog_on = False;  dbg = False
    roi_frac = 0.55
    nL       = len(os.listdir(os.path.join('cam_data', 'Left')))
    nR       = len(os.listdir(os.path.join('cam_data', 'Right')))

    clase    = '---';  conf    = 0.0;   prob_raw = 0.5
    pol      = 'normal'; estab = 0.0
    img_bin  = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    img_proc = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    hog_vis  = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    roi_bgr  = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]
        rw = int(fw*roi_frac);  rh = int(fh*roi_frac)
        rx1 = (fw-rw)//2;      ry1 = (fh-rh)//2
        rx2 = rx1+rw;           ry2 = ry1+rh

        if not pausa:
            roi_bgr  = frame[ry1:ry2, rx1:rx2].copy()
            (clase, conf, prob_raw, pol,
             _, img_bin, img_proc, hog_vis) = inferir(roi_bgr, suav)
            estab = suav.estab

        col_borde = (CO if clase=='Derecha' else CB) if conf>=0.75 else \
                    (CY if conf>=0.60 else CGR)
        gb = 3 if conf>=0.75 else (2 if conf>=0.60 else 1)

        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), col_borde, gb)
        cv2.line(frame, ((rx1+rx2)//2, ry1+6), ((rx1+rx2)//2, ry2-6),
                 (160,160,160), 1, cv2.LINE_AA)

        draw_panel(frame, clase, conf, prob_raw, pol, estab,
                   pausa, hog_on, dbg, nL, nR)

        xs = fw-90-8;  ys = fh-90-8-14
        insertar_mini(frame, img_proc, xs, ys, label='Proc.')
        if hog_on:
            insertar_mini(frame,
                          (hog_vis*255).clip(0,255).astype(np.uint8),
                          xs-98, ys, label='HOG', cmap=cv2.COLORMAP_MAGMA)

        if conf >= 0.60:
            etiq = f'{clase}  {conf*100:.0f}%'
            tw   = cv2.getTextSize(etiq, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
            cv2.putText(frame, etiq,
                        (rx1+(rw-tw)//2, ry2+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col_borde, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'acerca la flecha al cuadro',
                        (rx1+8, ry2+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, CGR, 1)

        if dbg and roi_bgr is not None:
            draw_debug(roi_bgr, img_bin, img_proc)
        elif not dbg:
            try: cv2.destroyWindow('DEBUG')
            except: pass

        cv2.imshow('Izquierda o Derecha v3', frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('l'), ord('L')) and roi_bgr is not None:
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            ts   = datetime.now().strftime('%H%M%S_%f')[:12]
            cv2.imwrite(os.path.join('cam_data', 'Left', f'cam_{ts}.png'), gray)
            nL   = len(os.listdir(os.path.join('cam_data', 'Left')))
            print(f"   ← Izquierda guardada ({nL})")
        elif key in (ord('r'), ord('R')) and roi_bgr is not None:
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            ts   = datetime.now().strftime('%H%M%S_%f')[:12]
            cv2.imwrite(os.path.join('cam_data', 'Right', f'cam_{ts}.png'), gray)
            nR   = len(os.listdir(os.path.join('cam_data', 'Right')))
            print(f"   → Derecha guardada ({nR})")
        elif key in (ord('s'), ord('S')):
            ts = datetime.now().strftime('%H%M%S_%f')[:10]
            cv2.imwrite(f'captura_{ts}.png', frame)
            print(f"   📸  captura_{ts}.png")
        elif key in (ord('p'), ord('P')):
            pausa = not pausa
        elif key in (ord('h'), ord('H')):
            hog_on = not hog_on
        elif key in (ord('d'), ord('D')):
            dbg = not dbg
        elif key == ord('+'):
            roi_frac = min(roi_frac+0.05, 0.92)
        elif key == ord('-'):
            roi_frac = max(roi_frac-0.05, 0.20)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n Fin.   cam_data/Left: {nL}   cam_data/Right: {nR}")
    if nL >= 20 and nR >= 20:
        print("    Suficientes muestras. Ejecuta: python retrain.py")


if __name__ == '__main__':
    main()