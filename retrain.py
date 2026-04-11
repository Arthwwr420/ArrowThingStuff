"""
╔══════════════════════════════════════════════════════════════╗
║  retrain.py — Reentrenamiento con muestras de cámara        ║
║                                                              ║
║  Combina el dataset original con las capturas de cam_data/  ║
║  y guarda el modelo mejorado como modelo_flechas.joblib     ║
║                                                              ║
║  Uso: python retrain.py                                      ║
║  (ejecutar después de capturar ≥20 muestras de cada clase)  ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import joblib
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.feature import hog
from skimage         import morphology, exposure
import cv2

from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics         import accuracy_score, f1_score, classification_report
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline

import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)


# ══════════════════════════════════════════════════════════════
# CARGAR MODELO ANTERIOR (para reusar parámetros HOG)
# ══════════════════════════════════════════════════════════════
print(" Cargando configuración del modelo anterior...")
datos_ant  = joblib.load('modelo_flechas.joblib')
HOG_PARAMS = datos_ant['hog_params']
IMG_SIZE   = datos_ant['img_size']
print(f"   IMG_SIZE={IMG_SIZE}  HOG={HOG_PARAMS}")


# ══════════════════════════════════════════════════════════════
# PARÁMETROS DE CARPETAS
# ══════════════════════════════════════════════════════════════
VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Dataset original
ORIG_LEFT  = os.path.join('dataset', 'Left')
ORIG_RIGHT = os.path.join('dataset', 'Right')

# Capturas de cámara
CAM_LEFT   = os.path.join('cam_data', 'Left')
CAM_RIGHT  = os.path.join('cam_data', 'Right')


# ══════════════════════════════════════════════════════════════
# PREPROCESAMIENTO (mismo que real_time_cam.py)
# ══════════════════════════════════════════════════════════════
def binarizar(img_float: np.ndarray) -> np.ndarray:
    img_u8  = (img_float * 255).clip(0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_u8, (3, 3), 0).astype(np.float32) / 255.0
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    eq      = clahe.apply((blurred*255).astype(np.uint8)).astype(np.float32)/255.0
    bin_o   = (eq > threshold_otsu(eq)).astype(np.float32)
    bin_s   = (eq > threshold_sauvola(eq, window_size=15)).astype(np.float32)
    combined = ((bin_o + bin_s) > 0.5).astype(np.float32)
    mask    = morphology.remove_small_objects(combined.astype(bool), min_size=15)
    mask    = morphology.remove_small_holes(mask, area_threshold=40)
    return mask.astype(np.float32)


def normalizar_polaridad(img_bin: np.ndarray) -> np.ndarray:
    """Para dataset estático: normaliza polaridad por media."""
    return 1.0 - img_bin if img_bin.mean() > 0.55 else img_bin


def extract_features(img_proc: np.ndarray) -> np.ndarray:
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
                       grad_asym, quad_asym, tl, tr, bl, br], dtype=np.float32)
    return np.concatenate([hog_f, asym_f])


# ══════════════════════════════════════════════════════════════
# CARGA DE IMÁGENES
# ══════════════════════════════════════════════════════════════
def load_folder(folder, label, es_camara=False):
    """
    Carga imágenes de una carpeta.
    es_camara=True: aplica binarizar() directamente sin normalizar polaridad
                    (las capturas de cámara ya fueron guardadas en el estado correcto)
    es_camara=False: aplica binarizar() + normalizar_polaridad()
    """
    feats, labels = [], []
    if not os.path.isdir(folder):
        return feats, labels

    files = [f for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in VALID_EXT]

    for fname in files:
        path = os.path.join(folder, fname)
        try:
            img = (Image.open(path)
                   .convert('L')
                   .resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS))
            arr = np.array(img, dtype=np.float32) / 255.0

            img_bin = binarizar(arr)

            if es_camara:
                # Capturas de cámara: usar polarity_vote simplificado
                # (normalizar si la mayoría de píxeles son fondo)
                img_proc = img_bin if img_bin.mean() <= 0.55 else 1.0 - img_bin
            else:
                img_proc = normalizar_polaridad(img_bin)

            feat = extract_features(img_proc)
            feats.append(feat)
            labels.append(label)
        except Exception as e:
            print(f"   ⚠  {fname}: {e}")

    return feats, labels


print("\n Cargando imágenes...")

# Dataset original
fo_L, lo_L = load_folder(ORIG_LEFT,  0, es_camara=False)
fo_R, lo_R = load_folder(ORIG_RIGHT, 1, es_camara=False)

# Capturas de cámara
fc_L, lc_L = load_folder(CAM_LEFT,  0, es_camara=True)
fc_R, lc_R = load_folder(CAM_RIGHT, 1, es_camara=True)

n_orig = len(fo_L) + len(fo_R)
n_cam  = len(fc_L) + len(fc_R)

print(f"   Dataset original : {len(fo_L)} izq + {len(fo_R)} der = {n_orig}")
print(f"   Capturas cámara  : {len(fc_L)} izq + {len(fc_R)} der = {n_cam}")

if n_cam == 0:
    print("\n No hay capturas de cámara en cam_data/.")
    print("   Ejecuta real_time_cam.py y usa L/R para capturar muestras.")
    print("   Reentrenando solo con el dataset original...\n")

all_feats  = fo_L + fo_R + fc_L + fc_R
all_labels = lo_L + lo_R + lc_L + lc_R

X = np.array(all_feats,  dtype=np.float32)
y = np.array(all_labels, dtype=int)
total = len(y)
print(f"\n   Total : {total} muestras  "
      f"({int((y==0).sum())} izq / {int((y==1).sum())} der)")


# ══════════════════════════════════════════════════════════════
# ENTRENAMIENTO CON GRIDSEARCHCV
# ══════════════════════════════════════════════════════════════
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n GridSearchCV (esto puede tardar ~60 s)...")

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(max_iter=2000, random_state=42,
                                  class_weight='balanced')),
])
param_grid = {
    'clf__C'      : [0.001, 0.01, 0.1, 1, 10, 100],
    'clf__solver' : ['lbfgs', 'liblinear'],
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs  = GridSearchCV(pipe, param_grid, cv=skf,
                   scoring='f1', n_jobs=-1, verbose=0)
gs.fit(X_tr, y_tr)

best_pipe = gs.best_estimator_
y_pred    = best_pipe.predict(X_te)

acc = accuracy_score(y_te, y_pred)
f1  = f1_score(y_te, y_pred, zero_division=0)
roc = gs.best_score_

print(f"\n   Mejor C      : {gs.best_params_['clf__C']}")
print(f"   Mejor solver : {gs.best_params_['clf__solver']}")
print(f"   CV F1 (train): {gs.best_score_:.4f}")
print(f"   Test Accuracy: {acc:.4f}")
print(f"   Test F1      : {f1:.4f}")
print(f"\n{classification_report(y_te, y_pred, target_names=['Izquierda','Derecha'])}")

# Comparar con modelo anterior
acc_ant = datos_ant['metricas']['accuracy']
f1_ant  = datos_ant['metricas']['f1']
print(f"   Modelo anterior → Accuracy: {acc_ant}  F1: {f1_ant}")
mejora_acc = acc - acc_ant
mejora_f1  = f1  - f1_ant
print(f"   Cambio          → Accuracy: {mejora_acc:+.4f}  F1: {mejora_f1:+.4f}")


# ══════════════════════════════════════════════════════════════
# GUARDAR MODELO ACTUALIZADO
# ══════════════════════════════════════════════════════════════
# Backup del modelo anterior
import shutil
shutil.copy('modelo_flechas.joblib', 'modelo_flechas_v_anterior.joblib')
print("\n    Backup guardado → modelo_flechas_v_anterior.joblib")

nuevo_modelo = {
    'pipeline'   : best_pipe,
    'hog_params' : HOG_PARAMS,
    'img_size'   : IMG_SIZE,
    'clases'     : {0: 'Izquierda', 1: 'Derecha'},
    'metricas'   : {
        'accuracy' : round(acc, 4),
        'f1'       : round(f1,  4),
        'roc_auc'  : round(roc, 4),
    },
    'n_orig'  : n_orig,
    'n_cam'   : n_cam,
    'n_total' : total,
}
joblib.dump(nuevo_modelo, 'modelo_flechas.joblib', compress=3)
size_kb = os.path.getsize('modelo_flechas.joblib') / 1024
print(f"    Modelo actualizado → modelo_flechas.joblib  ({size_kb:.1f} KB)")
print(f"\n Listo. Vuelve a ejecutar real_time_cam.py con el modelo mejorado.")