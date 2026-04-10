import os
import joblib
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from skimage.filters   import threshold_otsu
from skimage.feature   import hog
from skimage           import morphology, exposure

from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics         import (confusion_matrix, accuracy_score, precision_score,
                                     recall_score, f1_score, roc_curve, auc,
                                     ConfusionMatrixDisplay)
from sklearn.preprocessing   import StandardScaler
from sklearn.decomposition   import PCA
from sklearn.pipeline        import Pipeline

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paleta ────────────────────────────────────────────────────
BLUE   = '#1f77b4'
DARK   = '#003B4A'
ORANGE = '#E07C24'
GREEN  = '#2E8B57'
GRAY   = '#6E6E6E'
RED    = '#C0392B'
PURPLE = '#7B2D8B'

plt.rcParams.update({
    'font.family'      : 'sans-serif',  'font.size'        : 9,
    'axes.titlesize'   : 11,            'axes.titleweight'  : 'bold',
    'axes.labelsize'   : 9,             'axes.spines.top'   : False,
    'axes.spines.right': False,         'figure.facecolor'  : 'white',
    'figure.dpi'       : 120,
})

def stitle(fig, t, s=''):
    fig.suptitle(t, fontsize=13, fontweight='bold', color=DARK, y=0.99)
    if s: fig.text(0.5, 0.95, s, ha='center', fontsize=9, color=GRAY)


# ══════════════════════════════════════════════════════════════
# PARÁMETROS
# ══════════════════════════════════════════════════════════════
IMG_SIZE  = 64
BASE_DIR  = 'dataset'
DIR_LEFT  = os.path.join(BASE_DIR, 'Left')
DIR_RIGHT = os.path.join(BASE_DIR, 'Right')
VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# HOG: con IMG_SIZE=64, pixels_per_cell=8 → 8×8 celdas → 8×8×4×9 = 2304 features
HOG_PARAMS = dict(
    orientations    = 9,
    pixels_per_cell = (8, 8),
    cells_per_block = (2, 2),
    block_norm      = 'L2-Hys',
    visualize       = True,
)


# ══════════════════════════════════════════════════════════════
# 1. CARGA
# ══════════════════════════════════════════════════════════════
def load_folder(folder_path, label):
    images, labels, names = [], [], []
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"\n Carpeta '{folder_path}' no encontrada."
            f"\n    Ejecuta desde la carpeta que contiene dataset/")
    files = sorted(f for f in os.listdir(folder_path)
                   if os.path.splitext(f)[1].lower() in VALID_EXT)
    if not files:
        raise ValueError(f" No hay imágenes en '{folder_path}'.")
    for fname in files:
        path = os.path.join(folder_path, fname)
        try:
            img = (Image.open(path)
                   .convert('L')
                   .resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS))
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr); labels.append(label); names.append(fname)
        except Exception as e:
            print(f"  ⚠  No se pudo leer '{fname}': {e}")
    return images, labels, names

print("\n" + "═"*55)
print("  ¿Izquierda o Derecha? v2 — Otsu + HOG + GridSearch")
print("═"*55)
print("\n Cargando dataset...")

imgs_L, lbl_L, names_L = load_folder(DIR_LEFT,  label=0)
imgs_R, lbl_R, names_R = load_folder(DIR_RIGHT, label=1)

images_raw = np.array(imgs_L + imgs_R, dtype=np.float32)
labels_all = np.array(lbl_L  + lbl_R,  dtype=int)
names_all  = names_L + names_R

n_total = len(images_raw)
n_left  = int((labels_all == 0).sum())
n_right = int((labels_all == 1).sum())
print(f"   Total  : {n_total}  |  Izquierda: {n_left}  |  Derecha: {n_right}")


# ══════════════════════════════════════════════════════════════
# 2. PREPROCESAMIENTO ROBUSTO — MEJORA 1
#    Otsu separa la flecha del fondo sin importar el color
#    ni el contraste de cada imagen.
# ══════════════════════════════════════════════════════════════
def preprocess_otsu(img: np.ndarray) -> np.ndarray:
    """
    Pipeline de preprocesamiento robusto:
      1. Ecualización de histograma  → contraste uniforme
      2. Umbralización Otsu          → binariza automáticamente
      3. Normalización de polaridad  → flecha siempre oscura (1.0)
      4. Limpieza morfológica        → elimina ruido pequeño

    Devuelve imagen float32 ∈ [0, 1] con la flecha en blanco
    sobre fondo negro.
    """
    # 1. Ecualización adaptativa para igualar imágenes de distintos contrastes
    img_eq = exposure.equalize_adapthist(img, clip_limit=0.03)

    # 2. Umbral de Otsu: encuentra automáticamente el mejor corte
    thresh = threshold_otsu(img_eq)
    binary = (img_eq > thresh).astype(np.float32)

    # 3. Normalización de polaridad
    #    Si la flecha es oscura en el original → es 0 tras Otsu → invertir
    if binary.mean() > 0.5:          # mayoría de píxeles son 1 → fondo blanco
        binary = 1.0 - binary        # invertir para que flecha=1, fondo=0

    # 4. Limpieza: eliminar manchas pequeñas (ruido de compresión, etc.)
    bool_mask = binary.astype(bool)
    cleaned   = morphology.remove_small_objects(bool_mask, min_size=20)
    cleaned   = morphology.remove_small_holes(cleaned, area_threshold=50)

    return cleaned.astype(np.float32)


print("  Aplicando preprocesamiento Otsu...")
images_proc = np.array([preprocess_otsu(img) for img in images_raw])
print("      Listo.")


# ── Figura comparativa: raw vs Otsu ──────────────────────────
def show_otsu_comparison(raw, proc, labels, n=4):
    idx_L = np.where(labels == 0)[0][:n//2]
    idx_R = np.where(labels == 1)[0][:n//2]
    idx   = list(idx_L) + list(idx_R)
    cls_c = {0: BLUE, 1: ORANGE}
    cls_t = {0: '← Izq', 1: 'Der →'}

    fig, axes = plt.subplots(3, n, figsize=(n*2.2, 6.5))
    stitle(fig, 'Preprocesamiento: Otsu + limpieza morfológica',
           'Fila 1: original | Fila 2: Otsu binarizado | Fila 3: HOG visual')

    for col, i in enumerate(idx):
        c = labels[i]
        # Original
        axes[0, col].imshow(raw[col], cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(cls_t[c], color=cls_c[c], fontsize=9)
        # Otsu
        axes[1, col].imshow(proc[col], cmap='gray', vmin=0, vmax=1)
        # HOG visualización
        _, hog_img = hog(proc[col], **HOG_PARAMS)
        hog_img_eq = exposure.rescale_intensity(hog_img, in_range=(0, 0.1))
        axes[2, col].imshow(hog_img_eq, cmap='magma', vmin=0, vmax=1)

    for row, label in enumerate(['Original', 'Binarizado\n(Otsu)', 'HOG visual']):
        axes[row, 0].set_ylabel(label, fontsize=8, color=DARK)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('fig_preprocesamiento_otsu.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("    fig_preprocesamiento_otsu.png")

show_otsu_comparison(images_raw, images_proc, labels_all)


# ══════════════════════════════════════════════════════════════
# 3. EXTRACCIÓN DE FEATURES — MEJORA 2
#    HOG captura la orientación de los bordes de la flecha.
#    La punta de una flecha → tiene bordes diagonales únicos
#    que apuntan izq o der. HOG los cuantifica directamente.
# ══════════════════════════════════════════════════════════════
def extract_hog_features(img: np.ndarray) -> np.ndarray:
    """
    Histograma de Gradientes Orientados (HOG).
    Con IMG_SIZE=64 y pixels_per_cell=8:
      - 8×8 celdas → 4 bloques × 9 orientaciones × 4 = 1296 features
    Añadimos features manuales de asimetría izq/der como complemento.
    """
    features_hog, _ = hog(img, **HOG_PARAMS)
    return features_hog


def extract_asymmetry_features(img: np.ndarray) -> np.ndarray:
    """
    Features manuales de asimetría horizontal (mismos que v1).
    Complementan al HOG cuando la flecha es muy simple.
    """
    dark     = img.copy()                    # ya es binario: flecha=1, fondo=0
    col_proj = dark.mean(axis=0)
    row_proj = dark.mean(axis=1)
    half     = IMG_SIZE // 2

    mass_L   = dark[:, :half].mean()
    mass_R   = dark[:, half:].mean()
    lr_diff  = mass_L - mass_R
    lr_ratio = mass_L / (mass_R + 1e-9)

    cols  = np.arange(IMG_SIZE, dtype=float)
    total = dark.sum() + 1e-9
    cx    = (dark.sum(axis=0) @ cols) / total / IMG_SIZE

    proj_asym = col_proj[:half].mean() - col_proj[half:].mean()
    grad_asym = (np.gradient(col_proj)[:half].mean() -
                 np.gradient(col_proj)[half:].mean())

    # Cuadrantes (división en 4 zonas)
    tl = dark[:half, :half].mean()
    tr = dark[:half, half:].mean()
    bl = dark[half:, :half].mean()
    br = dark[half:, half:].mean()
    quad_asym = (tl + bl) - (tr + br)    # >0 → más masa izquierda

    return np.array([lr_diff, lr_ratio, cx, proj_asym,
                     grad_asym, quad_asym, tl, tr, bl, br],
                    dtype=np.float32)


def extract_features_v2(images: np.ndarray) -> np.ndarray:
    """
    Combina HOG + features de asimetría manual.
    HOG aporta detalle de orientación local.
    Asimetría aporta información global izq/der.
    """
    feats = []
    for img in images:
        hog_f   = extract_hog_features(img)
        asym_f  = extract_asymmetry_features(img)
        feats.append(np.concatenate([hog_f, asym_f]))
    return np.array(feats, dtype=np.float32)


print("\n Extrayendo features HOG + asimetría...")
X = extract_features_v2(images_proc)
y = labels_all.copy()
print(f"   Vector x: {X.shape[1]}-D por imagen")
print(f"   ({hog(images_proc[0], **HOG_PARAMS)[0].shape[0]} HOG + 10 asimetría)")


# ── Visualización: importancia de orientaciones HOG ──────────
def show_hog_orientation(images, labels):
    """Muestra cómo el HOG responde diferente a izq vs der."""
    idx_L = np.where(labels == 0)[0][0]
    idx_R = np.where(labels == 1)[0][0]

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
    stitle(fig, 'HOG: orientaciones de gradiente por clase',
           'La punta de la flecha genera un patrón HOG espejado entre izq y der')

    for row, (i, tag, color) in enumerate([
            (idx_L, '← Izquierda', BLUE),
            (idx_R, 'Derecha →',   ORANGE)]):
        img = images[i]
        feat, hog_img = hog(img, **HOG_PARAMS)
        hog_eq = exposure.rescale_intensity(hog_img, in_range=(0, 0.08))

        axes[row, 0].imshow(images_raw[i], cmap='gray', vmin=0, vmax=1)
        axes[row, 0].set_title(f'Original — {tag}', color=color, fontsize=9)

        axes[row, 1].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title('Binarizado (Otsu)', color=DARK, fontsize=9)

        axes[row, 2].imshow(hog_eq, cmap='magma')
        axes[row, 2].set_title('HOG — orientaciones', color=DARK, fontsize=9)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('fig_hog_orientaciones.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("    fig_hog_orientaciones.png")

show_hog_orientation(images_proc, labels_all)


# ══════════════════════════════════════════════════════════════
# 4. ENTRENAMIENTO CON GRIDSEARCHCV — MEJORA 3
# ══════════════════════════════════════════════════════════════
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X, y, np.arange(len(y)),
    test_size=0.2, random_state=42, stratify=y)

print(f"\n División: {len(X_tr)} entrenamiento / {len(X_te)} prueba")

# Pipeline: escalado → regresión logística
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(max_iter=2000, random_state=42,
                                  class_weight='balanced')),
])

# Grilla de hiperparámetros
#   C controla la regularización: C pequeño → más regularización → más robusto
#   solver: lbfgs funciona bien para datos densos (HOG)
param_grid = {
    'clf__C'      : [0.001, 0.01, 0.1, 1, 10, 100],
    'clf__solver' : ['lbfgs', 'liblinear'],
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n Ejecutando GridSearchCV (esto puede tardar 30-60 s)...")
gs = GridSearchCV(pipe, param_grid, cv=skf,
                  scoring='f1', n_jobs=-1, verbose=0)
gs.fit(X_tr, y_tr)

best_C      = gs.best_params_['clf__C']
best_solver = gs.best_params_['clf__solver']
best_cv_f1  = gs.best_score_
print(f"   Mejor C      : {best_C}")
print(f"   Mejor solver : {best_solver}")
print(f"   Mejor CV F1  : {best_cv_f1:.4f}")

# Modelo final con los mejores parámetros
best_pipe = gs.best_estimator_

# También guardar el scaler y clf por separado para predecir luego
scaler_final = best_pipe.named_steps['scaler']
clf_final    = best_pipe.named_steps['clf']

# Predicciones
y_pred  = best_pipe.predict(X_te)
y_proba = best_pipe.predict_proba(X_te)[:, 1]

acc     = accuracy_score(y_te, y_pred)
prec    = precision_score(y_te, y_pred, zero_division=0)
rec     = recall_score(y_te, y_pred,    zero_division=0)
f1      = f1_score(y_te, y_pred,        zero_division=0)
cm      = confusion_matrix(y_te, y_pred)
fpr, tpr, _ = roc_curve(y_te, y_proba)
roc_auc     = auc(fpr, tpr)
tn, fp, fn, tp = cm.ravel()

print(f"\n Resultados en test:")
print(f"   Accuracy  : {acc :.4f}")
print(f"   Precision : {prec:.4f}")
print(f"   Recall    : {rec :.4f}")
print(f"   F1-score  : {f1  :.4f}")
print(f"   ROC AUC   : {roc_auc:.4f}")


# ══════════════════════════════════════════════════════════════
# 5. VISUALIZACIÓN COMPLETA
# ══════════════════════════════════════════════════════════════

# ── Figura 1: GridSearch — mapa de calor C vs solver ─────────
def show_gridsearch(gs, param_grid):
    results = gs.cv_results_
    Cs      = param_grid['clf__C']
    solvers = param_grid['clf__solver']

    scores = np.zeros((len(solvers), len(Cs)))
    for i, sol in enumerate(solvers):
        for j, c in enumerate(Cs):
            mask = ((np.array(results['param_clf__C']) == c) &
                    (np.array(results['param_clf__solver']) == sol))
            scores[i, j] = results['mean_test_score'][mask][0]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    stitle(fig, 'GridSearchCV — F1 por hiperparámetros',
           f'Mejor C={best_C}, solver={best_solver}, CV F1={best_cv_f1:.3f}')
    im = ax.imshow(scores, cmap='YlOrRd', vmin=scores.min()-0.01,
                   vmax=min(scores.max()+0.01, 1.0), aspect='auto')
    ax.set_xticks(range(len(Cs)));      ax.set_xticklabels([str(c) for c in Cs])
    ax.set_yticks(range(len(solvers))); ax.set_yticklabels(solvers)
    ax.set_xlabel('C (regularización)')
    ax.set_ylabel('Solver')
    plt.colorbar(im, ax=ax, label='CV F1')
    for i in range(len(solvers)):
        for j in range(len(Cs)):
            ax.text(j, i, f'{scores[i,j]:.3f}', ha='center', va='center',
                    fontsize=8, color='black' if scores[i,j] < 0.85 else 'white')
    # Marcar el mejor
    bi = solvers.index(best_solver)
    bj = Cs.index(best_C)
    ax.add_patch(plt.Rectangle((bj-0.5, bi-0.5), 1, 1,
                                fill=False, edgecolor='blue', lw=2.5))
    plt.tight_layout()
    plt.savefig('fig_gridsearch.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("    fig_gridsearch.png")

show_gridsearch(gs, param_grid)


# ── Figura 2: Evaluación completa ────────────────────────────
fig = plt.figure(figsize=(14, 9))
stitle(fig, 'Evaluación del Clasificador v2 — Otsu + HOG + GridSearch',
       f'C={best_C}  solver={best_solver}  |  {n_total} imágenes')
gs_grid = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# (a) Matriz de confusión
ax_cm = fig.add_subplot(gs_grid[0, 0])
ConfusionMatrixDisplay(cm, display_labels=['← Izquierda', 'Derecha →']).plot(
    ax=ax_cm, colorbar=False, cmap=plt.cm.Blues)
ax_cm.set_title('Matriz de Confusión')
ax_cm.text(0.5, -0.22, f'TN={tn}  FP={fp}  FN={fn}  TP={tp}',
           transform=ax_cm.transAxes, ha='center', fontsize=8, color=GRAY)

# (b) Curva ROC
ax_roc = fig.add_subplot(gs_grid[0, 1])
ax_roc.plot(fpr, tpr, color=BLUE, lw=2, label=f'AUC = {roc_auc:.3f}')
ax_roc.plot([0,1], [0,1], 'k--', lw=1, alpha=0.4, label='Azar')
ax_roc.fill_between(fpr, tpr, alpha=0.08, color=BLUE)
ax_roc.set_xlabel('Tasa Falsos Positivos')
ax_roc.set_ylabel('Tasa Verdaderos Positivos')
ax_roc.set_title('Curva ROC')
ax_roc.legend(fontsize=8); ax_roc.set_xlim(0,1); ax_roc.set_ylim(0,1.02)

# (c) Métricas en barras
ax_met = fig.add_subplot(gs_grid[0, 2])
metricas = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC AUC']
valores  = [acc, prec, rec, f1, roc_auc]
colores  = [BLUE, ORANGE, GREEN, PURPLE, RED]
bars = ax_met.barh(metricas, valores, color=colores, alpha=0.82, height=0.55)
for bar, val in zip(bars, valores):
    ax_met.text(val+0.01, bar.get_y()+bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8, fontweight='bold')
ax_met.set_xlim(0, 1.18); ax_met.set_title('Resumen de Métricas')
ax_met.axvline(0.5, color='gray', lw=0.8, ls='--', alpha=0.5)

# (d) PCA del espacio HOG
ax_pca = fig.add_subplot(gs_grid[1, 0])
X_sc   = scaler_final.transform(X)
pca    = PCA(n_components=2, random_state=42)
X_2d   = pca.fit_transform(X_sc)
ax_pca.scatter(X_2d[y==0, 0], X_2d[y==0, 1], c=BLUE,   alpha=0.5, s=18,
               label='Izquierda ←', edgecolors='none')
ax_pca.scatter(X_2d[y==1, 0], X_2d[y==1, 1], c=ORANGE, alpha=0.5, s=18,
               label='Derecha →',   edgecolors='none')
ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax_pca.set_title('Espacio HOG (PCA 2D)')
ax_pca.legend(fontsize=8)

# (e) Distribución de probabilidades
ax_prob = fig.add_subplot(gs_grid[1, 1])
bins = np.linspace(0, 1, 22)
ax_prob.hist(y_proba[y_te==0], bins=bins, color=BLUE,   alpha=0.65,
             density=True, label='Izquierda (real)')
ax_prob.hist(y_proba[y_te==1], bins=bins, color=ORANGE, alpha=0.65,
             density=True, label='Derecha (real)')
ax_prob.axvline(0.5, color=RED, lw=1.5, ls='--', label='Umbral 0.5')
ax_prob.set_xlabel('P(Derecha | x)')
ax_prob.set_ylabel('Densidad')
ax_prob.set_title('Distribución de Probabilidades')
ax_prob.legend(fontsize=7)

# (f) Errores
ax_err = fig.add_subplot(gs_grid[1, 2])
err_mask = y_pred != y_te
n_err    = err_mask.sum()
ax_err.set_facecolor('#FFF5F5')
if n_err > 0:
    n_show_err = min(n_err, 8)
    err_imgs   = images_proc[idx_te[err_mask]][:n_show_err]
    err_preds  = y_pred[err_mask][:n_show_err]
    err_reals  = y_te[err_mask][:n_show_err]
    for k in range(n_show_err):
        xi = (k % 4) / 4
        yi = 1 - (k // 4 + 1) * 0.47
        ax_err.imshow(err_imgs[k], cmap='gray', vmin=0, vmax=1,
                      extent=[xi, xi+0.23, yi, yi+0.4], aspect='auto')
        r = '←' if err_reals[k]==0 else '→'
        p = '←' if err_preds[k]==0 else '→'
        ax_err.text(xi+0.115, yi-0.05, f'Real:{r} Pred:{p}',
                    ha='center', fontsize=6, color=RED)
    ax_err.set_xlim(0,1); ax_err.set_ylim(0,1)
    ax_err.set_title(f'Errores de clasificación ({n_err})')
else:
    ax_err.text(0.5, 0.5, '¡Sin errores! ',
                ha='center', va='center', fontsize=14, color=GREEN, fontweight='bold')
    ax_err.set_title('Errores de clasificación (0)')
ax_err.axis('off')

plt.savefig('fig_evaluacion_v2.png', bbox_inches='tight', dpi=150)
plt.show()
print("    fig_evaluacion_v2.png")


# ── Figura 3: componentes HOG más importantes ────────────────
def show_top_hog_weights(clf, n_top=20):
    """
    Visualiza los pesos más grandes del modelo para entender
    qué orientaciones HOG son más discriminativas.
    """
    w         = clf.coef_[0]
    n_hog     = X.shape[1] - 10          # últimos 10 son asimetría manual
    w_hog     = w[:n_hog]
    w_asym    = w[n_hog:]
    asym_names = ['lr_diff','lr_ratio','cx','proj_asym',
                  'grad_asym','quad_asym','tl','tr','bl','br']

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    stitle(fig, 'Pesos del modelo — ¿qué aprendió?',
           'Azul = evidencia de izquierda  |  Naranja = evidencia de derecha')

    # Histograma de pesos HOG
    ax = axes[0]
    pos_w = np.where(w_hog > 0, w_hog, 0)
    neg_w = np.where(w_hog < 0, w_hog, 0)
    ax.bar(range(len(w_hog)), pos_w, color=ORANGE, alpha=0.7, width=1.0, label='→ Derecha')
    ax.bar(range(len(w_hog)), neg_w, color=BLUE,   alpha=0.7, width=1.0, label='← Izquierda')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('Feature HOG (índice)')
    ax.set_ylabel('Peso w')
    ax.set_title('Distribución de pesos HOG')
    ax.legend(fontsize=8)

    # Features de asimetría manual
    ax2 = axes[1]
    colors_w = [ORANGE if v > 0 else BLUE for v in w_asym]
    bars = ax2.barh(asym_names, w_asym, color=colors_w, alpha=0.82, height=0.6)
    for bar, val in zip(bars, w_asym):
        ax2.text(val + (0.002 if val >= 0 else -0.002),
                 bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right',
                 fontsize=7)
    ax2.axvline(0, color='black', lw=0.8)
    ax2.set_title('Pesos features de asimetría manual')
    ax2.set_xlabel('Peso w')

    plt.tight_layout()
    plt.savefig('fig_pesos_modelo.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("    fig_pesos_modelo.png")

show_top_hog_weights(clf_final)


# ══════════════════════════════════════════════════════════════
# 6. REPORTE FINAL
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*55)
print("   REPORTE FINAL — v2")
print("═"*55)
print(f"\n   Dataset : {n_total} imágenes  ({n_left} izq / {n_right} der)")
print(f"   Features : HOG({hog(images_proc[0],**HOG_PARAMS)[0].shape[0]}) + asimetría(10) = {X.shape[1]}-D")
print(f"   Modelo   : LogisticRegression  C={best_C}  solver={best_solver}")
print(f"\n   Test Accuracy  : {acc :.4f}")
print(f"   Test Precision : {prec:.4f}")
print(f"   Test Recall    : {rec :.4f}")
print(f"   Test F1-score  : {f1  :.4f}")
print(f"   ROC AUC        : {roc_auc:.4f}")
print(f"\n   Errores en test: {n_err} / {len(y_te)}")
print("\n   Figuras:")
for fname in ['fig_preprocesamiento_otsu.png', 'fig_hog_orientaciones.png',
              'fig_gridsearch.png', 'fig_evaluacion_v2.png', 'fig_pesos_modelo.png']:
    print(f"   • {fname}")
print("\n" + "═"*55)

# ── Guardar modelo ────────────────────────────────────────────
modelo_exportado = {
    'pipeline'   : best_pipe,       # StandardScaler + LogisticRegression
    'hog_params' : HOG_PARAMS,      # parámetros HOG usados en entrenamiento
    'img_size'   : IMG_SIZE,        # tamaño de imagen esperado
    'clases'     : {0: 'Izquierda', 1: 'Derecha'},
    'metricas'   : {
        'accuracy' : round(acc,  4),
        'f1'       : round(f1,   4),
        'roc_auc'  : round(roc_auc, 4),
    }
}
 
joblib.dump(modelo_exportado, 'modelo_flechas.joblib', compress=3)
print("\n Modelo exportado → modelo_flechas.joblib")
print(f"   Tamaño: {os.path.getsize('modelo_flechas.joblib') / 1024:.1f} KB")
print(f"   Accuracy guardada: {acc:.4f}  |  F1: {f1:.4f}")

# ══════════════════════════════════════════════════════════════
# 7. FUNCIÓN DE PREDICCIÓN
# ══════════════════════════════════════════════════════════════
def predecir(ruta_imagen: str, mostrar: bool = True) -> str:
    """
    Predice si una flecha apunta a la izquierda o a la derecha.

    Uso:
        predecir('dataset/Left/flecha_01.png')
        predecir('mi_foto.jpg')
    """
    img_pil = (Image.open(ruta_imagen)
               .convert('L')
               .resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS))
    img_raw = np.array(img_pil, dtype=np.float32) / 255.0
    img_proc = preprocess_otsu(img_raw)

    feat   = extract_features_v2(img_proc[np.newaxis])  # (1, D)
    pred   = best_pipe.predict(feat)[0]
    proba  = best_pipe.predict_proba(feat)[0]
    clase  = 'Derecha →' if pred == 1 else 'Izquierda ←'
    conf   = proba[pred]

    if mostrar:
        _, hog_vis = hog(img_proc, **HOG_PARAMS)
        hog_eq     = exposure.rescale_intensity(hog_vis, in_range=(0, 0.08))

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(img_raw,  cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Original', fontsize=9)
        axes[1].imshow(img_proc, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Binarizado (Otsu)', fontsize=9)
        axes[2].imshow(hog_eq,   cmap='magma')
        axes[2].set_title('HOG visual', fontsize=9)
        for ax in axes: ax.axis('off')

        color = ORANGE if pred == 1 else BLUE
        fig.suptitle(f'Predicción: {clase}   (confianza: {conf:.1%})',
                     fontsize=12, fontweight='bold', color=color)
        plt.tight_layout()
        plt.show()

    print(f" {clase}  (confianza: {conf:.1%})")
    return clase


print("\n Función 'predecir()' lista.")
print("    Uso: predecir('ruta/imagen.png')\n")