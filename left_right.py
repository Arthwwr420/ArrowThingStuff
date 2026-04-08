import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from PIL import Image

from sklearn.linear_model  import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics        import (confusion_matrix, accuracy_score, precision_score,
                                    recall_score, f1_score, roc_curve, auc,
                                    ConfusionMatrixDisplay)
from sklearn.preprocessing  import StandardScaler
from sklearn.decomposition  import PCA

warnings.filterwarnings('ignore')
np.random.seed(42)

# Paleta de colores
DARK   = '#003B4A'
ORANGE = '#E07C24'
GREEN  = '#2E8B57'
GRAY   = '#6E6E6E'
RED    = '#C0392B'
LIGHT  = '#E8F6F9'
PURPLE = '#7B2D8B'
BLUE   = '#1f77b4'

plt.rcParams.update({
    'font.family'        : 'sans-serif',
    'font.size'          : 9,
    'axes.titlesize'     : 11,
    'axes.titleweight'   : 'bold',
    'axes.labelsize'     : 9,
    'axes.spines.top'    : False,
    'axes.spines.right'  : False,
    'figure.facecolor'   : 'white',
    'figure.dpi'         : 120,
})

def stitle(fig, t, s=''):
    fig.suptitle(t, fontsize=13, fontweight='bold', color=DARK, y=0.99)
    if s:
        fig.text(0.5, 0.95, s, ha='center', fontsize=9, color=GRAY)

print("Librerías cargadas correctamente.")

# %% [markdown]
# ## 1. Carga del Dataset
# Cargamos todas las imágenes de **Data-L/** (izquierda → clase 0)
# y **Data-R/** (derecha → clase 1).

# %%
# ── Configuración de rutas ───────────────────────────────────
IMG_SIZE    = 64       # píxeles × píxeles de normalización
BASE_DIR = 'dataset'
DIR_LEFT  = os.path.join(BASE_DIR, 'Left')
DIR_RIGHT = os.path.join(BASE_DIR, 'Right')
VALID_EXT   = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.tif'}

def load_folder(folder_path: str, label: int) -> tuple:
    """
    Carga todas las imágenes de una carpeta como arrays numpy
    normalizados en escala de grises.

    Returns
    -------
    images : list of ndarray  (IMG_SIZE, IMG_SIZE) float32 ∈ [0,1]
    labels : list of int
    names  : list of str
    """
    images, labels, names = [], [], []
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"No se encontró la carpeta '{folder_path}'.\n"
            f"   Asegúrate de ejecutar este script desde la carpeta del proyecto.")
    
    files = sorted(f for f in os.listdir(folder_path)
                   if os.path.splitext(f)[1].lower() in VALID_EXT)
    
    if not files:
        raise ValueError(f"No hay imágenes válidas en '{folder_path}'.")
    
    for fname in files:
        path = os.path.join(folder_path, fname)
        try:
            img = (Image.open(path)
                   .convert('L')                              # escala de grises
                   .resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS))
            arr = np.array(img, dtype=np.float32) / 255.0    # normalizar a [0,1]
            images.append(arr)
            labels.append(label)
            names.append(fname)
        except Exception as e:
            print(f"  ⚠ No se pudo leer '{fname}': {e}")
    
    return images, labels, names

# Carga
imgs_L, lbl_L, names_L = load_folder(DIR_LEFT,  label=0)
imgs_R, lbl_R, names_R = load_folder(DIR_RIGHT, label=1)

images_all = np.array(imgs_L + imgs_R,  dtype=np.float32)
labels_all = np.array(lbl_L + lbl_R,   dtype=int)
names_all  = names_L + names_R

n_total = len(images_all)
n_left  = int((labels_all == 0).sum())
n_right = int((labels_all == 1).sum())

print(f"Dataset cargado:")
print(f"   Total  : {n_total} imágenes")
print(f"   Clase 0 (izquierda) : {n_left}")
print(f"   Clase 1 (derecha)   : {n_right}")
print(f"   Balance : {'Balanceado' if abs(n_left-n_right) <= 0.1*n_total else 'Desbalanceado'}")

def show_samples(images, labels, n_per_class=6, title='Muestras del dataset'):
    cols = n_per_class
    fig, axes = plt.subplots(2, cols, figsize=(cols * 1.8, 4.2))
    stitle(fig, title, f'Clase 0 = Izquierda  |  Clase 1 = Derecha')

    class_names = {0: 'Izquierda ←', 1: 'Derecha →'}
    colors_cls  = {0: BLUE, 1: ORANGE}

    for row, cls in enumerate([0, 1]):
        idx = np.where(labels == cls)[0]
        sample_idx = np.random.choice(idx, size=min(cols, len(idx)), replace=False)
        for col, i in enumerate(sample_idx):
            ax = axes[row, col]
            ax.imshow(images[i], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(class_names[cls], color=colors_cls[cls],
                              fontsize=9, fontweight='bold')
        for col in range(len(sample_idx), cols):
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('fig1_muestras.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("💾 Guardado: fig1_muestras.png")

show_samples(images_all, labels_all)

def normalize_polarity(images: np.ndarray) -> np.ndarray:
    """
    Si la imagen tiene fondo oscuro (media < 0.5), la invierte.
    Garantiza que el fondo sea siempre claro.
    """
    out = []
    for img in images:
        out.append(1.0 - img if img.mean() < 0.5 else img.copy())
    return np.array(out, dtype=np.float32)


def show_preprocessing(images, labels):
    """Visualiza el efecto del preprocesamiento."""
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.5))
    stitle(fig, 'Etapa de Preprocesamiento',
           'Original (escala de grises) → Normalización de polaridad')
    
    idx_L = np.where(labels == 0)[0][:2]
    idx_R = np.where(labels == 1)[0][:2]
    idx_show = list(idx_L) + list(idx_R)
    cls_labels = ['Izquierda','Izquierda','Derecha','Derecha']
    colors = [BLUE, BLUE, ORANGE, ORANGE]
    
    imgs_norm = normalize_polarity(images)
    
    for col, (i, cls_name, c) in enumerate(zip(idx_show, cls_labels, colors)):
        ax_orig = axes[0, col]
        ax_norm = axes[1, col]
        ax_orig.imshow(images[i],    cmap='gray', vmin=0, vmax=1)
        ax_norm.imshow(imgs_norm[i], cmap='gray', vmin=0, vmax=1)
        ax_orig.set_title(f'{cls_name}', color=c, fontsize=9)
        for ax in [ax_orig, ax_norm]: ax.axis('off')
    
    axes[0, 0].set_ylabel('Original',    fontsize=9, color=DARK)
    axes[1, 0].set_ylabel('Normalizado', fontsize=9, color=DARK)
    
    plt.tight_layout()
    plt.savefig('fig2_preprocesamiento.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("💾 Guardado: fig2_preprocesamiento.png")

show_preprocessing(images_all, labels_all)

# Aplicar preprocesamiento al dataset completo
images_proc = normalize_polarity(images_all)
print("Preprocesamiento aplicado.")

def extract_features(images: np.ndarray, method: str = 'combined') -> np.ndarray:
    """
    Extrae vector de características por imagen.
    
    La proyección por columnas captura directamente la asimetría izq/der:
    - Flecha →: más masa oscura en columnas de la derecha
    - Flecha ←: más masa oscura en columnas de la izquierda
    
    Flechas oscuras sobre fondo claro → invertimos para que alta señal = flecha.
    """
    features = []
    for img in images:
        dark = 1.0 - img           # inversión: píxeles de flecha → valores altos

        # Perfiles de proyección
        col_proj = dark.mean(axis=0)    # (IMG_SIZE,) — promedio por columna
        row_proj = dark.mean(axis=1)    # (IMG_SIZE,) — promedio por fila

        # Ratio izquierda vs derecha
        half      = IMG_SIZE // 2
        mass_L    = dark[:, :half].mean()
        mass_R    = dark[:, half:].mean()
        lr_diff   = mass_L - mass_R     # >0 → más masa izq → flecha izq (clase 0)
        lr_ratio  = mass_L / (mass_R + 1e-9)

        # Centroide horizontal normalizado [0,1]
        cols  = np.arange(IMG_SIZE, dtype=float)
        total = dark.sum() + 1e-9
        cx    = (dark.sum(axis=0) @ cols) / total / IMG_SIZE

        # Asimetría del perfil de columnas
        left_proj  = col_proj[:half].mean()
        right_proj = col_proj[half:].mean()
        proj_asym  = left_proj - right_proj

        # Gradiente horizontal (detecta dónde están los bordes de la punta)
        grad = np.gradient(col_proj)
        grad_asym = grad[:half].mean() - grad[half:].mean()

        if method == 'raw':
            feat = dark.ravel()
        elif method == 'projection':
            feat = np.concatenate([col_proj, row_proj])
        else:  # combined
            feat = np.concatenate([
                col_proj, row_proj,
                [lr_diff, lr_ratio, cx, proj_asym, grad_asym]
            ])
        features.append(feat)
    
    return np.array(features, dtype=np.float32)


def show_feature_intuition(images, labels):
    """Visualiza la proyección por columnas de ejemplos izq vs der."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5))
    stitle(fig, 'Intuición de las Características',
           'Proyección por columnas — la asimetría horizontal distingue izq de der')
    
    idx_L = np.where(labels == 0)[0][:2]
    idx_R = np.where(labels == 1)[0][:2]
    pairs = [(idx_L[0], 'Izquierda ←', BLUE, 0),
             (idx_L[1], 'Izquierda ←', BLUE, 1),
             (idx_R[0], 'Derecha →',   ORANGE, 2),
             (idx_R[1], 'Derecha →',   ORANGE, 3)]
    
    imgs_n = normalize_polarity(images)
    
    for i_img, label, color, col in pairs:
        img  = imgs_n[i_img]
        dark = 1.0 - img
        col_proj = dark.mean(axis=0)
        half = IMG_SIZE // 2
        mass_L = dark[:, :half].mean()
        mass_R = dark[:, half:].mean()
        
        ax_img  = axes[0, col]
        ax_proj = axes[1, col]
        
        ax_img.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax_img.axvline(half, color='red', lw=1, ls='--', alpha=0.7)
        ax_img.set_title(label, color=color, fontsize=9, fontweight='bold')
        ax_img.axis('off')
        
        xs = np.arange(IMG_SIZE)
        ax_proj.bar(xs[:half], col_proj[:half], color=BLUE,   alpha=0.7,
                    label=f'Izq: {mass_L:.3f}', width=1.0)
        ax_proj.bar(xs[half:], col_proj[half:], color=ORANGE, alpha=0.7,
                    label=f'Der: {mass_R:.3f}', width=1.0)
        ax_proj.axvline(half, color='red', lw=1, ls='--', alpha=0.7)
        ax_proj.legend(fontsize=7, loc='upper center')
        ax_proj.set_xlabel('Columna (px)', fontsize=7)
        if col == 0: ax_proj.set_ylabel('Masa oscura', fontsize=7)
    
    axes[0, 0].set_ylabel('Imagen', fontsize=8, color=DARK, labelpad=4)
    axes[1, 0].set_ylabel('Proyección', fontsize=8, color=DARK, labelpad=4)
    
    plt.tight_layout()
    plt.savefig('fig3_features.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("Guardado: fig3_features.png")

show_feature_intuition(images_proc, labels_all)

# ── Extracción de características ────────────────────────────
METHOD = 'combined'   # 'raw' | 'projection' | 'combined'
X = extract_features(images_proc, method=METHOD)
y = labels_all.copy()

print(f"Vector de características: {X.shape[1]}-D por imagen (método='{METHOD}')")

# ── División entrenamiento / prueba ──────────────────────────
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X, y, np.arange(len(y)),
    test_size=0.2, random_state=42, stratify=y)

print(f"\nDivisión entrenamiento/prueba:")
print(f"  Entrenamiento: {len(X_tr)} muestras  "
      f"(L={int((y_tr==0).sum())}, R={int((y_tr==1).sum())})")
print(f"  Prueba       : {len(X_te)} muestras  "
      f"(L={int((y_te==0).sum())}, R={int((y_te==1).sum())})")

# ── Escalado ─────────────────────────────────────────────────
scaler   = StandardScaler()
X_tr_s   = scaler.fit_transform(X_tr)
X_te_s   = scaler.transform(X_te)

# ── Validación cruzada (5-fold) ──────────────────────────────
clf    = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)
skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(clf, X_tr_s, y_tr, cv=skf, scoring='accuracy')

print(f"\nValidación cruzada (5-fold):")
print(f"  Accuracy por fold: {np.round(cv_acc, 3)}")
print(f"  Media: {cv_acc.mean():.3f}  ±  {cv_acc.std():.3f}")

# ── Entrenamiento final ───────────────────────────────────────
clf.fit(X_tr_s, y_tr)
y_pred  = clf.predict(X_te_s)
y_proba = clf.predict_proba(X_te_s)[:, 1]
print("\nModelo entrenado.")

# %% [markdown]
# ## 6. Evaluación del Modelo

# %%
# ── Métricas ─────────────────────────────────────────────────
acc  = accuracy_score(y_te, y_pred)
prec = precision_score(y_te, y_pred, zero_division=0)
rec  = recall_score(y_te, y_pred,    zero_division=0)
f1   = f1_score(y_te, y_pred,        zero_division=0)
cm   = confusion_matrix(y_te, y_pred)
fpr, tpr, _ = roc_curve(y_te, y_proba)
roc_auc     = auc(fpr, tpr)

print("Métricas en test:")
print(f"   Accuracy  : {acc :.4f}")
print(f"   Precision : {prec:.4f}")
print(f"   Recall    : {rec :.4f}")
print(f"   F1-score  : {f1  :.4f}")
print(f"   ROC AUC   : {roc_auc:.4f}")

# ── Figura de evaluación ─────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
stitle(fig, 'Evaluación del Clasificador',
       f'Regresión Logística — método: {METHOD} — {n_total} imágenes')

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── (a) Matriz de confusión ──────────────────────────────────
ax_cm = fig.add_subplot(gs[0, 0])
disp = ConfusionMatrixDisplay(cm, display_labels=['Izquierda ←', 'Derecha →'])
disp.plot(ax=ax_cm, colorbar=False,
          cmap=plt.cm.Blues if True else None)
ax_cm.set_title('Matriz de Confusión')
ax_cm.set_xlabel('Predicción', fontsize=9)
ax_cm.set_ylabel('Etiqueta real', fontsize=9)
tn, fp, fn, tp = cm.ravel()
ax_cm.text(0.5, -0.22, f'TN={tn}  FP={fp}  FN={fn}  TP={tp}',
           transform=ax_cm.transAxes, ha='center', fontsize=8, color=GRAY)

# ── (b) Curva ROC ─────────────────────────────────────────────
ax_roc = fig.add_subplot(gs[0, 1])
ax_roc.plot(fpr, tpr, color=BLUE, lw=2, label=f'AUC = {roc_auc:.3f}')
ax_roc.plot([0,1], [0,1], 'k--', lw=1, alpha=0.4, label='Azar')
ax_roc.fill_between(fpr, tpr, alpha=0.07, color=BLUE)
ax_roc.set_xlabel('Tasa de Falsos Positivos')
ax_roc.set_ylabel('Tasa de Verdaderos Positivos')
ax_roc.set_title('Curva ROC')
ax_roc.legend(fontsize=8)
ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.02])

# ── (c) Métricas en barras ────────────────────────────────────
ax_met = fig.add_subplot(gs[0, 2])
metricas  = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC AUC']
valores   = [acc, prec, rec, f1, roc_auc]
colores   = [BLUE, ORANGE, GREEN, PURPLE, RED]
bars = ax_met.barh(metricas, valores, color=colores, alpha=0.82, height=0.55)
for bar, val in zip(bars, valores):
    ax_met.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8, fontweight='bold')
ax_met.set_xlim(0, 1.15)
ax_met.set_xlabel('Valor')
ax_met.set_title('Resumen de Métricas')
ax_met.axvline(0.5, color='gray', lw=0.8, ls='--', alpha=0.5)

# ── (d) Validación cruzada ───────────────────────────────────
ax_cv = fig.add_subplot(gs[1, 0])
folds = [f'Fold {i+1}' for i in range(len(cv_acc))]
bar_cv = ax_cv.bar(folds, cv_acc, color=BLUE, alpha=0.75, width=0.5)
ax_cv.axhline(cv_acc.mean(), color=ORANGE, lw=2, ls='--',
              label=f'Media = {cv_acc.mean():.3f}')
ax_cv.fill_between([-0.5, len(cv_acc)-0.5],
                   cv_acc.mean()-cv_acc.std(),
                   cv_acc.mean()+cv_acc.std(),
                   color=ORANGE, alpha=0.12, label=f'±1 std = {cv_acc.std():.3f}')
for bar, val in zip(bar_cv, cv_acc):
    ax_cv.text(bar.get_x()+bar.get_width()/2, val+0.005,
               f'{val:.3f}', ha='center', fontsize=8)
ax_cv.set_ylim(0, 1.15)
ax_cv.set_ylabel('Accuracy')
ax_cv.set_title('Validación Cruzada (5-fold)')
ax_cv.legend(fontsize=7)

# ── (e) Distribución de probabilidades ──────────────────────
ax_prob = fig.add_subplot(gs[1, 1])
prob_L = y_proba[y_te == 0]
prob_R = y_proba[y_te == 1]
bins = np.linspace(0, 1, 20)
ax_prob.hist(prob_L, bins=bins, color=BLUE,   alpha=0.65, label='Izquierda (real)', density=True)
ax_prob.hist(prob_R, bins=bins, color=ORANGE, alpha=0.65, label='Derecha (real)',   density=True)
ax_prob.axvline(0.5, color='red', lw=1.5, ls='--', label='Umbral 0.5')
ax_prob.set_xlabel('P(Derecha | x)')
ax_prob.set_ylabel('Densidad')
ax_prob.set_title('Distribución de Probabilidades')
ax_prob.legend(fontsize=7)

# ── (f) Errores de clasificación ────────────────────────────
ax_err = fig.add_subplot(gs[1, 2])
err_mask = y_pred != y_te
n_err = err_mask.sum()
ax_err.set_facecolor('#FFF5F5')
if n_err > 0:
    err_imgs  = images_proc[idx_te[err_mask]]
    err_preds = y_pred[err_mask]
    err_reals = y_te[err_mask]
    n_show    = min(n_err, 8)
    for k in range(n_show):
        xi = (k % 4) / 4
        yi = 1 - (k // 4 + 1) * 0.45
        ax_err.imshow(err_imgs[k], cmap='gray', vmin=0, vmax=1,
                      extent=[xi, xi+0.23, yi, yi+0.38], aspect='auto')
        cls_r = '←' if err_reals[k]==0 else '→'
        cls_p = '←' if err_preds[k]==0 else '→'
        ax_err.text(xi+0.115, yi-0.04,
                    f'Real:{cls_r} Pred:{cls_p}',
                    ha='center', fontsize=6, color=RED)
    ax_err.set_xlim(0, 1); ax_err.set_ylim(0, 1)
    ax_err.set_title(f'Errores de clasificación ({n_err})')
else:
    ax_err.text(0.5, 0.5, '¡Sin errores! 🎉',
                ha='center', va='center', fontsize=13, color=GREEN, fontweight='bold')
    ax_err.set_title('Errores de clasificación (0)')
ax_err.axis('off')

plt.savefig('fig4_evaluacion.png', bbox_inches='tight', dpi=150)
plt.show()
print("Guardado: fig4_evaluacion.png")

# ── Comparación de métodos ───────────────────────────────────
print("Comparando métodos de características...")
results = {}
for method in ['raw', 'projection', 'combined']:
    Xm     = extract_features(images_proc, method=method)
    Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(
        Xm, y, test_size=0.2, random_state=42, stratify=y)
    sc     = StandardScaler()
    Xm_tr_s = sc.fit_transform(Xm_tr)
    Xm_te_s = sc.transform(Xm_te)
    m      = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    cv_a   = cross_val_score(m, Xm_tr_s, ym_tr, cv=5, scoring='accuracy')
    m.fit(Xm_tr_s, ym_tr)
    yp     = m.predict(Xm_te_s)
    results[method] = {
        'dim'     : Xm.shape[1],
        'cv_mean' : cv_acc.mean() if method == METHOD else cv_a.mean(),
        'cv_std'  : cv_acc.std()  if method == METHOD else cv_a.std(),
        'acc_test': accuracy_score(ym_te, yp),
        'f1_test' : f1_score(ym_te, yp, zero_division=0),
    }
    print(f"  {method:12s} | dim={Xm.shape[1]:5d} | CV={cv_a.mean():.3f}±{cv_a.std():.3f} | test_acc={accuracy_score(ym_te,yp):.3f}")

# ── Figura de generalización ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
stitle(fig, 'Análisis de Generalización',
       'Comparación de métodos de características y visualización PCA')

# ── (a) Comparación accuracy CV vs test ──────────────────────
ax = axes[0]
method_names = list(results.keys())
cv_means  = [results[m]['cv_mean'] for m in method_names]
cv_stds   = [results[m]['cv_std']  for m in method_names]
test_accs = [results[m]['acc_test'] for m in method_names]
test_f1s  = [results[m]['f1_test']  for m in method_names]
x = np.arange(len(method_names))
w = 0.28
ax.bar(x - w, cv_means,  width=w, color=BLUE,   alpha=0.8, label='CV Accuracy',  yerr=cv_stds, capsize=4)
ax.bar(x,      test_accs, width=w, color=ORANGE, alpha=0.8, label='Test Accuracy')
ax.bar(x + w,  test_f1s,  width=w, color=GREEN,  alpha=0.8, label='Test F1-score')
ax.set_xticks(x); ax.set_xticklabels(method_names, fontsize=9)
ax.set_ylim(0, 1.2)
ax.set_ylabel('Valor')
ax.set_title('Métodos de Características')
ax.legend(fontsize=7, loc='upper right')
ax.axhline(0.5, color='gray', lw=0.8, ls='--', alpha=0.5)
for i in range(len(method_names)):
    ax.text(i-w, cv_means[i]+0.04, f'{cv_means[i]:.2f}', ha='center', fontsize=7, color=BLUE)
    ax.text(i,   test_accs[i]+0.04,f'{test_accs[i]:.2f}',ha='center', fontsize=7, color=ORANGE)
    ax.text(i+w, test_f1s[i]+0.04, f'{test_f1s[i]:.2f}', ha='center', fontsize=7, color=GREEN)

# ── (b) PCA 2D del espacio de características ────────────────
ax_pca = axes[1]
X_best = extract_features(images_proc, method='combined')
sc2    = StandardScaler()
X_sc2  = sc2.fit_transform(X_best)
pca    = PCA(n_components=2, random_state=42)
X_2d   = pca.fit_transform(X_sc2)

idx_L_all = np.where(y == 0)[0]
idx_R_all = np.where(y == 1)[0]
ax_pca.scatter(X_2d[idx_L_all, 0], X_2d[idx_L_all, 1],
               c=BLUE, alpha=0.55, s=22, label='Izquierda ←', edgecolors='none')
ax_pca.scatter(X_2d[idx_R_all, 0], X_2d[idx_R_all, 1],
               c=ORANGE, alpha=0.55, s=22, label='Derecha →', edgecolors='none')
ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax_pca.set_title('Espacio de Características (PCA)')
ax_pca.legend(fontsize=8)

# ── (c) Pesos del modelo (proyección por columnas) ───────────
ax_w = axes[2]
# Los primeros IMG_SIZE pesos corresponden a la proyección por columnas
w_col = clf.coef_[0][:IMG_SIZE]
colors_w = [ORANGE if v > 0 else BLUE for v in w_col]
ax_w.bar(np.arange(IMG_SIZE), w_col, color=colors_w, alpha=0.8, width=1.0)
ax_w.axhline(0, color='black', lw=0.8)
ax_w.axvline(IMG_SIZE//2, color='red', lw=1.5, ls='--', alpha=0.7, label='Centro')
ax_w.set_xlabel('Columna (px)')
ax_w.set_ylabel('Peso w')
ax_w.set_title('Pesos del Modelo\n(proyección por columnas)')
ax_w.legend(fontsize=8)
# Anotación interpretativa
ymax = max(abs(w_col.min()), abs(w_col.max()))
ax_w.text(IMG_SIZE*0.15, ymax*0.85, '← Izquierda', color=BLUE, fontsize=7, ha='center')
ax_w.text(IMG_SIZE*0.85, ymax*0.85, 'Derecha →',   color=ORANGE, fontsize=7, ha='center')

plt.tight_layout()
plt.savefig('fig5_generalizacion.png', bbox_inches='tight', dpi=150)
plt.show()
print("💾 Guardado: fig5_generalizacion.png")

print("=" * 55)
print("          REPORTE FINAL — ¿Izquierda o Derecha?")
print("=" * 55)
print(f"\n Dataset")
print(f"   Total imágenes     : {n_total}")
print(f"   Izquierda (clase 0): {n_left}")
print(f"   Derecha   (clase 1): {n_right}")
print(f"   Tamaño normalizado : {IMG_SIZE}×{IMG_SIZE} px")

print(f"\n  Modelo")
print(f"   Método de features : {METHOD}")
print(f"   Dimensión de x     : {X.shape[1]}")
print(f"   Regularización C   : {clf.C}")
print(f"   Solver             : {clf.solver}")

print(f"\n Desempeño")
print(f"   CV Accuracy (5-fold): {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"   Test Accuracy       : {acc:.4f}")
print(f"   Test Precision      : {prec:.4f}")
print(f"   Test Recall         : {rec:.4f}")
print(f"   Test F1-score       : {f1:.4f}")
print(f"   ROC AUC             : {roc_auc:.4f}")

print(f"\n Matriz de Confusión")
print(f"   VP (Der predicha correcta) : {tp}")
print(f"   VN (Izq predicha correcta) : {tn}")
print(f"   FP (Izq predicha como Der) : {fp}")
print(f"   FN (Der predicha como Izq) : {fn}")

print(f"\n  Figuras guardadas")
for i, fname in enumerate(['fig1_muestras.png', 'fig2_preprocesamiento.png',
                            'fig3_features.png', 'fig4_evaluacion.png',
                            'fig5_generalizacion.png'], 1):
    print(f"   {i}. {fname}")
print("\n" + "=" * 55)

def predecir(ruta_imagen: str, mostrar: bool = True) -> str:
    """
    Predice si una imagen contiene una flecha hacia la izquierda o derecha.
    
    Parameters
    ----------
    ruta_imagen : str
        Ruta al archivo de imagen.
    mostrar : bool
        Si True, muestra la imagen y el resultado.
    
    Returns
    -------
    str : 'Izquierda' o 'Derecha'
    """
    img = (Image.open(ruta_imagen)
           .convert('L')
           .resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS))
    arr = np.array(img, dtype=np.float32) / 255.0
    
    # Preprocesar
    if arr.mean() < 0.5:
        arr = 1.0 - arr
    
    # Extraer features y escalar
    feat = extract_features(arr[np.newaxis], method=METHOD)
    feat_s = scaler.transform(feat)
    
    # Predecir
    pred  = clf.predict(feat_s)[0]
    proba = clf.predict_proba(feat_s)[0]
    clase = 'Derecha →' if pred == 1 else 'Izquierda ←'
    conf  = proba[pred]
    
    if mostrar:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.8))
        ax1.imshow(arr, cmap='gray', vmin=0, vmax=1)
        ax1.set_title(ruta_imagen, fontsize=7)
        ax1.axis('off')
        ax2.barh(['Izquierda ←', 'Derecha →'], [proba[0], proba[1]],
                 color=[BLUE, ORANGE], alpha=0.8)
        ax2.set_xlim(0, 1)
        ax2.set_title(f'Predicción: {clase}\nConfianza: {conf:.2%}',
                      color=(ORANGE if pred==1 else BLUE), fontsize=10, fontweight='bold')
        ax2.axvline(0.5, color='red', lw=1, ls='--')
        plt.tight_layout()
        plt.show()
    
    print(f"🎯 Predicción: {clase}  (confianza: {conf:.2%})")
    return clase

# ── Ejemplo de uso ────────────────────────────────────────────
# predecir('mi_flecha.png')
# predecir('Data-L/flecha_001.png')

print("Función 'predecir()' lista.")
print("   Uso: predecir('ruta/a/imagen.png')")
