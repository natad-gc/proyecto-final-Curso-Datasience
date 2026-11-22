import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro

# 1. Semilla fija para reproducibilidad
SEED = 60
np.random.seed(SEED)

# Crear carpetas si no existen
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# 2. Cargar el dataset Breast Cancer Wisconsin (Diagnostic)
data = load_breast_cancer(as_frame=True)
df = data.frame
X = data.data
y = data.target  # 0 = Maligno, 1 = Benigno

print(f"Dimensiones del dataset: {df.shape}")
print(f"Valores faltantes totales: {df.isnull().sum().sum()}")
print("\nProporción de clases:")
print(y.value_counts(normalize=True).rename({0: "Maligno (0)", 1: "Benigno (1)"}).round(4))

# Nombres en español de las variables (solo para las gráficas)
ATRIBUTOS_ES = {
    "radius": "radio", "texture": "textura", "perimeter": "perímetro", "area": "área",
    "smoothness": "suavidad", "compactness": "compacidad", "concavity": "concavidad",
    "concave points": "puntos cóncavos", "symmetry": "simetría", "fractal dimension": "dimensión fractal",
}
PREFIJOS_ES = {"mean": "media", "error": "error", "worst": "peor"}

def nombre_es(col):
    """Traduce p. ej. 'worst concave points' -> 'puntos cóncavos (peor)'."""
    if col == "target":
        return "diagnóstico (0 = maligno)"
    for pref, pref_es in PREFIJOS_ES.items():
        if col.startswith(pref + " "):
            return f"{ATRIBUTOS_ES[col[len(pref) + 1:]]} ({pref_es})"
    return col

# Configurar estilo limpio para figuras del artículo IEEE
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

# --- GRÁFICA 1: Proporción de clases ---
plt.figure(figsize=(6, 4))
ax = sns.countplot(x=y.map({0: "Maligno", 1: "Benigno"}), palette=["#d62728", "#1f77b4"])
plt.title("Distribución de Clases (Breast Cancer Wisconsin)")
plt.xlabel("Diagnóstico")
plt.ylabel("Número de muestras")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig("reports/figures/fig1_clases.png", dpi=300)
plt.close()

# --- GRÁFICA 2: Mapa de correlación (10 variables más correlacionadas con el diagnóstico) ---
corr_matrix = df.corr()
top_cols = corr_matrix["target"].abs().sort_values(ascending=False).head(11).index
plt.figure(figsize=(9, 7))
corr_es = df[top_cols].corr().rename(index=nombre_es, columns=nombre_es)
sns.heatmap(corr_es, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Matriz de Correlación (Top 10 Variables vs Diagnóstico)")
plt.tight_layout()
plt.savefig("reports/figures/fig2_correlacion.png", dpi=300)
plt.close()

# --- GRÁFICA 3: Distribución de las 4 variables más discriminantes por clase ---
top_4_features = [c for c in top_cols if c != "target"][:4]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
df_plot = df.copy()
df_plot["Diagnóstico"] = df_plot["target"].map({0: "Maligno", 1: "Benigno"})

for ax, col in zip(axes.flatten(), top_4_features):
    sns.boxplot(data=df_plot, x="Diagnóstico", y=col, palette=["#d62728", "#1f77b4"], ax=ax)
    ax.set_title(f"Distribución de {nombre_es(col)}")
    ax.set_ylabel(nombre_es(col))
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig("reports/figures/fig3_distribuciones.png", dpi=300)
plt.close()

print("\nGráficas guardadas en reports/figures/")

# 3. División estratificada: 70% Train, 15% Validation, 15% Test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

print(f"\nTamaños de división -> Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# 4. Escalado (ajustado ÚNICAMENTE con X_train para evitar fuga de datos)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

# 5. Guardar datos procesados y escalador
X_train_scaled.to_csv("data/processed/X_train.csv", index=False)
X_val_scaled.to_csv("data/processed/X_val.csv", index=False)
X_test_scaled.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_val.to_csv("data/processed/y_val.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
joblib.dump(scaler, "models/scaler.joblib")

print("Preprocesamiento terminado y archivos guardados en data/processed/ y models/.")
# Prueba de normalidad Shapiro-Wilk por clase (supuesto de Naive Bayes)
rej_m = sum(shapiro(X.loc[y == 0, col])[1] < 0.05 for col in X.columns)
rej_b = sum(shapiro(X.loc[y == 1, col])[1] < 0.05 for col in X.columns)
print(f"Shapiro-Wilk (p < 0.05) -> Maligno: {rej_m}/30, Benigno: {rej_b}/30")