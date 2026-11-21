import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# 1. Cargar datos de prueba (Test)
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

nombres_modelos = [
    "Regresion_Logistica",
    "Naive_Bayes",
    "SVM",
    "Red_Neuronal_MLP"
]

resultados_test = []
predicciones = {}
probabilidades = {}

# Configurar estilo para las figuras del artículo IEEE
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

# --- GRÁFICA 4: Matrices de Confusión (2x2) ---
fig_cm, axes = plt.subplots(2, 2, figsize=(10, 8))

for ax, nombre in zip(axes.flatten(), nombres_modelos):
    modelo = joblib.load(f"models/{nombre}.joblib")
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    
    predicciones[nombre] = y_pred
    probabilidades[nombre] = y_prob
    
    # Calcular métricas en Test
    resultados_test.append({
        "Modelo": nombre,
        "Exactitud": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "ROC_AUC": round(roc_auc_score(y_test, y_prob), 4)
    })
    
    # Graficar matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Maligno (0)", "Benigno (1)"],
                yticklabels=["Maligno (0)", "Benigno (1)"])
    ax.set_title(nombre.replace("_", " "))
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")

plt.tight_layout()
plt.savefig("reports/figures/fig4_matrices_confusion.png", dpi=300)
plt.close(fig_cm)

# --- GRÁFICA 5: Curvas ROC Comparativas ---
plt.figure(figsize=(7, 5))
for nombre in nombres_modelos:
    fpr, tpr, _ = roc_curve(y_test, probabilidades[nombre])
    auc_val = roc_auc_score(y_test, probabilidades[nombre])
    plt.plot(fpr, tpr, lw=2, label=f"{nombre.replace('_', ' ')} (AUC = {auc_val:.4f})")

plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Azar (AUC = 0.50)")
plt.title("Curvas ROC Comparativas en el Conjunto de Prueba")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("reports/figures/fig5_curvas_roc.png", dpi=300)
plt.close()

# 2. Guardar e imprimir tabla comparativa final en Test
df_test = pd.DataFrame(resultados_test)
df_test.to_csv("reports/resultados_test.csv", index=False)

print("=== TABLA COMPARATIVA EN CONJUNTO DE PRUEBA (TEST) ===")
print(df_test.to_string(index=False))

# 3. Análisis de Errores (casos donde al menos un modelo se equivocó)
df_errores = pd.DataFrame({"Real": y_test})
for nombre in nombres_modelos:
    df_errores[nombre] = predicciones[nombre]

# Contar cuántos modelos fallaron en cada muestra
df_errores["Total_Fallos"] = sum(df_errores[m] != df_errores["Real"] for m in nombres_modelos)
casos_fallidos = df_errores[df_errores["Total_Fallos"] > 0]

print(f"\n=== ANÁLISIS DE ERRORES ({len(casos_fallidos)} muestras con al menos 1 fallo de {len(y_test)}) ===")
print(casos_fallidos.to_string())
casos_fallidos.to_csv("reports/analisis_errores.csv")

print("\nGráficas guardadas en reports/figures/ (fig4 y fig5) y tablas en reports/.")