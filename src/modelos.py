import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score

# Semilla fija para reproducibilidad
SEED = 60
np.random.seed(SEED)

# 1. Cargar datos procesados (entrenamiento y validación)
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
X_val = pd.read_csv("data/processed/X_val.csv")
y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()

# Validación cruzada estratificada (5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# 2. Definir los 4 modelos y su espacio de búsqueda de hiperparámetros
modelos_config = {
    "Regresion_Logistica": {
        "modelo": LogisticRegression(random_state=SEED, max_iter=1000),
        "params": {
            "C": [0.01, 0.1, 1, 10, 100]
        }
    },
    "Naive_Bayes": {
        "modelo": GaussianNB(),
        "params": {
            "var_smoothing": np.logspace(-11, -7, 5)
        }
    },
    "SVM": {
        "modelo": SVC(probability=True, random_state=SEED),
        "params": [
            {"kernel": ["linear"], "C": [0.1, 1, 10, 100]},
            {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1]}
        ]
    },
    "Red_Neuronal_MLP": {
        "modelo": MLPClassifier(
            random_state=SEED,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=15
        ),
        "params": {
            "hidden_layer_sizes": [(32,), (64, 32), (32, 16)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01]  # Regularización L2
        }
    }
}

resultados_cv = []

print("=== INICIANDO ENTRENAMIENTO Y BÚSQUEDA DE HIPERPARÁMETROS (CV=5) ===\n")

# 3. Ejecutar GridSearchCV para cada modelo
for nombre, cfg in modelos_config.items():
    print(f"Entrenando {nombre}...")
    grid = GridSearchCV(
        estimator=cfg["modelo"],
        param_grid=cfg["params"],
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        refit=True
    )
    grid.fit(X_train, y_train)
    
    # Mejor modelo y métricas de validación cruzada
    mejor_modelo = grid.best_estimator_
    idx_mejor = grid.best_index_
    cv_mean = grid.cv_results_["mean_test_score"][idx_mejor]
    cv_std = grid.cv_results_["std_test_score"][idx_mejor]
    
    # Evaluar en el conjunto de validación (X_val)
    y_val_pred = mejor_modelo.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    # Guardar el mejor modelo entrenado
    ruta_modelo = f"models/{nombre}.joblib"
    joblib.dump(mejor_modelo, ruta_modelo)
    
    resultados_cv.append({
        "Modelo": nombre,
        "Mejores_Hiperparametros": str(grid.best_params_),
        "CV_F1_Media": round(cv_mean, 4),
        "CV_F1_DesvEst": round(cv_std, 4),
        "Val_F1": round(val_f1, 4),
        "Val_Exactitud": round(val_acc, 4)
    })
    
    print(f"  -> Mejores parámetros: {grid.best_params_}")
    print(f"  -> CV F1: {cv_mean:.4f} (+/- {cv_std:.4f}) | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}\n")

# 4. Guardar tabla resumen de validación cruzada
df_res = pd.DataFrame(resultados_cv)
os.makedirs("reports", exist_ok=True)
df_res.to_csv("reports/resultados_cv.csv", index=False)

print("=== RESUMEN FINAL (GUARDADO EN reports/resultados_cv.csv) ===")
print(df_res[["Modelo", "CV_F1_Media", "CV_F1_DesvEst", "Val_F1", "Val_Exactitud"]].to_string(index=False))