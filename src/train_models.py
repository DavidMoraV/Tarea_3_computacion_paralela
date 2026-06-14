"""
train_models.py
Preprocesamiento para ML, entrenamiento y evaluación de tres modelos de
clasificación (Regresión Logística, Random Forest, Gradient Boosting).
Ejecutable directamente:  python src/train_models.py
"""
import time
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from preprocessing import cargar_vuelos, crear_objetivo, filtrar_y_limpiar
from feature_engineering import construir_features

COLS_NUM = ["MONTH", "DAY", "DAY_OF_WEEK", "DEPARTURE_DELAY", "TAXI_OUT", "SCHEDULED_TIME",
            "DISTANCE", "HORA_SALIDA", "TRIMESTRE", "ES_FIN_SEMANA", "VUELOS_ORIGEN"]
COLS_CAT = ["AIRLINE", "PERIODO_DIA"]


def preparar_datos(df_fe):
    """Codifica categóricas (one-hot) y devuelve X (float32), y y los nombres de features."""
    df_model = (df_fe.select(COLS_NUM + COLS_CAT + ["RETRASADO"])
                     .to_dummies(columns=COLS_CAT, drop_first=True))
    y = df_model["RETRASADO"].to_numpy()
    X = df_model.drop("RETRASADO").to_numpy().astype(np.float32)
    nombres = [c for c in df_model.columns if c != "RETRASADO"]
    return X, y, nombres


def entrenar_evaluar(X, y):
    """Entrena y evalúa los tres modelos. Devuelve la lista de métricas y los modelos ajustados."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    escalador = StandardScaler()
    X_tr_sc = escalador.fit_transform(X_tr).astype(np.float32)
    X_te_sc = escalador.transform(X_te).astype(np.float32)

    modelos = {
        "Regresión Logística":      (LogisticRegression(max_iter=1000, class_weight="balanced"), True),
        "Random Forest":            (RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1,
                                         class_weight="balanced", max_samples=0.5, random_state=42), False),
        "Gradient Boosting (Hist)": (HistGradientBoostingClassifier(max_iter=200, class_weight="balanced",
                                         random_state=42), False),
    }
    resultados, ajustados = [], {}
    for nombre, (modelo, escalar) in modelos.items():
        Xtr, Xte = (X_tr_sc, X_te_sc) if escalar else (X_tr, X_te)
        t0 = time.perf_counter(); modelo.fit(Xtr, y_tr); t_fit = time.perf_counter() - t0
        proba = modelo.predict_proba(Xte)[:, 1]; pred = (proba >= 0.5).astype(int)
        resultados.append({
            "modelo": nombre, "t_entren_s": round(t_fit, 2),
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "f1": round(f1_score(y_te, pred), 4),
            "auc": round(roc_auc_score(y_te, proba), 4),
        })
        ajustados[nombre] = modelo
    return resultados, ajustados


if __name__ == "__main__":
    ruta = Path(__file__).resolve().parent.parent / "data" / "raw"
    df = filtrar_y_limpiar(crear_objetivo(cargar_vuelos(ruta)))
    df_fe = construir_features(df, ruta)
    X, y, nombres = preparar_datos(df_fe)
    print(f"Matriz de modelado: {X.shape} | clase positiva: {y.mean()*100:.2f}%")
    resultados, _ = entrenar_evaluar(X, y)
    print(pl.DataFrame(resultados).sort("auc", descending=True))
