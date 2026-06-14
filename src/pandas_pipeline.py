"""
pandas_pipeline.py
Réplica exacta del pipeline en Pandas con medición de tiempo por etapa.
Ejecutable directamente:  python src/pandas_pipeline.py
"""
import time
from pathlib import Path
import pandas as pd
from preprocessing import COLS_FUGA, COLS_ID, UMBRAL_RETRASO


def pipeline_pandas(ruta_raw):
    """Ejecuta el pipeline completo en Pandas y devuelve (DataFrame, dict de tiempos)."""
    ruta_raw = Path(ruta_raw)
    t = {}

    s = time.perf_counter()
    df = pd.read_csv(ruta_raw / "flights.csv", na_values=["", "NA"])
    t["lectura"] = time.perf_counter() - s

    s = time.perf_counter()
    df = df[df["ARRIVAL_DELAY"].notna()].copy()
    df["RETRASADO"] = (df["ARRIVAL_DELAY"] > UMBRAL_RETRASO).astype("int8")
    df = (df.drop(columns=COLS_FUGA + COLS_ID)
            .dropna(subset=["DEPARTURE_DELAY", "TAXI_OUT", "SCHEDULED_TIME"]))
    t["filtrado"] = time.perf_counter() - s

    s = time.perf_counter()
    df["HORA_SALIDA"] = (df["SCHEDULED_DEPARTURE"] // 100).clip(0, 23)
    df["TRIMESTRE"] = (df["MONTH"] - 1) // 3 + 1
    df["ES_FIN_SEMANA"] = df["DAY_OF_WEEK"].isin([6, 7]).astype("int8")
    df["PERIODO_DIA"] = pd.cut(df["HORA_SALIDA"], bins=[-1, 5, 11, 17, 23],
                               labels=["madrugada", "mañana", "tarde", "noche"])
    t["feature_eng"] = time.perf_counter() - s

    s = time.perf_counter()
    airlines = pd.read_csv(ruta_raw / "airlines.csv").rename(columns={"AIRLINE": "AEROLINEA_NOMBRE"})
    df = df.merge(airlines, left_on="AIRLINE", right_on="IATA_CODE", how="left")
    t["joins"] = time.perf_counter() - s

    s = time.perf_counter()
    vo = df.groupby("ORIGIN_AIRPORT").size().reset_index(name="VUELOS_ORIGEN")
    df = df.merge(vo, on="ORIGIN_AIRPORT", how="left")
    t["agregacion"] = time.perf_counter() - s

    t["total"] = sum(t.values())
    return df, t


if __name__ == "__main__":
    ruta = Path(__file__).resolve().parent.parent / "data" / "raw"
    df, tiempos = pipeline_pandas(ruta)
    print(f"Pandas -> {len(df):,} filas x {df.shape[1]} columnas")
    for etapa, seg in tiempos.items():
        print(f"  {etapa:<14} {seg:7.3f}s")
