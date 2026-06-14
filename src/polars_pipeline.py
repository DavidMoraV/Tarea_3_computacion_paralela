"""
polars_pipeline.py
Pipeline completo en Polars con medición de tiempo por etapa.
Ejecutable directamente:  python src/polars_pipeline.py
"""
import time
from pathlib import Path
import polars as pl
from preprocessing import COLS_FUGA, COLS_ID, UMBRAL_RETRASO


def pipeline_polars(ruta_raw):
    """Ejecuta el pipeline completo y devuelve (DataFrame, dict de tiempos por etapa)."""
    ruta_raw = Path(ruta_raw)
    t = {}

    s = time.perf_counter()
    df = pl.read_csv(str(ruta_raw / "flights.csv"), null_values=["", "NA"], infer_schema_length=10000)
    t["lectura"] = time.perf_counter() - s

    s = time.perf_counter()
    df = df.filter(pl.col("ARRIVAL_DELAY").is_not_null())
    df = df.with_columns((pl.col("ARRIVAL_DELAY") > UMBRAL_RETRASO).cast(pl.Int8).alias("RETRASADO"))
    df = (df.drop(COLS_FUGA + COLS_ID)
            .drop_nulls(subset=["DEPARTURE_DELAY", "TAXI_OUT", "SCHEDULED_TIME"]))
    t["filtrado"] = time.perf_counter() - s

    s = time.perf_counter()
    df = df.with_columns([
        (pl.col("SCHEDULED_DEPARTURE") // 100).clip(0, 23).alias("HORA_SALIDA"),
        ((pl.col("MONTH") - 1) // 3 + 1).alias("TRIMESTRE"),
        pl.col("DAY_OF_WEEK").is_in([6, 7]).cast(pl.Int8).alias("ES_FIN_SEMANA"),
    ])
    df = df.with_columns(
        pl.when(pl.col("HORA_SALIDA") < 6).then(pl.lit("madrugada"))
          .when(pl.col("HORA_SALIDA") < 12).then(pl.lit("mañana"))
          .when(pl.col("HORA_SALIDA") < 18).then(pl.lit("tarde"))
          .otherwise(pl.lit("noche")).alias("PERIODO_DIA"))
    t["feature_eng"] = time.perf_counter() - s

    s = time.perf_counter()
    airlines = pl.read_csv(str(ruta_raw / "airlines.csv")).rename({"AIRLINE": "AEROLINEA_NOMBRE"})
    df = df.join(airlines, left_on="AIRLINE", right_on="IATA_CODE", how="left")
    t["joins"] = time.perf_counter() - s

    s = time.perf_counter()
    vo = df.group_by("ORIGIN_AIRPORT").agg(pl.len().alias("VUELOS_ORIGEN"))
    df = df.join(vo, on="ORIGIN_AIRPORT", how="left")
    t["agregacion"] = time.perf_counter() - s

    t["total"] = sum(t.values())
    return df, t


if __name__ == "__main__":
    ruta = Path(__file__).resolve().parent.parent / "data" / "raw"
    df, tiempos = pipeline_polars(ruta)
    print(f"Polars -> {df.height:,} filas x {df.width} columnas")
    for etapa, seg in tiempos.items():
        print(f"  {etapa:<14} {seg:7.3f}s")
