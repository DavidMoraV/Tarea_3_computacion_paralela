"""
feature_engineering.py
Transformaciones, creación de nuevas características, join con airlines.csv
y agregación group_by por aeropuerto de origen.
"""
from pathlib import Path
import polars as pl


def agregar_features_temporales(df):
    """Crea HORA_SALIDA, TRIMESTRE, ES_FIN_SEMANA y PERIODO_DIA."""
    df = df.with_columns([
        (pl.col("SCHEDULED_DEPARTURE") // 100).clip(0, 23).alias("HORA_SALIDA"),
        ((pl.col("MONTH") - 1) // 3 + 1).alias("TRIMESTRE"),
        pl.col("DAY_OF_WEEK").is_in([6, 7]).cast(pl.Int8).alias("ES_FIN_SEMANA"),
    ])
    return df.with_columns(
        pl.when(pl.col("HORA_SALIDA") < 6).then(pl.lit("madrugada"))
          .when(pl.col("HORA_SALIDA") < 12).then(pl.lit("mañana"))
          .when(pl.col("HORA_SALIDA") < 18).then(pl.lit("tarde"))
          .otherwise(pl.lit("noche")).alias("PERIODO_DIA"))


def unir_aerolineas(df, ruta_raw):
    """Join con airlines.csv para obtener el nombre legible de la aerolínea."""
    airlines = (pl.read_csv(str(Path(ruta_raw) / "airlines.csv"))
                  .rename({"AIRLINE": "AEROLINEA_NOMBRE"}))
    return df.join(airlines, left_on="AIRLINE", right_on="IATA_CODE", how="left")


def agregar_congestion(df):
    """group_by por aeropuerto de origen: VUELOS_ORIGEN como proxy de congestión."""
    vo = df.group_by("ORIGIN_AIRPORT").agg(pl.len().alias("VUELOS_ORIGEN"))
    return df.join(vo, on="ORIGIN_AIRPORT", how="left")


def construir_features(df, ruta_raw):
    """Orquesta todo el feature engineering sobre un DataFrame ya limpio."""
    df = agregar_features_temporales(df)
    df = unir_aerolineas(df, ruta_raw)
    df = agregar_congestion(df)
    return df
