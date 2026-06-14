"""
preprocessing.py
Carga del dataset de vuelos, creación de la variable objetivo y limpieza
(filtrado de cancelados, eliminación de columnas con fuga de datos y nulos).
"""
from pathlib import Path
import polars as pl

UMBRAL_RETRASO = 15  # minutos: un vuelo está "retrasado" si ARRIVAL_DELAY supera este valor

# Columnas con fuga de datos: se conocen DURANTE o DESPUÉS del vuelo, no al predecir
COLS_FUGA = ["ARRIVAL_DELAY", "ELAPSED_TIME", "AIR_TIME", "ARRIVAL_TIME", "WHEELS_ON",
             "TAXI_IN", "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
             "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY", "CANCELLATION_REASON",
             "DIVERTED", "CANCELLED"]
# Identificadores y constantes sin valor predictivo
COLS_ID = ["YEAR", "FLIGHT_NUMBER", "TAIL_NUMBER", "DEPARTURE_TIME", "WHEELS_OFF"]


def cargar_vuelos(ruta_raw):
    """Lee flights.csv con Polars desde la carpeta data/raw."""
    return pl.read_csv(str(Path(ruta_raw) / "flights.csv"),
                       null_values=["", "NA"], infer_schema_length=10000)


def crear_objetivo(df):
    """Agrega la columna binaria RETRASADO (1 si ARRIVAL_DELAY > umbral)."""
    return df.with_columns(
        (pl.col("ARRIVAL_DELAY") > UMBRAL_RETRASO).cast(pl.Int8).alias("RETRASADO"))


def filtrar_y_limpiar(df):
    """Elimina cancelados/desviados, columnas con fuga e identificadores, y nulos residuales.
    Requiere que RETRASADO ya esté creada (crear_objetivo)."""
    return (df.filter(pl.col("ARRIVAL_DELAY").is_not_null())
              .drop(COLS_FUGA + COLS_ID)
              .drop_nulls(subset=["DEPARTURE_DELAY", "TAXI_OUT", "SCHEDULED_TIME"]))
