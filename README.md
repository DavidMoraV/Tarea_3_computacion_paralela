# Tarea 3 — Polars vs Pandas: Pipeline de Datos y Machine Learning

**Curso:** Computación Paralela
**Profesor:** Johansell Villalobos Cubillo
**Autor:** David Mora V.

## 1. Descripción del problema

Pipeline completo de análisis de datos y aprendizaje automático construido con **Polars**,
comparando sistemáticamente su rendimiento contra **Pandas** en cada etapa del procesamiento.

**Problema:** clasificación binaria de vuelos — predecir si un vuelo sufrirá un **retraso
significativo en la llegada** (umbral: retraso de llegada > 15 minutos).

## 2. Fuente del dataset

**2015 Flight Delays and Cancellations** — U.S. Department of Transportation (USDOT).

- Kaggle: https://www.kaggle.com/datasets/usdot/flight-delays
- Licencia: CC0 (Dominio Público)
- Archivos: `flights.csv` (~5.8M filas, 31 columnas), `airlines.csv`, `airports.csv`

> El dataset **no se incluye** en el repositorio por su tamaño. Ver "Instrucciones de instalación".

## 3. Requisitos de software

- Python 3.10+
- Dependencias en `requirements.txt`

## 4. Instrucciones de instalación

```bash
# 1. Crear y activar entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar el dataset desde Kaggle a data/raw/
#    (ver opciones en la sección de ejecución)
```

## 5. Instrucciones de ejecución

```bash
jupyter notebook notebooks/Tarea_3_Polars_David.ipynb
```

## 6. Estructura del repositorio

```
Tarea_3/
├── data/
│   ├── raw/           # CSVs originales de Kaggle (ignorados por git)
│   └── processed/     # datos transformados
├── notebooks/
│   └── Tarea_3_Polars_David.ipynb
├── src/
│   ├── polars_pipeline.py
│   ├── pandas_pipeline.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── train_models.py
├── figures/           # gráficas para el informe
├── results/           # tablas y métricas exportadas
├── report/
│   └── report.pdf
├── requirements.txt
└── README.md
```

## 7. Resumen de resultados

_(Se completará al finalizar los experimentos: tablas comparativas Polars vs Pandas,
speedups por etapa, métricas de los modelos y conclusiones.)_
