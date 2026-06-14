# Tarea 3 — Polars vs Pandas: Pipeline de Datos y Machine Learning

**Curso:** Computación Paralela
**Profesor:** Johansell Villalobos Cubillo
**Autor:** David Mora V.

## 1. Descripción del problema

Pipeline completo de análisis de datos y aprendizaje automático construido con **Polars**,
comparando sistemáticamente su rendimiento contra **Pandas** en cada etapa del procesamiento.

**Problema:** clasificación binaria — predecir si un vuelo sufrirá un **retraso significativo en la
llegada** (`ARRIVAL_DELAY > 15 min`). La clase positiva está desbalanceada (~18%).

## 2. Fuente del dataset

**2015 Flight Delays and Cancellations** — U.S. Department of Transportation (USDOT).

- Kaggle: https://www.kaggle.com/datasets/usdot/flight-delays
- Licencia: CC0 (Dominio Público)
- Archivos: `flights.csv` (5,819,079 filas, 31 columnas), `airlines.csv`, `airports.csv`

> El dataset **no se incluye** en el repositorio por su tamaño. Ver "Instrucciones de instalación".

## 3. Requisitos de software

- Python 3.10+
- Dependencias en `requirements.txt`

## 4. Instrucciones de instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Descargar el dataset a data/raw/
pip install kaggle
kaggle datasets download usdot/flight-delays -p data/raw/ --unzip
# (o descarga manual desde la URL de Kaggle)
```

## 5. Instrucciones de ejecución

```bash
# Notebook principal
jupyter notebook notebooks/Tarea_3_computacion_paralela.ipynb

# O ejecutar los módulos por separado
python src/polars_pipeline.py
python src/pandas_pipeline.py
python src/train_models.py
```

## 6. Estructura del repositorio

```
Tarea_3/
├── data/
│   ├── raw/           # CSVs originales de Kaggle (ignorados por git)
│   └── processed/     # datos transformados
├── notebooks/
│   └── Tarea_3_computacion_paralela.ipynb
├── src/
│   ├── preprocessing.py        # carga, objetivo y limpieza
│   ├── feature_engineering.py  # features, join y group_by
│   ├── polars_pipeline.py      # pipeline Polars cronometrado
│   ├── pandas_pipeline.py      # pipeline Pandas cronometrado
│   └── train_models.py         # entrenamiento y evaluación
├── figures/           # gráficas para el informe
├── results/           # tablas y métricas exportadas
├── report/
│   └── report.pdf
├── requirements.txt
└── README.md
```

## 7. Resumen de resultados

### Benchmark Polars vs Pandas (dataset completo)

| Etapa | Polars (s) | Pandas (s) | Speedup |
|-------|-----------|-----------|---------|
| Lectura | 1.419 | 21.232 | **14.97x** |
| Filtrado | 0.683 | 3.485 | **5.10x** |
| Feature engineering | 0.082 | 0.590 | **7.16x** |
| Joins | 2.249 | 1.797 | 0.80x |
| Agregación | 0.667 | 2.596 | **3.89x** |
| **TOTAL** | **5.100** | **29.699** | **5.82x** |

Polars destaca en lectura de CSV y transformaciones vectorizadas. El único punto a favor de Pandas
fue el join con tablas pequeñas (`airlines.csv`).

### Escalabilidad

| Porcentaje | Filas | Speedup |
|-----------|-------|---------|
| 25% | 1,454,769 | 5.47x |
| 50% | 2,909,539 | 12.07x |
| 75% | 4,364,309 | 15.60x |
| 100% | 5,819,079 | **15.72x** |

El speedup **crece con el tamaño** del dataset (promedio 12.21x): Polars es más beneficioso cuanto
mayor es el volumen de datos.

### Lazy Execution (Eager vs Lazy)

| Modo | Tiempo total (s) | Memoria pico (MB) |
|------|-----------------|-------------------|
| Eager (`read_csv`) | 11.768 | 3934.3 |
| Lazy (`scan_csv`) | **5.926** | **1429.6** |

La ejecución lazy fue **~2x más rápida** y usó **~64% menos memoria**, gracias a *projection* y
*predicate pushdown*.

### Modelos de Machine Learning

| Modelo | Tiempo (s) | Accuracy | F1 | AUC |
|--------|-----------|----------|----|----|
| Regresión Logística | 17.69 | 0.9257 | 0.8120 | 0.9653 |
| Random Forest | 301.98 | 0.9436 | 0.8448 | 0.9701 |
| Gradient Boosting (Hist) | 126.47 | 0.9314 | 0.8242 | **0.9719** |

**Mejor modelo:** Gradient Boosting (mejor AUC y buen balance de tiempo). Variable más importante:
**DEPARTURE_DELAY**.

## 8. Conclusiones

- Polars ofrece una ventaja de rendimiento sustancial sobre Pandas (5.82x total, hasta 15.72x), con
  un beneficio que **aumenta con el tamaño** de los datos.
- La evaluación diferida (lazy) duplica la velocidad y reduce la memoria ~64%, clave para datasets
  que no caben en RAM.
- Los tres modelos superan AUC 0.96; Gradient Boosting da el mejor balance desempeño/tiempo.
- Estrategia recomendada: **Polars para el ETL pesado** y el ecosistema **Pandas/numpy en la etapa
  final** de modelado y visualización.
