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
#    Opción A: Usar Kaggle CLI
#    pip install kaggle
#    kaggle datasets download usdot/flight-delays -p data/raw/ --unzip
#    
#    Opción B: Descarga manual desde https://www.kaggle.com/datasets/usdot/flight-delays

# Instrucciones de ejecución
jupyter notebook notebooks/Tarea_3_computacion_paralela.ipynb

# Estructura del repositorio
Tarea_3/
├── data/
│   ├── raw/           # CSVs originales de Kaggle (ignorados por git)
│   └── processed/     # datos transformados
├── notebooks/
│   └── Tarea_3_computacion_paralela.ipynb
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

##  Resumen de resultados

###  Benchmark Polars vs Pandas

| Etapa | Polars (s) | Pandas (s) | Speedup |
|-------|------------|------------|---------|
| lectura | 1.419 | 21.232 | **14.97x** |
| filtrado | 0.683 | 3.485 | **5.10x** |
| feature_eng | 0.082 | 0.590 | **7.16x** |
| joins | 2.249 | 1.797 | 0.80x |
| agregacion | 0.667 | 2.596 | **3.89x** |
| **TOTAL** | **5.100** | **29.699** | **5.82x** |

**Observaciones:** 
- Polars es excepcionalmente rápido en lectura de CSV (~15x más rápido)
- Las transformaciones vectorizadas son muy eficientes (7.16x)
- El único punto donde Pandas fue más rápido fue en joins con tablas pequeñas (airlines.csv)

###  Escalabilidad

| Porcentaje | Filas | Polars (s) | Pandas (s) | Speedup |
|------------|-------|------------|------------|---------|
| 25% | 1,454,769 | 0.238 | 1.301 | 5.47x |
| 50% | 2,909,539 | 0.194 | 2.342 | 12.07x |
| 75% | 4,364,309 | 0.272 | 4.250 | 15.60x |
| 100% | 5,819,079 | 0.316 | 4.971 | **15.72x** |

**Speedup promedio:** 12.21x  
**Tendencia:** El speedup **aumenta** con el tamaño del dataset → Polars es más beneficioso para datasets grandes.

###  Lazy Execution (Eager vs Lazy)

| Método | Tiempo total (s) |
|--------|------------------|
| Eager (read_csv) | X.XXX |
| Lazy (scan_csv) | X.XXX |

*Resultados pendientes de completar*

###  Modelos de Machine Learning

| Modelo | Tiempo entrenamiento (s) | Accuracy | F1 | AUC |
|--------|-------------------------|----------|----|-----|
| Regresión Logística | 17.69 | 0.9257 | 0.812 | 0.9653 |
| Random Forest | 301.98 | 0.9436 | 0.8448 | 0.9701 |
| Gradient Boosting | 126.47 | 0.9314 | 0.8242 | **0.9719** |

**Mejor modelo:** Gradient Boosting (Hist) con AUC = 0.9719

###  Variables más importantes (Random Forest)

1. **DEPARTURE_DELAY** (retraso de salida)
2. **SCHEDULED_TIME** (tiempo programado de vuelo)
3. **TAXI_OUT** (tiempo de taxi en origen)
4. **DISTANCE** (distancia del vuelo)
5. **HORA_SALIDA** (hora programada de salida)

##  Conclusiones

*(Se completarán al finalizar todos los experimentos)*
