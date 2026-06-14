# 8. Análisis de Resultados

**1. ¿Qué ventajas observó al utilizar Polars?**
Mayor velocidad en todo el pipeline (5.82× más rápido que Pandas en total, hasta 14.97× en lectura), menor consumo de memoria (la ejecución lazy usó ~64% menos memoria pico), paralelización multinúcleo automática, una API de expresiones vectorizadas muy concisa, y la evaluación diferida (lazy) con optimización del plan de consultas.

**2. ¿Qué operaciones obtuvieron el mayor speedup?**
La **lectura de CSV** fue la más beneficiada (14.97×), seguida del **feature engineering vectorizado** (7.16×) y el **filtrado** (5.10×). Son operaciones intensivas en I/O y cómputo columnar, donde la arquitectura de Polars (Apache Arrow + paralelismo) rinde al máximo.

**3. ¿En cuáles operaciones la diferencia fue pequeña?**
En los **joins con tablas pequeñas** (`airlines.csv`), donde Pandas fue incluso un poco más rápido (0.80×). Con tablas chicas, el overhead de coordinación paralela de Polars no se amortiza y no hay ganancia.

**4. ¿Qué beneficios aporta Lazy Execution?**
Sobre el dataset completo, la versión lazy (`scan_csv().collect()`) fue ~2× más rápida (11.77s → 5.93s) y usó ~64% menos memoria pico (3934 MB → 1430 MB) que la versión eager. Esto se debe al *projection pushdown* (lee solo las columnas necesarias) y al *predicate pushdown* (aplica los filtros durante el escaneo), evitando materializar datos que luego se descartan. Permite procesar datasets que no cabrían en memoria con el enfoque eager.

**5. ¿Qué limitaciones encontró en Polars?**
Ecosistema más nuevo y menos maduro que Pandas; menor integración directa con librerías que esperan pandas/numpy (scikit-learn, matplotlib), obligando a conversiones; sin ventaja —o desventaja— en operaciones sobre tablas pequeñas; curva de aprendizaje de la API de expresiones; y diferencias semánticas en el manejo de nulos (p. ej. `NaN > 15` produce `null` en Polars en vez de `False`).

**6. ¿Qué ventajas mantiene Pandas?**
Ecosistema maduro e integración nativa con prácticamente todo el stack de ciencia de datos, abundante documentación y soporte de comunidad, mejor desempeño relativo en datasets pequeños (sin overhead de paralelización), y una API ampliamente conocida.

**7. ¿La aceleración observada justifica migrar un proyecto existente?**
Depende del caso. Para pipelines con datasets grandes (>1M filas) y operaciones pesadas de lectura/agregación, los 5–15× de speedup justifican la migración. Para proyectos pequeños o muy acoplados al ecosistema Pandas, el costo de migrar puede no compensar. Recomendación práctica: migrar primero las etapas que son cuello de botella, no el proyecto completo.

**8. ¿Cómo afecta el tamaño del dataset al beneficio obtenido?**
El speedup **crece con el tamaño**: 5.47× al 25% del dataset, 12.07× al 50%, 15.60× al 75% y 15.72× al 100% (promedio 12.21×). Cuanto más grande el dataset, mayor el beneficio de Polars, porque se amortizan mejor el paralelismo y la arquitectura columnar.

**9. ¿Qué modelo produjo el mejor desempeño predictivo?**
Por AUC, el **Gradient Boosting (HistGradientBoosting)** con AUC = 0.9719. El **Random Forest** lidera en accuracy (0.9436) y F1 (0.8448) pero es el más lento (302s). La **Regresión Logística** es muy competitiva (AUC 0.9653) y la más rápida (18s). El Gradient Boosting ofrece el mejor balance rendimiento/tiempo. En todos, `DEPARTURE_DELAY` (retraso de salida) fue el predictor dominante.

**10. ¿Qué recomendaciones daría para proyectos futuros?**
Usar Polars (preferentemente en modo lazy con `scan_csv`) para el ETL y procesamiento de datos grandes; mantener Pandas/numpy solo en el borde final para interoperar con ML y visualización; convertir entre formatos lo más tarde posible; preferir `HistGradientBoosting` o `LightGBM` sobre `RandomForest` cuando el tiempo importa en datasets grandes; y cuidar la fuga de datos al diseñar las características (excluir variables conocidas solo después del evento a predecir).
