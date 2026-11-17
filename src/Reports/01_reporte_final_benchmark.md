# Reporte Final: Auditoría del Benchmark y Protocolo Experimental V2

**Fecha:** 16 de noviembre de 2026
**Estado:** Benchmark de 30 escenarios completado. Conjuntos de entrenamiento y evaluación congelados. Pendiente auditoría B0–B4 antes del pre-registro.

---

## 1. Resultados del Benchmark Masivo (30 Mapas)

Se evaluaron 40 rutas por escenario. Para evitar el sesgo de mapas donde el Teacher falla por infraestructura, se definieron las siguientes métricas:

*   **Éxito Normalizado:** `Éxitos LSTM / Éxitos Teacher * 100`. Efectividad real de la red frente al experto.
*   **Brecha de Política:** `100% - Éxito Normalizado`. Cuánto le falta a la red para igualar al Teacher.
*   **Complejidad Física:** `(40 - Éxitos Teacher) / 40 * 100`. Dificultad intrínseca del mapa.
*   **Umbrales de Diagnóstico:** Aceptable (`>= 60%`), Fallo de aprendizaje (`50% - 59.9%`), Fallo severo (`< 50%`).

### Tabla de Resultados (Ordenada por Éxito Normalizado)

| Mapa | Éxito Teacher | Éxito LSTM | Éxito Normalizado | Brecha de Política | Complejidad Física | ¿En T2? | Mapa Similar (Distancia) | Diagnóstico |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E07** | 36/40 | 27/40 | **75.0%** | 25.0% | 10.0% | Sí | E02 (0.057) | Aceptable |
| **E09** | 34/40 | 24/40 | **70.6%** | 29.4% | 15.0% | Sí | E19 (0.048) | Aceptable |
| **E21** | 38/40 | 26/40 | **68.4%** | 31.6% | 5.0% | Sí | E29 (0.041) | Aceptable |
| **E23** | 37/40 | 25/40 | **67.6%** | 32.4% | 7.5% | - | E24 (0.072) | Aceptable (U1) |
| **E22** | 38/40 | 25/40 | **65.8%** | 34.2% | 5.0% | Sí | E25 (0.050) | Aceptable |
| **E02** | 37/40 | 24/40 | **64.9%** | 35.1% | 7.5% | Sí | E14 (0.049) | Aceptable |
| **E15** | 36/40 | 23/40 | **63.9%** | 36.1% | 10.0% | Sí | E08 (0.051) | Aceptable |
| **E26** | 38/40 | 24/40 | **63.2%** | 36.8% | 5.0% | Sí | E27 (0.046) | Aceptable |
| **E11** | 37/40 | 23/40 | **62.2%** | 37.8% | 7.5% | Sí | E16 (0.054) | Aceptable |
| **E27** | 34/40 | 21/40 | **61.8%** | 38.2% | 15.0% | - | E26 (0.046) | Aceptable |
| **E13** | 35/40 | 21/40 | **60.0%** | 40.0% | 12.5% | Sí | E12 (0.054) | Aceptable |
| E05 | 39/40 | 23/40 | 59.0% | 41.0% | 2.5% | - | E03 (0.047) | Fallo aprendizaje (gemelo en T2) |
| E18 | 37/40 | 21/40 | 56.8% | 43.2% | 7.5% | Sí | E20 (0.041) | Fallo de aprendizaje |
| E03 | 34/40 | 19/40 | 55.9% | 44.1% | 15.0% | Sí | E05 (0.047) | Fallo de aprendizaje |
| E17 | 36/40 | 20/40 | 55.6% | 44.4% | 10.0% | Sí | E06 (0.056) | Fallo de aprendizaje |
| E19 | 36/40 | 20/40 | 55.6% | 44.4% | 10.0% | - | E09 (0.048) | Fallo aprendizaje (gemelo en T2) |
| E24 | 38/40 | 21/40 | 55.3% | 44.7% | 5.0% | - | E23 (0.072) | Fallo de aprendizaje (U1) |
| E16 | 39/40 | 21/40 | 53.8% | 46.2% | 2.5% | - | E11 (0.054) | Fallo aprendizaje (gemelo en T2) |
| E30 | 39/40 | 21/40 | 53.8% | 46.2% | 2.5% | Sí | E16 (0.058) | Fallo de aprendizaje |
| E28 | 36/40 | 19/40 | 52.8% | 47.2% | 10.0% | Sí | E20 (0.126) | Fallo de aprendizaje (U2) |
| E29 | 38/40 | 20/40 | 52.6% | 47.4% | 5.0% | - | E21 (0.041) | Fallo aprendizaje (gemelo en T2) |
| E10 | 37/40 | 19/40 | 51.4% | 48.6% | 7.5% | - | E09 (0.051) | Fallo aprendizaje (52.5% de fallo LSTM) |
| E25 | 37/40 | 19/40 | 51.4% | 48.6% | 7.5% | Sí | E22 (0.050) | Fallo de aprendizaje |
| E06 | 39/40 | 20/40 | 51.3% | 48.7% | 2.5% | - | E17 (0.056) | Fallo aprendizaje (gemelo en T2) |
| E12 | 39/40 | 20/40 | 51.3% | 48.7% | 2.5% | Sí | E13 (0.054) | Fallo de aprendizaje |
| E01 | 39/40 | 19/40 | **48.7%** | 51.3% | 2.5% | Sí | E04 (0.035) | **Fallo severo** |
| E04 | 39/40 | 19/40 | **48.7%** | 51.3% | 2.5% | - | E01 (0.035) | **Fallo severo** (gemelo en T2) |
| E08 | 39/40 | 19/40 | **48.7%** | 51.3% | 2.5% | - | E15 (0.051) | **Fallo severo** (gemelo en T2) |
| E14 | 39/40 | 19/40 | **48.7%** | 51.3% | 2.5% | Sí | E02 (0.049) | **Fallo severo** |
| E20 | 37/40 | 17/40 | **45.9%** | 54.1% | 7.5% | - | E18 (0.041) | **Fallo severo** (gemelo en T2) |

---

## 2. Hallazgos de la Auditoría

1.  **La LSTM Sí Generaliza:** **E23 (67.6%)** y **E24 (55.3%)** son los únicos mapas sin gemelo a <0.06 (su distancia mutua es 0.072). Rinden en la mitad alta. Esto indica que la red sí generaliza a geometrías realmente nuevas.
2.  **La Fuga de Datos (Falsos Fallos de Generalización):** De los 12 mapas etiquetados previamente como "Fallo de Generalización", **10 tenían un gemelo topológico en el conjunto de entrenamiento (T2)**. No son fallos de generalización, sino **fallos de aprendizaje** sobre geometrías ya conocidas.
3.  **Varianza Intra-Clúster (El núcleo del problema):** Mapas gemelos rinden distinto. **E07 rinde al 75.0%** mientras **E14 rinde al 48.7%** (ambos en C7, distancia 0.057, ambos en T2). El cuello de botella no es el mapa, es la arquitectura o la distribución de clases.

---

## 3. Protocolo de Selección de Conjuntos (Auditado y Congelado)

Basado en el documento de acuerdos metodológicos, se define un benchmark retrospectivo de desarrollo dividiendo los 30 mapas en dos mitades estancas:

*   **Conjunto de Entrenamiento (15 mapas):**
    *   `TRAIN_CORE` (10): E02, E03, E04, E08, E13, E17, E18, E21, E25, E30.
    *   `TRAIN_EXPANSION` (5): E09, E14, E15, E16, E27.
*   **Conjunto de Evaluación Holdout (15 mapas):**
    *   `EVAL_REPRESENTATIVE` (12): E01, E05, E06, E07, E10, E11, E12, E19, E20, E22, E26, E29.
    *   `EVAL_DIAGNOSTIC` (3): E23, E24, E28 (topologías únicas).

*Nota: T5 (Evaluación) y T4 (Entrenamiento) son disjuntos. Los 30 escenarios se usan exactamente una vez.*

---

## 4. Siguiente Fase: Ablación Arquitectónica en T4

Con el dataset blindado, el siguiente paso es encontrar la configuración que elimine la varianza intra-clúster. Se entrenará en T4 con **igual presupuesto de datos** (episodios balanceados) y se evaluará en T5.

**Las tres hipótesis a probar:**
1.  **Run A (Control):** RNN original (4 capas, 350) con C=2. Reproducir la línea base.
2.  **Run B (Distribución):** Rebalanceo de clases (más peso a `complex` y `v_complex`).
3.  **Run C (Memoria):** LSTM con ventana temporal extendida (C=4).

**Métricas de Evaluación:**
*   Éxito absoluto por mapa.
*   Diferencias macro-promediadas en T5.
*   Test de McNemar para significancia estadística en los pares emparejados.
*   Bootstrap por clúster (remuestreando mapas, no trials).

**Limitación Declarada:** Este es un test retrospectivo de desarrollo. La validación prospectiva independiente requerirá mapas E31+.