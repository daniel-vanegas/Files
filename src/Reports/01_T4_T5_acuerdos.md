# Selección de escenarios T4 y T5

**Estado:** asignación de mapas calculada; pendiente de auditoría geométrica y de maniobras B0–B4 antes del pre-registro. Los conjuntos no están congelados todavía.

## 1. Conjuntos seleccionados

| Conjunto | Escenarios | Mapas |
|---|---|---:|
| **T4 — `TRAIN_CORE`** | E02, E03, E04, E08, E13, E17, E18, E21, E25, E30 | **10** |
| **`TRAIN_EXPANSION`** | E09, E14, E15, E16, E27 | **5** |
| **T5** | E02, E03, E04, E08, E09, E13, E14, E15, E16, E17, E18, E21, E25, E27, E30 | **15** |
| **`EVAL_REPRESENTATIVE`** | E01, E05, E06, E07, E10, E11, E12, E19, E20, E22, E26, E29 | **12** |
| **`EVAL_DIAGNOSTIC`** | E23, E24, E28 | **3** |
| **`EVAL_HOLDOUT`** | E01, E05, E06, E07, E10, E11, E12, E19, E20, E22, E23, E24, E26, E28, E29 | **15** |

**Relaciones:** `T5 = TRAIN_CORE ∪ TRAIN_EXPANSION`; `EVAL_HOLDOUT = EVAL_REPRESENTATIVE ∪ EVAL_DIAGNOSTIC`; `T5 ∩ EVAL_HOLDOUT = ∅`. Los 30 escenarios aparecen exactamente una vez entre T5 y EVAL_HOLDOUT.

## 2. Justificación

- **Reserva ambiental:** T4 utiliza 10/30 mapas (33.3 %) y T5, 15/30 (50 %). Quedan 15 mapas completos de evaluación común.
- **Selección reproducible:** la asignación se calculó con la matriz Wasserstein Sinkhorn de 30×30, restricciones de representación de seis familias del generador y cobertura ambiental; no se utilizaron éxitos históricos para escoger mapas. La distancia Wasserstein también responde a escala física, por lo que no equivale por sí sola a novedad topológica.
- **Suficiencia de entrenamiento:** cada escenario de T4 y T5 dispone de al menos 145 episodios y 30 de cada clase global (`straight`, `semi_complex`, `complex`, `v_complex`). E05 y E29 tienen menos episodios; E26 presenta menor cobertura de `v_complex`, y por ello se reservaron para evaluación.
- **Anidamiento:** los cinco escenarios de `TRAIN_EXPANSION` permiten estudiar ampliación ambiental sin cambiar los diez escenarios del núcleo. Una comparación a igual presupuesto supervisado controla el volumen de objetivos de entrenamiento, aunque no separa por completo todos los cambios en la distribución de demostraciones.
- **Diagnóstico:** E23 y E24 son un par diagnóstico (distancia mutua publicada de 0.0728), no un componente conexo al umbral 0.06; E28 es un caso distante, cuyo vecino más próximo en la matriz es E18 (0.0850). Se reportarán por separado.

## 3. Datos disponibles

| Indicador | T4 | T5 |
|---|---:|---:|
| Mapas | 10 | 15 |
| Episodios registrados | 1,791 | 2,690 |
| Muestras registradas | 75,528 | 113,207 |
| Media de distancia mínima de los 15 mapas reservados a entrenamiento | 0.0545 | 0.0527 |

Los conteos son **disponibilidad histórica**, no el presupuesto final de entrenamiento.

## 4. Reglas de experimentación y evaluación

1. Usar el mismo procedimiento de preparación de datos para ambas selecciones. Las clases de ruta y las anotaciones verificadas B0–B4 caracterizarán la cobertura; las clases locales no definirán particiones.
2. Mantener `EVAL_HOLDOUT` fuera del entrenamiento, la validación, la selección de hiperparámetros y la iteración adaptativa de arquitecturas. Es **un test retrospectivo de desarrollo**, no confirmatorio.
3. Comparar T4 y T5 con **igual número de objetivos supervisados**; evaluar separadamente el efecto de aumentar datos. La línea base experimental nueva utilizará la RNN vigente de cuatro capas y 350 unidades, sin confundirla con la LSTM histórica.
4. Utilizar recorridos emparejados y reportar éxito absoluto por mapa, diferencias macro-promediadas, colisiones, clases de ruta y distancia ambiental; Wilson y McNemar exacto por mapa como análisis complementarios.
5. Reportar `EVAL_REPRESENTATIVE` como análisis retrospectivo principal y `EVAL_DIAGNOSTIC` por separado. Los escenarios E31+ serán necesarios para una evaluación prospectiva independiente, con tamaño a justificar estadísticamente.
6. **Antes de congelar:** verificar familias geométricas reales, características físicas, anotaciones B0–B4, integridad de episodios y presupuesto supervisado; documentar y fechar cualquier cambio de selección antes de entrenar.

**Fuentes de la selección:** `30env_wasserstein_sinkhorn_mean.csv` y `dataset_30_scenarios_audited_counts.csv`; selección numérica anterior reproducible mediante `seleccionar_escenarios_T4_T5.py`.