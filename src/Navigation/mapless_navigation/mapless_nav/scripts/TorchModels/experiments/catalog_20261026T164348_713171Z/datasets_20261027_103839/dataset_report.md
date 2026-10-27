# Preparación de datasets ROS 2

Semilla de partición conservada: `1899493870`.

Semilla de selección: `2895327133`.

Catálogo SHA256: `548e0b2d18223a8b07d58c25832b41dff5426c9420bee277b0787917c672ce22`.

Los manifiestos contienen pares de dos observaciones consecutivas, sin solapamiento. La etiqueta procede de la segunda observación. Los NPZ y la LOM 0/100 se conservan.

`ready` indica que el manifiesto pasó las comprobaciones de preparación. El entrenamiento y la navegación simulada se validan en fases posteriores.

| Configuración | Estado | Pares seleccionados | Pares de validación | Fracción del entrenamiento disponible |
|---|---|---:|---:|---:|
| T0/D0 | ready | 9,463 | 2,384 | 100.0% |
| T0/D1 | needs_annotations | 0 | 2,384 | 0.0% |
| T0/D2 | needs_annotations | 0 | 2,384 | 0.0% |
| T0/D3 | needs_annotations | 0 | 2,384 | 0.0% |
| T1/D0 | ready | 25,305 | 6,430 | 100.0% |
| T1/D1 | needs_annotations | 0 | 6,430 | 0.0% |
| T1/D2 | needs_annotations | 0 | 6,430 | 0.0% |
| T1/D3 | needs_annotations | 0 | 6,430 | 0.0% |
| T2/D0 | ready | 53,576 | 13,237 | 100.0% |
| T2/D1 | ready | 156 | 13,237 | 0.3% |
| T2/D2 | ready | 180 | 13,237 | 0.3% |
| T2/D3 | needs_annotations | 0 | 13,237 | 0.0% |

## Comprobaciones

La partición guardada coincide con su semilla. No hay episodios ni contenido compartido entre entrenamiento y validación. Los manifiestos escritos se leen de nuevo y se comprueban sus observaciones, etiquetas, duplicados y roles. Cada conjunto ambiental comparte un único manifiesto de validación entre todas sus recetas.

Límite por episodio y comportamiento: `ceil(0.1 × cuota)` pares. Las evasiones seleccionadas requieren al menos 2 episodios distintos por lado y el mismo número de pares izquierdos y derechos. La selección no repite pares.

Las cuotas se redondean hacia abajo y las direccionales se ajustan a números pares. El presupuesto se reduce cuando hace falta para cumplir los límites sin cambiar las proporciones solicitadas.

## Cobertura revisada de T0

Pares de entrenamiento disponibles: 9,463. Con clase local revisada: 90. Sin clase local asignada: 9,373. Con clase direccional y lado desconocido: 0.

Las anotaciones sin clase, pendientes y con lado desconocido se conservan en D0. No completan cuotas de evasión balanceada. Esta cobertura describe las anotaciones disponibles, no demuestra ausencia de comportamientos en el resto del dataset.

| Comportamiento/lado | Pares | Episodios | Escenarios |
|---|---:|---:|---:|
| B1/left | 57 | 3 | 3 |
| B1/right | 9 | 1 | 1 |
| B2/left | 24 | 3 | 3 |
| unassigned/not_applicable | 8 | 1 | 1 |
| unassigned/right | 11 | 1 | 1 |
| unassigned/unassigned | 9,354 | 450 | 3 |

Las maniobras globales se conservan como contexto. No convierten sus pares en etiquetas locales B3/B4.

| Maniobra/estado | Intervalos | Pares cubiertos | Episodios |
|---|---:|---:|---:|
| frontal_detour/reviewed | 1 | 33 | 1 |
| goal_progress/reviewed | 1 | 18 | 1 |
| lateral_avoidance/reviewed | 1 | 27 | 1 |
| restricted_avoidance/reviewed | 1 | 25 | 1 |

## Cobertura revisada de T1

Pares de entrenamiento disponibles: 25,305. Con clase local revisada: 191. Sin clase local asignada: 25,114. Con clase direccional y lado desconocido: 10.

Las anotaciones sin clase, pendientes y con lado desconocido se conservan en D0. No completan cuotas de evasión balanceada. Esta cobertura describe las anotaciones disponibles, no demuestra ausencia de comportamientos en el resto del dataset.

| Comportamiento/lado | Pares | Episodios | Escenarios |
|---|---:|---:|---:|
| B0/neutral | 13 | 2 | 2 |
| B1/left | 110 | 9 | 7 |
| B1/right | 20 | 2 | 2 |
| B2/left | 30 | 4 | 4 |
| B2/right | 8 | 1 | 1 |
| B2/unknown | 10 | 2 | 2 |
| unassigned/left | 4 | 1 | 1 |
| unassigned/not_applicable | 8 | 1 | 1 |
| unassigned/right | 19 | 2 | 2 |
| unassigned/unassigned | 25,083 | 1,213 | 8 |

Las maniobras globales se conservan como contexto. No convierten sus pares en etiquetas locales B3/B4.

| Maniobra/estado | Intervalos | Pares cubiertos | Episodios |
|---|---:|---:|---:|
| frontal_detour/reviewed | 3 | 82 | 3 |
| goal_progress/reviewed | 3 | 33 | 3 |
| lateral_avoidance/reviewed | 4 | 73 | 4 |
| restricted_avoidance/reviewed | 2 | 44 | 2 |

## Cobertura revisada de T2

Pares de entrenamiento disponibles: 53,576. Con clase local revisada: 462. Sin clase local asignada: 53,114. Con clase direccional y lado desconocido: 22.

Las anotaciones sin clase, pendientes y con lado desconocido se conservan en D0. No completan cuotas de evasión balanceada. Esta cobertura describe las anotaciones disponibles, no demuestra ausencia de comportamientos en el resto del dataset.

| Comportamiento/lado | Pares | Episodios | Escenarios |
|---|---:|---:|---:|
| B0/neutral | 62 | 9 | 8 |
| B1/left | 168 | 13 | 10 |
| B1/right | 49 | 6 | 6 |
| B2/left | 118 | 11 | 10 |
| B2/right | 43 | 6 | 5 |
| B2/unknown | 22 | 3 | 3 |
| unassigned/left | 4 | 1 | 1 |
| unassigned/not_applicable | 8 | 1 | 1 |
| unassigned/right | 23 | 3 | 3 |
| unassigned/unassigned | 53,079 | 2,569 | 18 |

Las maniobras globales se conservan como contexto. No convierten sus pares en etiquetas locales B3/B4.

| Maniobra/estado | Intervalos | Pares cubiertos | Episodios |
|---|---:|---:|---:|
| door_passage/pending_review | 1 | 21 | 1 |
| door_passage/reviewed | 1 | 18 | 1 |
| frontal_detour/reviewed | 8 | 204 | 8 |
| goal_progress/pending_review | 1 | 15 | 1 |
| goal_progress/reviewed | 10 | 122 | 10 |
| lateral_avoidance/reviewed | 8 | 145 | 8 |
| restricted_avoidance/reviewed | 5 | 117 | 5 |

## T0/D0

Estado: `ready`.

## T0/D1

Estado: `needs_annotations`.

- B1/right: insufficient distinct reviewed episodes

- B2/right: insufficient distinct reviewed episodes

## T0/D2

Estado: `needs_annotations`.

- B0: no reviewed pairs

- B1/right: insufficient distinct reviewed episodes

- B2/right: insufficient distinct reviewed episodes

## T0/D3

Estado: `needs_annotations`.

- B0: no reviewed pairs

- B1/right: insufficient distinct reviewed episodes

- B2/right: insufficient distinct reviewed episodes

- B3/left: insufficient distinct reviewed episodes

- B3/right: insufficient distinct reviewed episodes

- B4/left: insufficient distinct reviewed episodes

- B4/right: insufficient distinct reviewed episodes

## T1/D0

Estado: `ready`.

## T1/D1

Estado: `needs_annotations`.

- B2/right: insufficient distinct reviewed episodes

## T1/D2

Estado: `needs_annotations`.

- B2/right: insufficient distinct reviewed episodes

## T1/D3

Estado: `needs_annotations`.

- B2/right: insufficient distinct reviewed episodes

- B3/left: insufficient distinct reviewed episodes

- B3/right: insufficient distinct reviewed episodes

- B4/left: insufficient distinct reviewed episodes

- B4/right: insufficient distinct reviewed episodes

## T2/D0

Estado: `ready`.

## T2/D1

Estado: `ready`.

Presupuesto nominal inicial: 186. Presupuesto nominal utilizado: 158. Pares seleccionados después del redondeo: 156.

| Comportamiento | Pares | Proporción solicitada | Proporción obtenida | Máximo por episodio |
|---|---:|---:|---:|---:|
| B0 | 52 | 33.3% | 33.3% | 6 |
| B1 | 52 | 33.3% | 33.3% | 6 |
| B2 | 52 | 33.3% | 33.3% | 6 |

## T2/D2

Estado: `ready`.

Presupuesto nominal inicial: 215. Presupuesto nominal utilizado: 184. Pares seleccionados después del redondeo: 180.

| Comportamiento | Pares | Proporción solicitada | Proporción obtenida | Máximo por episodio |
|---|---:|---:|---:|---:|
| B0 | 36 | 20.0% | 20.0% | 4 |
| B1 | 72 | 40.0% | 40.0% | 8 |
| B2 | 72 | 40.0% | 40.0% | 8 |

## T2/D3

Estado: `needs_annotations`.

- B3/left: insufficient distinct reviewed episodes

- B3/right: insufficient distinct reviewed episodes

- B4/left: insufficient distinct reviewed episodes

- B4/right: insufficient distinct reviewed episodes
