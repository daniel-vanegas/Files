# Resumen de Selección de Escenarios

## 1. Diseño Experimental de Escenarios

Se definieron tres conjuntos para medir la capacidad de generalización del modelo:

| Conjunto | Escenarios | Rol |
| :--- | :--- | :--- |
| **T0** | E01, E02, E03 | Línea base histórica |
| **T1** | E01-E03 + E12, E17, E22, E27, E30 | Cobertura mínima (1 por familia SC=3, 5, 8, 10, 30) |
| **T2** | E01-E03 + 15 adicionales | Cobertura densa con redundancia |
| **Desarrollo** | E08, E16, E20 | Selección de modelos |
| **Test final** | E04, E05, E06, E10, E19, E23, E24, E29 | Reserva intocable |

### Verificación Ambiental (Matriz Wasserstein/Sinkhorn)

- **T1** y **T2** presentan una diversidad interna equivalente ($0.0895$ vs $0.0899$).
- **T1** representa un test más exigente: *nearest-train* de test = $0.0595$ vs $0.0488$ (**T2**).
- La diferencia se debe casi exclusivamente a que **T1** deja la instancia `E28` fuera del conjunto de entrenamiento (*train*), mientras que **T2** la absorbe.

#### Métrico de Distancia *Nearest-Train*
Para cada escenario del conjunto de test, se calcula la distancia al escenario más cercano de *train*, promediando las distancias mínimas:
$$\bar{d}_{\min}(T_{\text{test}}, T_{\text{train}}) = \frac{1}{|T_{\text{test}}|} \sum_{t \in T_{\text{test}}} \min_{s \in T_{\text{train}}} D(t,s)$$

Donde $D$ es la distancia Wasserstein/Sinkhorn (adimensional) entre las distribuciones geométricas de dos mapas calculada sobre las LOM.

| Valor | Interpretación | Significado |
| :---: | :--- | :--- |
| **0.0595 (T1)** | Menor similitud con *train* | En promedio, cada escenario de test dista $0.0595$ de su vecino más cercano. Tarea **más exigente** (+22% más lejos). |
| **0.0488 (T2)** | Mayor similitud con *train* | En promedio, cada escenario de test dista $0.0488$ de su vecino más cercano. Tarea **más fácil**. |

> **Nota:** Valores bajos ($\sim 0.03\text{–}0.05$) indican mapas casi idénticos; valores altos ($\sim 0.10\text{–}0.17$) reflejan mapas significativamente distintos.

---

## 2. Etiquetado Humano

Se construyó un visor web (`review.html`) para etiquetar intervalos de pares con $5$ comportamientos locales diferenciados:

| Clase | Descripción | Pares Etiquetados | Porcentaje |
| :---: | :--- | :---: | :---: |
| **B0** | Avance libre | 245 | 15.0% |
| **B1** | Evasión simple | 688 | 42.1% |
| **B2** | Evasión restringida | 534 | 32.7% |
| **B3** | Rodeo frontal | 34 | 2.1% |
| **B4** | Recuperación | 134 | 8.2% |
| **Total** | | **1,635** | **100.0%** |

> **Sesgo detectado:** La clase **B1** domina el conjunto ($42.1\%$), mientras que **B3** se encuentra casi ausente ($2.1\%$). Esto motivó la exploración de estrategias de selección balanceada de datos.