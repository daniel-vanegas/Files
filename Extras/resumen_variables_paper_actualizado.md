**Resumen de variables y definiciones del paper**

Las arquitecturas de referencia acordadas son **Param_CNN_B** y **RNN**.

**Convenciones de navegación, LOM y entrenamiento.**

| Variable o término | Significado | Valor o dimensión | Estado |
|---|---|---|---|
| \(N\) | Número de píxeles por lado de la imagen LOM. | \(N=80\). | Definido |
| \(C\) | Número de fotogramas consecutivos, representados como canales de entrada. | CNN: 4; RNN: 2. | Está explicado en el texto. |
| Batch size | Número de muestras agrupadas en un lote; cada muestra contiene \(C\) fotogramas. | CNN: 8; RNN: 3. | Explicado; no necesita un símbolo adicional. |
| \(d\) | Distancia del objetivo respecto al robot. | Magnitud física en metros. | Significado definido; puede explicitarse la unidad una vez. |
| \(\theta\) | Ángulo del objetivo respecto al marco del robot. | Ángulo relativo. | Significado definido; precisar una vez unidad y convención angular. |
| \(d_c,\theta_c\) | Componentes del objetivo asociado al canal/fotograma \(c\). | Dos componentes por fotograma. | Definidos por contexto. |
| \(v_x,v_y\) | Velocidades lineales sobre los ejes del robot. | Comandos físicos en m/s. | Definidas; distinguirlas de sus etiquetas normalizadas cuando se detalle el entrenamiento. |
| \(\omega\) | Velocidad angular sobre el eje vertical del robot. | Comando físico en rad/s. | Definida; también se usa ahora en PF con el mismo significado. |
| \(i\) | Índice de muestra en la formulación del dataset. | Índice de ejemplo. | Su reutilización como índice de repetición en `rep_tgt` debe aclararse. |
| \(c\) | Índice de canal/fotograma dentro de una muestra. | \(c=1,\ldots,C\). | Definido por contexto. |
| \(\mathbf{LOM}^{(i)}\) | LOM agrupadas de la muestra \(i\). | \(\{0,100\}^{N\times N\times C}\). | La dimensión expresa la agrupación; puede conservarse el nombre. |
| \(\mathbf{tgt}^{(i,c)}\) | Vector del objetivo del fotograma \(c\) de la muestra \(i\). | \((d^{(i,c)},\theta^{(i,c)})\in\mathbb R^2\). | El vector está bien declarado; corregir la frase que lo llama una secuencia completa. |
| \(\mathbf{tgt}^{(i)}\) | Agrupación de los objetivos de una muestra. | Una representación posible es \(2\times C\). | Falta declarar explícitamente la relación con los vectores por fotograma. |
| \(\mathbf v^{(i,c)}\) | Velocidad de referencia asociada al fotograma \(c\). | \((v_x^{(i,c)},v_y^{(i,c)},\omega^{(i,c)})\in\mathbb R^3\). | El tipo es correcto; la descripción no debe confundirlo con la secuencia completa. |
| \(\mathrm{rep\_tgt}\) | Representación repetida del objetivo para su concatenación con características. | El texto aún declara \(\mathbb R^{2NC}\). | Pendiente de la refinación acordada por arquitectura. |
| \(\mathrm{Label}^{(i)}\) | Actualmente nombra la tupla completa de entradas y salida. | \(((\mathbf{LOM}^{(i)},\mathbf{tgt}^{(i)}),\mathbf v^{(i)})\). | Distinguir ejemplo completo de entrenamiento y etiqueta de referencia. |
| \(m\) | Longitud física del lado de la región representada por la LOM. | 4 m. | Ya definido. |
| \(\mathrm{Res}\) | Resolución expresada como cantidad de píxeles por metro. | El texto actual dice 20 cells/m. | La relación \(N=m\,\mathrm{Res}\) es coherente; unificar la redacción a pixels/m. |
| \(SC\) | Parámetro de configuración de subdivisiones del generador. | Original E1–E6: 2; conjunto ampliado: SC2–SC30. | Definido y diferenciado de \(C\). No fija por sí solo habitaciones, muebles o dificultad de navegación. |
| \(x,h\) | Entrada y estado oculto de una celda RNN. | Vectores de la celda recurrente. | El caption explicativo continúa comentado; falta una definición activa. |
| LVx, LVy, AVz | Etiquetas gráficas de las tres velocidades. | Equivalentes a \(v_x,v_y,\omega\). | Alias comprendidos; unificación visual reservada para la edición final. |

Los hiperparámetros ya descritos en la tabla no necesitan símbolos nuevos: learning rate, epochs, patience, Adam, filtros, kernels, pooling, dropout, unidades ocultas y capas recurrentes. **MSE** está definido como criterio de entrenamiento y validación; su normalización y agregación requieren precisión metodológica al cerrar la descripción de los experimentos.

**Similitud ambiental.** Las sustituciones \(N\rightarrow n_p\), \(C,C_{ij}\rightarrow\mathbf D,D_{ij}\) y \(\gamma,\gamma_{ij}\rightarrow\boldsymbol\Gamma,\Gamma_{ij}\) están integradas en la prosa, ecuaciones, captions y encabezados.

| Símbolo | Significado en el manuscrito | Valor o dimensión | Estado |
|---|---|---|---|
| \(\mathcal E\) | Ambiente representado para el análisis geométrico. | Un ambiente. | Definido. |
| \(\mathcal M\) | Mapa binario global empleado fuera de línea para representar el ambiente. | Matriz de ocupación. | Definido; distinto de la LOM local de la política. |
| \(\mathcal P\) | Conjunto de coordenadas de puntos ocupados del mapa. | \(\{p_i\}_{i=1}^{\lvert\mathcal P\rvert}\). | Definido. |
| \(\lvert\mathcal P\rvert\) | Número de puntos del conjunto completo. | Cardinalidad. | Definido por notación estándar. |
| \(p_i=(x_i,y_i)\) | Posición de un punto ocupado. | Coordenadas métricas en dos dimensiones. | Definido. |
| \(x_i,y_i\) | Coordenadas de los puntos usados en la comparación ambiental. | Metros. | Definidas; aclarar el criterio de centrado si se refina el método. |
| \(\delta_{p_i}\) | Medida de Dirac localizada en \(p_i\). | Masa puntual. | Definida; el texto usa el término “function”. |
| \(\mu\) | Medida empírica uniforme de todos los puntos ocupados. | \(\mu=\lvert\mathcal P\rvert^{-1}\sum_i\delta_{p_i}\). | Definida por ecuación. |
| \(n_p\) | Número de puntos seleccionados por ambiente en la formulación. | 4,000 principal; 2,000 en la comparación de densidad. | Actualización integrada; ya no se usa \(N\) para estos puntos. |
| \(\hat\mu\) | Medida empírica de los puntos seleccionados. | Distribución submuestreada. | Definida verbalmente. |
| \(\mathcal A,\mathcal B\) | Ambientes que se comparan. | Un par de ambientes. | Definidos. |
| \(\hat\mu_A,\hat\mu_B\) | Medidas empíricas correspondientes al par. | Dos distribuciones. | Definidas por contexto. |
| \(\mathbf D\) | Matriz de costos de transporte. | \(\mathbb R^{n_p\times n_p}\), según la formulación actual. | Actualización integrada; \(C\) queda reservado a la arquitectura. |
| \(D_{ij}\) | Costo entre el punto \(i\) de un ambiente y el punto \(j\) del otro. | El texto declara \(\|p_i-p_j\|_2\). | Actualización integrada; precisar su escala junto con la normalización. |
| \(\boldsymbol\Gamma\) | Plan de transporte entre las distribuciones. | Matriz de masas transportadas. | Actualización integrada. |
| \(\Gamma_{ij}\) | Masa transportada entre los puntos \(i,j\). | Entrada del plan. | Actualización integrada. |
| \(\Pi(\hat\mu_A,\hat\mu_B)\) | Conjunto de planes admisibles con esos marginales. | Conjunto de acoplamientos. | Definido. |
| \(H(\boldsymbol\Gamma)\) | Entropía del plan de transporte. | \(-\sum_{i,j}\Gamma_{ij}\log\Gamma_{ij}\). | Definida con la notación nueva. |
| \(\lambda\) | Regularización entrópica. | 0.001. | Definida. |
| \(d_{\mathrm{Sinkhorn}}(\mathcal A,\mathcal B)\) | Cantidad de disimilitud definida en la ecuación de transporte regularizado. | Escalar. | Definida en el texto; pendiente verificar la concordancia operacional de la fórmula con el escalar reportado. |
| \(s_{\mathrm{global}}\) | Escala utilizada para normalizar los costos. | Distancia de referencia. | Definida; la descripción de su estimación se conserva como revisión metodológica pendiente. |
| \(B\) | Número de repeticiones del remuestreo. | \(B=5\). | Definido; no es el símbolo de batch size. |
| \(\hat d(\mathcal A,\mathcal B)\) | Promedio de la cantidad calculada para el par entre las \(B\) repeticiones. | Escalar por par. | Definido verbalmente. |
| \(\sigma\) | Desviación estándar entre las repeticiones. | Dispersión por par. | Definida; no equivale por sí sola a un intervalo de confianza. |
| CV | Coeficiente de variación usado para resumir estabilidad. | Porcentaje. | Nombre definido; falta explicitar fórmula y agregación si se busca una formulación completa. |

Los índices \(i,j\) de los puntos tienen un uso local en esta sección. No es obligatorio renombrarlos por compartir letras con índices de otras secciones, siempre que su ámbito sea claro. La letra caligráfica \(\mathcal B\), que representa un ambiente, se distingue del entero \(B\), que cuenta repeticiones.

**PF y FSM.** La versión actual contiene nuevos nombres que separan la información del objetivo, las lecturas láser y las velocidades de control.

| Variable actual | Significado | Valor o precisión pendiente |
|---|---|---|
| \(r_i\) | Distancia medida por el rayo láser \(i\). | Sustituye al anterior \(d_i\); se distingue de \(d\), distancia al objetivo. |
| \(\phi_i\) | Ángulo del rayo láser \(i\). | Sustituye al anterior \(\theta_i\); conviene definir explícitamente su marco y unidad. |
| \(d_{\mathrm{inf}}\) | Distancia de influencia para la repulsión. | 0.5 m; sustituye a \(d_0\). |
| \(\theta_{\mathrm{err}}\) | Error angular entre la orientación del robot y la dirección deseada. | Sustituye a `error`; precisar signo y rango al refinar el controlador. |
| \(\mathbf F_{\mathrm{attr}}\) | Componente atractiva hacia el objetivo. | Definida. |
| \(\mathbf F_{\mathrm{rep}}\) | Componente repulsiva debida a obstáculos. | Sustituye a \(\mathbf F_{\mathrm{rej}}\). |
| \(\mathbf F\) | Combinación de las componentes. | \(\mathbf F_{\mathrm{attr}}+\mathbf F_{\mathrm{rep}}\). |
| \(\mathbf q\) | Vector unitario hacia el objetivo. | Definido; revisar la convención completa de signos en el cálculo del movimiento. |
| \(f\) | Magnitud de la contribución repulsiva de una lectura. | \(f=\zeta\sqrt{1/r_i-1/d_{\mathrm{inf}}}\). |
| \(\eta\) | Ganancia atractiva. | 1.5. |
| \(\zeta\) | Ganancia repulsiva. | 8.5. |
| \(\gamma_{\mathrm{occ}}\) | Ganancia de la contribución adicional de ocupación. | 0.4; sustituye a la anterior \(\gamma\) de PF. |
| \(v_{\mathrm{lin}}\) | Velocidad lineal escalar de PF. | Sustituye a \(v\); distinguirla de la salida vectorial \((v_x,v_y,\omega)\). |
| \(\omega\) | Velocidad angular de PF. | Sustituye a \(w\); coincide con el significado angular usado para las redes. |
| \(v_{\max}\) | Límite de velocidad lineal declarado para PF. | 0.4 m/s. |
| \(\omega_{\max}\) | Límite de velocidad angular declarado para PF. | 0.8 rad/s; sustituye a \(w_{\max}\). |
| \(\alpha\) | Parámetro de reducción de velocidad lineal según el error angular. | 0.8. |
| \(\beta\) | Parámetro de la respuesta angular. | 0.2. |
| \(s\), en FSM | Lado seleccionado para la evasión. | LEFT, RIGHT o BLOCKED según el algoritmo. |
| L, F, R | Regiones izquierda, frontal y derecha de percepción. | Definidas en la descripción/caption de FSM. |

La actualización de nombres de PF ya está presente. La concordancia de las ecuaciones, signos, umbrales y lógica de escape con el controlador evaluado sigue siendo una refinación metodológica separada.

**Pendientes de definición o precisión.** Son los puntos que quedan abiertos después de incorporar los cambios de notación.

1. **`Label`.** Distinguir la tupla de entradas y salida de la etiqueta de velocidad. Precisar la selección de la referencia del último fotograma y evitar describir una secuencia de salidas si el entrenamiento es many-to-one.
2. **Objetivo por fotograma y por muestra.** Mantener \(\mathbf{tgt}^{(i,c)}\in\mathbb R^2\) y declarar cómo se agrupan esos vectores en \(\mathbf{tgt}^{(i)}\).
3. **\(x,h\).** Activar una definición breve en la descripción de la celda RNN; los comentarios del fuente no forman parte del texto visible del artículo.
4. **Unidades y convenciones.** Unificar pixels/m para \(\mathrm{Res}\), declarar una vez las unidades y convenciones angulares, y separar velocidades físicas de etiquetas normalizadas.
5. **Similitud.** Precisar la cantidad reportada, la normalización por \(s_{\mathrm{global}}\), el procedimiento de submuestreo y la agregación de la dispersión. Los nombres \(n_p,\mathbf D,\boldsymbol\Gamma\) ya están corregidos y no requieren una nueva sustitución.
6. **PF.** Completar la definición de \(\phi_i\), el signo/rango de \(\theta_{\mathrm{err}}\) y las tolerancias de llegada y atasco. El cambio de nombres no reemplaza esa validación del método.

**Notación adicional propuesta para estudiar generalización.** Estos símbolos no se presentan como ya incorporados al manuscrito.

| Propuesta | Uso |
|---|---|
| \(\mathcal E_a,\mathcal E_b\) | Ambientes indexados. |
| \(\hat d_{ab}\) | Abreviatura de \(\hat d(\mathcal E_a,\mathcal E_b)\). |
| \(\sigma_{ab}\) | Dispersión asociada al mismo par. |
| \(\boldsymbol\Delta\) | Matriz de disimilitudes, con \(\Delta_{ab}=\hat d_{ab}\). |
| \(K\) | Número de ambientes, si se necesita una expresión como \(K(K-1)/2\). |
| \(\mathcal T,\mathcal V,\mathcal U\) | Conjuntos de ambientes de entrenamiento, validación y prueba. |
| \(\delta_a\) | Cercanía de un ambiente de prueba al conjunto de entrenamiento, según una regla predefinida. |
| \(\mathrm{SR}_a\) | Tasa de éxito de navegación en el ambiente \(a\). |
| \(\rho_S\) | Correlación de rangos de Spearman, si se incorpora una formulación estadística. |

Por ejemplo, una posible regla para la cercanía al conjunto de entrenamiento es

\[
\delta_a=\min_{\mathcal E_b\in\mathcal T}\hat d_{ab},
\qquad \mathcal E_a\in\mathcal U.
\]

Esta es una propuesta para los experimentos posteriores. No establece que la distancia ya prediga el desempeño ni que \(SC\) sea una medida de dificultad de navegación.

**Estado actualizado.** Las colisiones de notación entre LOM y similitud ya están resueltas: \(N\) se usa para la LOM y \(n_p\) para puntos; \(C\) para canales y \(\mathbf D\) para costos; \(\boldsymbol\Gamma\) para transporte y \(\gamma_{\mathrm{occ}}\) para la contribución de ocupación en PF. La revisión pendiente se concentra en `rep_tgt`, `Label`, las definiciones activas de la celda RNN y las precisiones metodológicas indicadas.

