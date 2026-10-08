# 03 — La capa de reglas y el modelo

## Dos etapas, una decisión explicable

El detector no es una caja negra. Primero pasa una **capa de reglas** deterministas;
si ninguna se activa, decide un **bosque aleatorio** (`RandomForestClassifier`). En
ambos casos el informe dice *por qué*.

## La capa de reglas: los casos inequívocos

Algunos defectos son visualmente inequívocos, y para ellos un único umbral escrito a
mano es correcto y se explica solo. Cada regla es una función pura del vector de
características y se activa **solo con evidencia fuerte**:

- **hotspot**: contraste de la mancha brillante por encima de `SV_HOTSPOT_CONTRAST_Z`
  (6,0 desviaciones robustas) y con área mínima.
- **dead_cell**: área oscura por encima de `SV_DEAD_CELL_DARK_FRAC` (2,0% del panel),
  compacta (relleno de caja alto) y poco alargada.
- **crack**: región oscura con elongación por encima de `SV_CRACK_MIN_ELONGATION`
  (4,0), fina (relleno de caja bajo) y con área mínima.

Las reglas **nunca inventan** un defecto a partir del ruido: todas exigen un área
mínima afectada, así que un par de píxeles ruidosos no las dispara. Y son **diales
de operador**: subir los umbrales manda más casos al modelo; bajarlos confía más en
las reglas.

## El modelo: el territorio ambiguo

Cuando ninguna regla se activa —el caso común, el punto medio difícil— decide el
bosque aleatorio sobre las 14 características. Es pequeño y está **sembrado**
(`random_state=0`), así que es determinista. Su probabilidad por clase es la
**confianza**; si queda por debajo de `SV_MIN_CONFIDENCE` (0,55), la evaluación se
marca como **revisión manual** en vez de afirmarse.

## El guardián de novedad (out-of-distribution)

Un clasificador entrenado con paneles devolverá **igualmente** alguna clase, y con
confianza, ante una imagen que **no es un panel** (un marco en blanco, una foto de
otra cosa, un artefacto no visto). Para frenar esa respuesta segura pero sin
sentido, un **isolation forest** entrenado con las mismas características actúa de
guardián: si una imagen cae **fuera del colector** (manifold) de entrenamiento, se
marca *out-of-distribution* y va a revisión, sin importar la confianza del bosque.
Por eso una imagen aleatoria no se etiqueta como "sana" con aplomo.

## La explicación (por imagen, no global)

- Si se activa una regla, el informe da la regla y su razón legible ("mancha a 7,0
  desviaciones robustas sobre el fondo…").
- Si decide el modelo, el informe da la **evidencia de esta imagen**: no el ranking
  global del bosque (que sería idéntico para todas), sino las características de
  *este* panel que más se desvían de uno típico, como z-scores con signo. Así un
  hotspot destaca su `bright_blob_contrast` alto y un crack su `dark_blob_elongation`
  alta — las medidas que de verdad movieron *esta* decisión.

Así, incluso en el camino del modelo, la decisión es inspeccionable **y específica
de la imagen**.
