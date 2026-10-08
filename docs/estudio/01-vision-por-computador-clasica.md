# 01 — Visión por computador clásica

## Por qué clásica y no una red neuronal

El encargo es **offline y determinista**: sin red, sin modelos preentrenados, sin
descargas de datos. Eso descarta usar una CNN preentrenada (necesita red y pesos
opacos). Y entrenar una CNN desde cero con imágenes sintéticas solo le enseñaría
las manías del generador, no algo transferible, además de añadir una dependencia
pesada y no determinista. La opción honesta es la **visión por computador clásica**:
cada característica es un estadístico que un técnico puede nombrar y entender.

## Las cuatro clases

| Clase       | Firma visual                                             |
| ----------- | -------------------------------------------------------- |
| `healthy`   | rejilla de celdas uniforme, sin anomalías localizadas    |
| `hotspot`   | una mancha brillante localizada (celda caliente)         |
| `crack`     | una línea oscura fina y alargada (microfractura)         |
| `dead_cell` | un bloque oscuro compacto del tamaño de una celda         |

## Las características (features)

Se agrupan para separar cada firma. Todas se calculan con NumPy y
`scipy.ndimage`:

- **Estadísticos globales de brillo** (media, desviación, colas clara y oscura):
  un resumen grueso del panel.
- **Detección de manchas brillantes** (umbral + componentes conexas con
  `scipy.ndimage.label`): un *hotspot* es una región compacta mucho más brillante
  que su entorno. Medimos su área y su **contraste** (cuántas desviaciones robustas
  sobresale el pico).
- **Densidad de bordes / líneas** (gradiente de **Sobel**): una *grieta* es una
  línea fina de gradiente alto. Como es de orientación aleatoria, aporta energía de
  borde **diagonal** que la rejilla de celdas (siempre vertical/horizontal) no
  tiene.
- **Forma de la región oscura** (umbral + componentes conexas + elongación +
  relleno de su caja): una *dead cell* es un bloque grande, compacto y bien
  relleno; la región oscura de una *grieta* es larga, fina y apenas rellena su caja.

## Umbrales robustos (mediana + MAD)

Para decidir qué píxel es "demasiado brillante" o "demasiado oscuro" no usamos la
media y la desviación estándar, porque **la propia anomalía las infla** y se
enmascara. Usamos la **mediana** y la **MAD** (desviación absoluta mediana),
escaladas por `1,4826` para que, con datos normales, el umbral robusto coincida
con el ordinario. Así, una mancha fuerte **sobresale** en lugar de esconderse, y su
contraste ("6,3 desviaciones robustas sobre el fondo") es una cantidad explicable
que la regla del hotspot puede umbralizar directamente.
