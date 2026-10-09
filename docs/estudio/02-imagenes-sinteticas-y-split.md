# 02 — Imágenes sintéticas y el split riguroso

## Cómo se genera un panel

Todo se dibuja desde cero con NumPy (no hay ninguna foto real). Un panel sano es:

1. un nivel de gris base con un **gradiente de iluminación** suave y aleatorio,
2. una **rejilla de celdas** con líneas finas más oscuras (los *busbars*),
3. una **variación de brillo por celda** (cada celda un poco más clara u oscura), y
4. **ruido** gaussiano de sensor.

Sobre esa base se estampa (o no) un defecto: una mancha gaussiana brillante
(*hotspot*), una línea oscura fina de orientación aleatoria (*crack*) o un bloque
oscuro del tamaño de una celda (*dead cell*). El resultado se guarda como PNG en
escala de grises mediante Pillow.

## La clave: severidad y tamaño

Cada defecto tiene dos escalares en `[0, 1]`: **severidad** (su contraste) y
**tamaño**. El extremo bajo de la severidad está a propósito **cerca del nivel de
ruido**, así que hay defectos desde "casi invisibles" hasta "evidentes". Esto es lo
que hace el problema realista: un defecto tenue se confunde de verdad con un panel
sano.

## La trampa del split aleatorio (y cómo la evitamos)

Si entrenas y evalúas con un **split aleatorio** de imágenes sintéticas, el conjunto
de test reutiliza casi las mismas configuraciones de defecto que ya viste en
entrenamiento: el modelo solo tiene que **reconocer** contrastes y tamaños que ya
había visto, y la exactitud sale cerca de 1,0 sin significar nada. Es el equivalente
en visión a memorizar plantillas.

### La corrección: split disjunto por *bins*

Discretizamos la severidad y el tamaño en **bins** (intervalos). `make_split_datasets`
reserva **bins enteros** para test: los valores de severidad/tamaño detrás de las
imágenes de test son **disjuntos** de los de entrenamiento. Así, **cada defecto de
test es una configuración nunca vista**.

Dos detalles que mantienen el rigor:

- Los bins reservados están **entrelazados** por todo el rango (vía un *shuffle* con
  semilla), no son "los fuertes" ni "los tenues". Train y test afrontan la **misma
  dificultad**; no es un split fácil ni difícil a propósito.
- La **posición** se muestrea libremente en ambos, porque las características son
  invariantes a la posición por diseño: no hay nada que reservar ahí.

El resultado: el número refleja la **generalización a configuraciones no vistas**, y
como el extremo tenue se solapa con "sano" bajo ruido, cae rigurosamente por debajo
de 1,0.
