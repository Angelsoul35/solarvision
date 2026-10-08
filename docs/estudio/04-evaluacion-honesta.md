# 04 — Evaluación honesta

## Qué medimos y cómo

Entrenamos el modelo con un conjunto de *bins* de severidad/tamaño y lo evaluamos
con un conjunto **disjunto** (ver [02](02-imagenes-sinteticas-y-split.md)), así que
cada defecto de test es una configuración no vista. Evaluamos el **detector
completo** (reglas + modelo) de extremo a extremo, porque es lo que corre en
producción.

El informe da tres números, no uno:

- **accuracy**: aciertos sobre el total.
- **macro-F1**: F1 promediado por clase con el mismo peso, para que una clase floja
  no se esconda tras las fáciles.
- **brecha train-test** (*gap*): si el modelo memorizara, la brecha sería grande.

```
$ python -m solarvision.evaluation
  accuracy (mean+/-std over 5 seeds): 0.88 +/- 0.03   # 4 clases; azar ~0,25
  macro-F1 (mean over 5 seeds)      : 0.89
  train accuracy                      : 0.98
  train-test gap                      : 0.10   # acotada -> aprende la señal, no las configuraciones
```

## Por qué promediamos varios seeds

El split depende de qué bins caen en test, y eso varía con la semilla: un seed puede
dar 0,91 (con suerte) y otro 0,83 (sin ella). Reportar uno solo sería el mismo
autoengaño que el split aleatorio. Por eso el comando **promedia 5 seeds** y muestra
la desviación.

## Lo que el número sí dice y lo que no dice

- **Sí** dice que el detector **generaliza** a contrastes y tamaños no vistos:
  interpola entre los bins reservados porque las características varían de forma
  suave con la severidad, y la brecha se mantiene pequeña (~0,10) — la firma de
  aprender la señal, no las configuraciones concretas.
- **No** dice que sea perfecto. El extremo **tenue** de la severidad se solapa de
  verdad con la clase sana bajo ruido, y ahí cae ese ~12% de error (sobre todo
  defecto tenue contra sano, y grieta tenue contra celda muerta). Cerrar esa brecha
  **no** es cuestión de más datos sintéticos.

## El alcance, dicho con claridad

Es un *benchmark* **sintético**, no software de inspección certificado. Un sistema
real se entrenaría y validaría con imágenes reales **de electroluminiscencia o
térmicas** etiquetadas, y casi con seguridad usaría una red convolucional. El
baseline honesto a batir ahí es este número con configuraciones no vistas, no el
~1,0 de dentro de distribución.
