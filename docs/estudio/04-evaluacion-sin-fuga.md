# 04 — Evaluación sin fuga

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

## Validación con imágenes reales (ELPV)

El sintético da una medición controlada de 4 clases; las **mismas** características se
miden también sobre **imágenes reales** de electroluminiscencia — el dataset ELPV
(funcional vs defectuoso), vendoreado en `datasets/elpv/`. Con
`python -m solarvision.real_benchmark`:

```
modelo                    CV accuracy       CV AUC   train   test   gap
Logística (regularizada)  0.878 ± 0.042     0.886    0.881   0.957  -0.076
Random forest             0.889 ± 0.035     0.940    1.000   0.957   0.043
```

Las características a mano **se trasladan directamente** a electroluminiscencia real:
~0,88 de accuracy y 0,89–0,94 de ROC-AUC sobre celdas reales, con un logístico bien
ajustado. Tiene sentido físico: un defecto real se ve como una región más **oscura** y
a menudo **agrietada**, justo lo que miden la cobertura oscura, la elongación y el
contraste — y cada veredicto sigue nombrando esas características. El random forest
sube el AUC a 0,94 pero memoriza el train (1,0): el trade-off de capacidad de siempre.

## El alcance, en positivo

Es un pipeline **interpretable y usable** validado en dos frentes: sintético (con
configuraciones de defecto no vistas, para medir generalización) y **real** (ELPV,
~0,88). Un despliegue de producción añadiría más tipos de defecto, imágenes térmicas y
graduación de severidad, y podría usar una CNN para algo más de accuracy — a cambio de
la explicación por característica que un inspector revisa. Aquí esa explicación se
conserva, y sobre dato real cuesta poco en accuracy.
