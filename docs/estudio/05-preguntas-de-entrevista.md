# 05 — Preguntas de entrevista

Preguntas típicas sobre este proyecto y cómo responderlas.

### "¿Por qué visión clásica y no deep learning?"
Porque el encargo es **offline y determinista**: sin red, sin pesos preentrenados,
sin descargas. Una CNN preentrenada queda descartada, y entrenar una desde cero con
datos sintéticos solo aprendería las manías del generador. La visión clásica da
características que un técnico puede **nombrar e inspeccionar**, que es justo lo que
permite la capa de reglas y las explicaciones. Documento el techo más bajo como un
límite de alcance, no lo escondo.

### "Tu exactitud, ¿no está inflada por ser datos sintéticos?"
Ese es exactamente el riesgo, y por eso **no** uso un split aleatorio. Reservo
*bins* enteros de severidad y tamaño para test, así que cada defecto de test es una
configuración **no vista**. El número honesto es **0,88 ± 0,03** sobre 5 seeds, con
una brecha train-test acotada (~0,10). Si hubiera memorizado, la brecha sería grande
y la exactitud, cercana a 1,0.

### "¿Por qué no sale 1,0 si las imágenes son sintéticas?"
Porque el extremo **tenue** de la severidad se solapa de verdad con la clase sana
bajo ruido. Un defecto apenas por encima del ruido es genuinamente ambiguo, y ahí
cae el error. Un 1,0 sería la señal de que el test era trivial.

### "¿Cómo distingues una grieta de las líneas de la rejilla?"
Por **forma y orientación**. Las líneas de la rejilla son axis-aligned y periódicas;
una grieta es una componente oscura aislada, **alargada** (elongación alta) y de
**relleno de caja bajo**, y al tener orientación aleatoria aporta energía de borde
**diagonal** que la rejilla no tiene. La celda muerta, en cambio, es un bloque
compacto (elongación baja, relleno alto).

### "¿Qué hace la capa de reglas y por qué va antes del modelo?"
Decide los casos **inequívocos** con un umbral explicable (mancha muy brillante =
hotspot; raya fina y larga = grieta; bloque oscuro grande = celda muerta). Ningún
modelo estadístico debería rebajar con menos confianza algo tan claro. Lo ambiguo
cae al bosque aleatorio. Las reglas exigen un área mínima, así que no inventan
defectos a partir del ruido.

### "¿Qué pasa si el modelo se equivoca?"
Si la confianza queda por debajo del umbral, **no se afirma**: se marca para
revisión manual. Y los umbrales de las reglas y de confianza son diales de operador:
se ajustan según cuánto error se tolere frente a cuánta revisión.

### "¿Qué harías para llevarlo a producción?"
Entrenar y validar con imágenes **reales** etiquetadas (electroluminiscencia o
térmicas), casi seguro con una **CNN**, y medir contra este score de configuraciones
no vistas, no contra el ~1,0 de dentro de distribución. Añadiría más tipos de
defecto, calibración de la cámara y detección multi-defecto por imagen.
