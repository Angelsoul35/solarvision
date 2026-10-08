# Guía de estudio — SolarVision

Una guía en español para entender el proyecto a fondo y poder defenderlo en una
entrevista. El código, los comentarios y la documentación técnica están en inglés
(como en un entorno profesional real); esta guía explica el *porqué* de cada
decisión.

## Orden sugerido

1. [01 — Visión por computador clásica](01-vision-por-computador-clasica.md)
   Qué mide cada característica (feature) y por qué no usamos una red neuronal.
2. [02 — Imágenes sintéticas y el split honesto](02-imagenes-sinteticas-y-split.md)
   Cómo se generan los paneles y cómo se separan train/test sin hacer trampa.
3. [03 — La capa de reglas y el modelo](03-capa-de-reglas-y-modelo.md)
   Por qué una regla determinista decide los casos claros y el bosque el resto.
4. [04 — Evaluación honesta](04-evaluacion-honesta.md)
   Por qué el número honesto es ~0,88 y no un sospechoso 1,0.
5. [05 — Preguntas de entrevista](05-preguntas-de-entrevista.md)
   Preguntas típicas y cómo responderlas con este proyecto.

## La idea en una frase

Un detector de defectos en paneles solares **offline y determinista** que usa
**visión por computador clásica** (sin modelos preentrenados ni descargas), decide
los casos inequívocos con una **capa de reglas explicable** y deja el resto a un
clasificador pequeño — y que se evalúa con **configuraciones de defecto no vistas**,
de modo que la exactitud publicada (~0,88) es honesta y no un 1,0 memorizado.
