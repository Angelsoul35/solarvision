# ADR 0001 — Classical computer vision, fully offline and deterministic

## Status
Accepted.

## Context
The project must run with no network access, no pretrained models and no dataset
downloads, and it must be reproducible bit-for-bit. Two tempting shortcuts are off
the table: downloading a pretrained CNN (needs network and opaque weights), and
training a CNN from scratch on the synthetic images (it would learn the renderer's
quirks rather than anything transferable, while adding a heavy, non-deterministic
dependency).

## Decision
Represent each grayscale panel with a small vector of **classical** features —
global brightness statistics, bright-blob detection via thresholding and connected
components, Sobel edge/line density, and dark-region shape — computed with NumPy
and `scipy.ndimage`. A small, seeded `RandomForestClassifier` maps that vector to a
class. Everything is deterministic given a seed.

## Consequences
- Fully offline; the only inputs are the renderer's seeds. No downloads, ever.
- Every feature is nameable and inspectable, which makes the rule layer and the
  explanations possible.
- A CNN could reach somewhat higher accuracy on real data, at the cost of the
  per-feature explanation this pipeline gives an inspector. On the real ELPV
  benchmark the classical features already reach ~0.88 accuracy / ~0.89–0.94 AUC
  (functional vs defective), so the interpretability trade-off costs little here.
