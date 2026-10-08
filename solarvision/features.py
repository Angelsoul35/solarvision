"""Classical computer-vision features: a grayscale panel -> a fixed vector.

No learned representation and no pretrained network is involved here. Every
feature is a hand-defined statistic a technician could reason about, computed
with NumPy and :mod:`scipy.ndimage`. They are grouped to separate the three
defect signatures from a healthy panel:

* **Global brightness stats** (mean/std/dark-tail/bright-tail) — a coarse summary
  of the whole panel.
* **Bright-blob detection** (threshold + connected components) — a *hotspot* is a
  compact region far brighter than its surroundings.
* **Edge / dark-line density** (Sobel gradient + diagonal share) — a *crack* is a
  thin high-gradient streak that raises the edge density above a healthy panel,
  and being randomly oriented it is not part of the periodic axis-aligned grid.
* **Dark-region shape** (threshold + connected components + elongation/fill) — a
  *dead cell* is a large, blocky, well-filled dark region; a *crack*'s dark region
  is long, thin and sparsely filled.

Thresholds are **robust** (median + MAD, not mean + standard deviation) so that
the very anomaly we want to measure does not inflate the threshold and hide
itself — the same reasoning used throughout the project.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass, fields

import numpy as np
from scipy import ndimage

# MAD -> sigma for normally distributed data, so a robust threshold matches the
# ordinary mean+std threshold when there is no anomaly.
_MAD_TO_SIGMA = 1.4826
_EPS = 1e-9

# Threshold multipliers (in robust sigmas) for the bright/dark/edge masks.
_BRIGHT_K = 3.0
_DARK_K = 2.6
_EDGE_K = 2.0
_MIN_BLOB_PX = 4  # connected components smaller than this are ignored as noise


@dataclass(frozen=True)
class Features:
    """One panel's feature vector. Field order is the vector order."""

    mean_intensity: float
    std_intensity: float
    bright_tail: float  # (p99 - median) in robust sigmas
    dark_tail: float  # (median - p1) in robust sigmas
    bright_blob_area: float  # largest bright component, as a fraction of the image
    bright_blob_contrast: float  # peak of that blob above background, in robust sigmas
    n_bright_blobs: float  # count of bright components above the min size
    edge_density: float  # fraction of pixels with a strong Sobel gradient
    mean_gradient: float  # mean Sobel magnitude
    diag_edge_ratio: float  # share of edge energy that is diagonal (vs axis-aligned)
    dark_area_frac: float  # fraction of pixels below the dark threshold
    dark_blob_area: float  # largest dark component, as a fraction of the image
    dark_blob_elongation: float  # major/minor axis ratio of that component
    dark_blob_fill: float  # component area / its bounding-box area (blockiness)

    def vector(self) -> np.ndarray:
        return np.array(astuple(self), dtype=np.float64)

    def as_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


FEATURE_NAMES: list[str] = [f.name for f in fields(Features)]


def _robust_stats(x: np.ndarray) -> tuple[float, float]:
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    sigma = _MAD_TO_SIGMA * mad
    return median, sigma


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Return a boolean mask of the largest connected component and its pixel area."""
    labelled, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool), 0
    counts = np.bincount(labelled.ravel())
    counts[0] = 0  # background
    biggest = int(np.argmax(counts))
    return labelled == biggest, int(counts[biggest])


# Per-axis variance floor (~one pixel of extent) so a 1px-wide line does not divide
# by ~0, and the maximum elongation we report, so values stay bounded and sane.
_VAR_FLOOR = 0.5
_MAX_ELONGATION = 50.0


def _elongation_and_fill(component: np.ndarray, area: int) -> tuple[float, float]:
    """Elongation (major/minor axis) and bounding-box fill of a component.

    Elongation uses the eigenvalues of the pixel-coordinate covariance: a line has
    one large and one tiny eigenvalue (high elongation); a square has two similar
    ones (elongation ~1). A per-axis variance floor keeps a 1px-wide streak from
    producing a divide-by-zero blow-up, and the result is capped. Fill is
    area / bounding-box area: a block fills its box, a thin streak barely does.

    A component below the minimum blob size is treated as absent (no meaningful
    shape): neutral elongation 1.0 and zero fill, so it triggers no shape rule.
    """
    if area < _MIN_BLOB_PX:
        return 1.0, 0.0
    ys, xs = np.nonzero(component)
    coords = np.stack([ys.astype(np.float64), xs.astype(np.float64)])
    cov = np.cov(coords)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    major = float(eigvals[-1]) + _VAR_FLOOR
    minor = float(eigvals[0]) + _VAR_FLOOR
    elongation = min(_MAX_ELONGATION, float(np.sqrt(major / minor)))
    bbox = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    fill = area / float(bbox)
    return elongation, fill


def extract(img: np.ndarray) -> Features:
    """Compute the classical feature vector for a grayscale panel in ``[0, 1]``."""
    img = np.asarray(img, dtype=np.float64)
    n_px = img.size
    median, sigma = _robust_stats(img)
    sigma = max(sigma, _EPS)

    p1, p99 = np.percentile(img, [1, 99])
    bright_tail = float((p99 - median) / sigma)
    dark_tail = float((median - p1) / sigma)

    # --- Bright blobs (hotspot signature) ---
    bright_mask = img > median + _BRIGHT_K * sigma
    labelled, n = ndimage.label(bright_mask)
    n_bright = 0
    bright_area = 0
    bright_component = np.zeros_like(bright_mask)
    if n:
        counts = np.bincount(labelled.ravel())
        counts[0] = 0
        n_bright = int((counts >= _MIN_BLOB_PX).sum())
        biggest = int(np.argmax(counts))
        bright_component = labelled == biggest
        bright_area = int(counts[biggest])
    bright_contrast = 0.0
    if bright_area:
        peak = float(img[bright_component].max())
        bright_contrast = (peak - median) / sigma

    # --- Edges / dark lines (crack signature) ---
    gx = ndimage.sobel(img, axis=1)
    gy = ndimage.sobel(img, axis=0)
    mag = np.hypot(gx, gy)
    mg_median, mg_sigma = _robust_stats(mag)
    edge_mask = mag > mg_median + _EDGE_K * max(mg_sigma, _EPS)
    edge_density = float(edge_mask.mean())
    mean_gradient = float(mag.mean())
    if edge_mask.any():
        ax = np.abs(gx[edge_mask])
        ay = np.abs(gy[edge_mask])
        diag_edge_ratio = float(np.mean(np.minimum(ax, ay) / (np.maximum(ax, ay) + _EPS)))
    else:
        diag_edge_ratio = 0.0

    # --- Dark regions (dead-cell vs crack shape) ---
    dark_mask = img < median - _DARK_K * sigma
    dark_area_frac = float(dark_mask.mean())
    dark_component, dark_area = _largest_component(dark_mask)
    elongation, fill = _elongation_and_fill(dark_component, dark_area)

    return Features(
        mean_intensity=float(img.mean()),
        std_intensity=float(img.std()),
        bright_tail=bright_tail,
        dark_tail=dark_tail,
        bright_blob_area=bright_area / n_px,
        bright_blob_contrast=float(bright_contrast),
        n_bright_blobs=float(n_bright),
        edge_density=edge_density,
        mean_gradient=mean_gradient,
        diag_edge_ratio=diag_edge_ratio,
        dark_area_frac=dark_area_frac,
        dark_blob_area=dark_area / n_px,
        dark_blob_elongation=elongation,
        dark_blob_fill=fill,
    )


def extract_matrix(images: list[np.ndarray]) -> np.ndarray:
    """Stack :func:`extract` over many images into an ``(n_images, n_features)`` matrix."""
    return np.vstack([extract(im).vector() for im in images])
