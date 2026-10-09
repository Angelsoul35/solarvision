"""Synthetic, labelled solar-panel images for the defect detector.

Everything here is rendered from scratch with NumPy: there is no dataset download
and no real photograph is ever used. A panel is a grayscale image of a regular
cell grid with a gentle illumination gradient, per-cell brightness jitter and
sensor noise. On top of that healthy base we stamp one of three defects, each
with the visual signature of its real electroluminescence/thermal counterpart:

* **hotspot** — a bright Gaussian blob (a shaded/shunted cell runs hot);
* **crack**   — a thin, dark, randomly oriented streak (a wafer micro-crack);
* **dead_cell** — a compact dark block the size of a cell (an inactive cell).

Design note (this matters for rigour). Each defect has a scalar **severity** in
``[0, 1]`` that drives its contrast and size, from barely-above-noise to obvious.
A naive random train/test split would let the test set reuse the exact severities
seen in training and report a meaningless near-1.0 — the computer-vision analogue
of template memorisation. :func:`make_split_datasets` instead holds out whole
**severity/size bins**: the parameter values behind the test images are disjoint
from those behind the training images, so every test defect is an *unseen*
configuration. The faint end of the severity range genuinely overlaps the healthy
class under noise, which is exactly why the leakage-free held-out accuracy sits below
1.0 rather than at a suspicious 1.0.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from .labels import ALL_CLASSES, DefectClass

# ----------------------------------------------------------------------------
# Panel geometry and base appearance. Small and fixed so rendering is fast and
# fully deterministic given a seed.
# ----------------------------------------------------------------------------
IMG_SIZE = 64  # pixels, square
CELL_SIZE = 10  # cell pitch in pixels; busbar/grid lines sit on this lattice

_BASE_LEVEL = 0.46  # mid-gray base reflectance
_GRADIENT_AMP = 0.05  # smooth illumination gradient across the panel
_GRID_CONTRAST = 0.05  # how much darker the thin cell-grid lines are
_CELL_JITTER = 0.030  # per-cell brightness variation (healthy, natural)
_NOISE_STD = 0.035  # additive Gaussian sensor noise

# Severity -> physical contrast, per defect. The low end is deliberately close to
# the noise floor so faint defects are genuinely hard; the high end is obvious.
_HOTSPOT_AMP = (0.06, 0.40)  # brightness added at the blob peak
_HOTSPOT_SIGMA = (2.5, 6.0)  # Gaussian radius in pixels (scaled by size_frac)
_CRACK_DARK = (0.06, 0.38)  # brightness removed along the streak
_CRACK_LEN = (10.0, 30.0)  # streak length in pixels (scaled by size_frac)
_DEAD_DARK = (0.07, 0.42)  # brightness removed inside the block
_DEAD_SIZE = (7.0, 13.0)  # block side in pixels (scaled by size_frac)


def _lerp(bounds: tuple[float, float], t: float) -> float:
    lo, hi = bounds
    return lo + (hi - lo) * float(t)


def _base_panel(rng: np.random.Generator) -> np.ndarray:
    """A healthy panel: lattice grid + illumination gradient + jitter + noise."""
    img = np.full((IMG_SIZE, IMG_SIZE), _BASE_LEVEL, dtype=np.float64)

    # Smooth illumination gradient with a random, gentle direction.
    ys, xs = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE] / (IMG_SIZE - 1)
    ang = rng.uniform(0.0, 2.0 * np.pi)
    ramp = np.cos(ang) * (xs - 0.5) + np.sin(ang) * (ys - 0.5)
    img += _GRADIENT_AMP * ramp

    # Per-cell brightness jitter: each cell is a little lighter or darker.
    n_cells = IMG_SIZE // CELL_SIZE + 1
    cell_offsets = rng.normal(0.0, _CELL_JITTER, size=(n_cells, n_cells))
    cell_map = np.repeat(np.repeat(cell_offsets, CELL_SIZE, axis=0), CELL_SIZE, axis=1)
    img += cell_map[:IMG_SIZE, :IMG_SIZE]

    # Thin, axis-aligned cell-grid lines (busbars / cell boundaries).
    line = (np.arange(IMG_SIZE) % CELL_SIZE) == 0
    img[line, :] -= _GRID_CONTRAST
    img[:, line] -= _GRID_CONTRAST

    img += rng.normal(0.0, _NOISE_STD, size=img.shape)
    return np.clip(img, 0.0, 1.0)


def _add_hotspot(
    img: np.ndarray,
    rng: np.random.Generator,
    severity: float,
    size_frac: float,
    pos: tuple[float, float],
) -> None:
    amp = _lerp(_HOTSPOT_AMP, severity)
    sigma = _lerp(_HOTSPOT_SIGMA, size_frac)
    cy, cx = pos[0] * (IMG_SIZE - 1), pos[1] * (IMG_SIZE - 1)
    ys, xs = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    blob = amp * np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2)))
    np.clip(img + blob, 0.0, 1.0, out=img)


def _add_crack(
    img: np.ndarray,
    rng: np.random.Generator,
    severity: float,
    size_frac: float,
    pos: tuple[float, float],
) -> None:
    dark = _lerp(_CRACK_DARK, severity)
    length = _lerp(_CRACK_LEN, size_frac)
    angle = rng.uniform(0.0, np.pi)
    cy, cx = pos[0] * (IMG_SIZE - 1), pos[1] * (IMG_SIZE - 1)
    dx, dy = np.cos(angle), np.sin(angle)
    steps = int(length * 2) + 1
    ts = np.linspace(-length / 2.0, length / 2.0, steps)
    for t in ts:
        x = int(round(cx + t * dx))
        y = int(round(cy + t * dy))
        if 0 <= x < IMG_SIZE and 0 <= y < IMG_SIZE:
            img[y, x] -= dark
            # A micro-crack is ~1px wide; soften the immediate neighbours slightly.
            if 0 <= x - 1 and x + 1 < IMG_SIZE:
                img[y, x - 1] -= dark * 0.35
                img[y, x + 1] -= dark * 0.35
    np.clip(img, 0.0, 1.0, out=img)


def _add_dead_cell(
    img: np.ndarray,
    rng: np.random.Generator,
    severity: float,
    size_frac: float,
    pos: tuple[float, float],
) -> None:
    dark = _lerp(_DEAD_DARK, severity)
    side = int(round(_lerp(_DEAD_SIZE, size_frac)))
    cy, cx = int(pos[0] * (IMG_SIZE - 1)), int(pos[1] * (IMG_SIZE - 1))
    y0 = max(0, min(IMG_SIZE - side, cy - side // 2))
    x0 = max(0, min(IMG_SIZE - side, cx - side // 2))
    img[y0 : y0 + side, x0 : x0 + side] -= dark
    np.clip(img, 0.0, 1.0, out=img)


_RENDERERS = {
    DefectClass.HOTSPOT: _add_hotspot,
    DefectClass.CRACK: _add_crack,
    DefectClass.DEAD_CELL: _add_dead_cell,
}


def render(
    defect: DefectClass | str,
    rng: np.random.Generator,
    severity: float = 0.6,
    size_frac: float = 0.5,
    pos: tuple[float, float] | None = None,
) -> np.ndarray:
    """Render one grayscale panel of the given class as a float array in ``[0, 1]``.

    ``severity`` and ``size_frac`` are in ``[0, 1]`` and map to the defect's
    physical contrast and size; ``pos`` is a ``(row_frac, col_frac)`` centre in
    ``[0, 1]`` (random if omitted). ``healthy`` ignores the defect parameters.
    """
    cls = DefectClass(defect)
    img = _base_panel(rng)
    if cls is DefectClass.HEALTHY:
        return img
    if pos is None:
        pos = (float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8)))
    _RENDERERS[cls](img, rng, severity, size_frac, pos)
    return img


# ----------------------------------------------------------------------------
# Datasets and the leakage-free, bin-disjoint train/test split.
# ----------------------------------------------------------------------------
_N_BINS = 20  # severity/size are discretised into this many bins for the split


def _sample_in_bins(rng: np.random.Generator, bins: np.ndarray) -> float:
    """Pick a bin from ``bins`` (indices into ``_N_BINS``) and a value inside it."""
    b = int(rng.choice(bins))
    lo, hi = b / _N_BINS, (b + 1) / _N_BINS
    return float(rng.uniform(lo, hi))


def _render_labelled(
    rng: np.random.Generator,
    n: int,
    sev_bins: np.ndarray,
    size_bins: np.ndarray,
) -> tuple[list[np.ndarray], list[str]]:
    images: list[np.ndarray] = []
    labels: list[str] = []
    for _ in range(n):
        cls = DefectClass(ALL_CLASSES[int(rng.integers(len(ALL_CLASSES)))])
        if cls is DefectClass.HEALTHY:
            images.append(render(cls, rng))
        else:
            sev = _sample_in_bins(rng, sev_bins)
            size = _sample_in_bins(rng, size_bins)
            images.append(render(cls, rng, severity=sev, size_frac=size))
        labels.append(cls.value)
    return images, labels


def make_dataset(n: int = 1200, seed: int = 7) -> tuple[list[np.ndarray], list[str]]:
    """A labelled dataset over the *full* severity/size range (used for deployment)."""
    rng = np.random.default_rng(seed)
    all_bins = np.arange(_N_BINS)
    return _render_labelled(rng, n, all_bins, all_bins)


def _split_bins(rng: np.random.Generator, holdout_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Partition ``_N_BINS`` bins into disjoint train/test pools (interleaved by shuffle)."""
    idx = rng.permutation(_N_BINS)
    cut = max(1, round(_N_BINS * (1.0 - holdout_frac)))
    return np.sort(idx[:cut]), np.sort(idx[cut:])


def make_split_datasets(
    n: int = 1200, seed: int = 7, holdout_frac: float = 0.35
) -> tuple[list[np.ndarray], list[str], list[np.ndarray], list[str]]:
    """Bin-disjoint train/test: whole severity and size bins are held out of
    training, so every test defect uses a contrast/size configuration never seen
    in training.

    Because the held-out bins are interleaved across the whole range (not just the
    strong or just the faint end), train and test face the same spread of
    difficulty — the number reflects generalisation to unseen configurations, not
    an easier or harder slice. Position is sampled freely in both (the features are
    position-invariant by design, so there is nothing to hold out there). Returns
    ``(train_images, train_labels, test_images, test_labels)``.
    """
    rng = np.random.default_rng(seed)
    sev_tr, sev_te = _split_bins(rng, holdout_frac)
    size_tr, size_te = _split_bins(rng, holdout_frac)
    tr_x, tr_y = _render_labelled(rng, n, sev_tr, size_tr)
    te_x, te_y = _render_labelled(rng, n // 3, sev_te, size_te)
    return tr_x, tr_y, te_x, te_y


# ----------------------------------------------------------------------------
# Image I/O helpers (PNG <-> float array <-> base64), shared by CLI/API/tests.
# ----------------------------------------------------------------------------
def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def to_png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(to_uint8(img), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def from_png_bytes(data: bytes) -> np.ndarray:
    """Decode a grayscale PNG to a float array in ``[0, 1]`` (any mode is coerced to L)."""
    image = Image.open(io.BytesIO(data)).convert("L")
    return np.asarray(image, dtype=np.float64) / 255.0


def encode_base64_png(img: np.ndarray) -> str:
    return base64.b64encode(to_png_bytes(img)).decode("ascii")


def decode_base64_png(data: str) -> np.ndarray:
    return from_png_bytes(base64.b64decode(data))
