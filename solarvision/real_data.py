"""Load the real ELPV electroluminescence solar-cell images as feature vectors.

The synthetic generator (:mod:`solarvision.generate`) gives a controlled,
config-disjoint evaluation; this module loads real EL cell images (ELPV dataset,
vendored under ``datasets/elpv/``) and runs them through the *same* classical
feature extractor, so the real benchmark measures the same approach on real imagery.

The standard ELPV binary task is used: ``functional`` vs ``defective`` (the three
defective severity levels grouped into one class). See ``datasets/elpv/README.md``
for provenance; ``fetch_elpv`` refreshes or enlarges the vendored subset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import extract

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ELPV_DIR = _REPO_ROOT / "datasets" / "elpv"
_CLASSES = ("functional", "defective")  # index 0 = functional, 1 = defective (positive)
_HF_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=mjphayes%2Felpv-dataset"
    "&config=default&split=train&offset={offset}&length=100"
)
# HF ClassLabel order for mjphayes/elpv-dataset.
_HF_FUNCTIONAL_LABEL = 1  # others (0, 2, 3) are defective severity levels


def load_elpv(elpv_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return ``(X, y, class_names)`` for the vendored real EL images.

    ``X`` is the classical feature matrix (one row per image), ``y`` is 0 for
    functional and 1 for defective. Raises if the images are missing.
    """
    from PIL import Image

    root = Path(elpv_dir) if elpv_dir is not None else _ELPV_DIR
    classes = [c for c in _CLASSES if (root / c).is_dir()]
    if len(classes) != 2:
        raise FileNotFoundError(
            f"Expected functional/ and defective/ under {root}. Run "
            "`python -c 'from solarvision.real_data import fetch_elpv; fetch_elpv()'`."
        )

    features: list[np.ndarray] = []
    labels: list[int] = []
    for idx, cls in enumerate(_CLASSES):
        paths = sorted((root / cls).glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No images for class {cls!r} under {root}.")
        for path in paths:
            img = np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0
            features.append(extract(img).vector())
            labels.append(idx)
    return np.asarray(features), np.asarray(labels), list(_CLASSES)


def fetch_elpv(n_per_class: int = 45, elpv_dir: str | Path | None = None) -> int:
    """Download a balanced functional/defective subset of ELPV into ``datasets/elpv``.

    Needs network access to the Hugging Face datasets-server. Returns the number of
    images written. Only needed to create or enlarge the vendored subset.
    """
    import json
    import urllib.request

    root = Path(elpv_dir) if elpv_dir is not None else _ELPV_DIR
    func: list[str] = []
    defe: list[str] = []
    offset = 0
    while (len(func) < n_per_class or len(defe) < n_per_class) and offset < 1846:
        url = _HF_ROWS_URL.format(offset=offset)
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed HTTPS URL
            payload = json.load(resp)
        for row in payload["rows"]:
            cell = row["row"]
            src = cell["image"]["src"]
            if cell["label"] == _HF_FUNCTIONAL_LABEL and len(func) < n_per_class:
                func.append(src)
            elif cell["label"] != _HF_FUNCTIONAL_LABEL and len(defe) < n_per_class:
                defe.append(src)
        offset += 100

    written = 0
    for srcs, cls in ((func, "functional"), (defe, "defective")):
        target = root / cls
        target.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(srcs):
            with urllib.request.urlopen(src, timeout=60) as resp:  # noqa: S310 - signed HF URL
                (target / f"{i:03d}.png").write_bytes(resp.read())
            written += 1
    return written
