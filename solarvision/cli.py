"""Command-line demo and entry point."""

from __future__ import annotations

import sys

import numpy as np

from .generate import IMG_SIZE
from .labels import ALL_CLASSES, DefectClass, describe
from .service import SolarVisionService

# Intensity ramp for the ASCII preview, dark -> bright.
_RAMP = " .:-=+*#%@"


def _ascii_art(img: np.ndarray, cols: int = 32, rows: int = 16) -> str:
    """Render a grayscale panel as ASCII by block-averaging it down to a small grid.

    Characters are taller than they are wide, so we use more columns than rows to
    keep the panel looking square in the terminal.
    """
    ys = np.linspace(0, IMG_SIZE, rows + 1).astype(int)
    xs = np.linspace(0, IMG_SIZE, cols + 1).astype(int)
    lines = []
    for r in range(rows):
        chars = []
        for c in range(cols):
            block = img[ys[r] : ys[r + 1], xs[c] : xs[c + 1]]
            level = float(block.mean()) if block.size else 0.0
            chars.append(_RAMP[min(len(_RAMP) - 1, int(level * len(_RAMP)))])
        lines.append("   " + "".join(chars))
    return "\n".join(lines)


def demo() -> None:
    svc = SolarVisionService()

    print("SolarVision demo\n" + "=" * 60)
    print(
        "\nGenerating one synthetic panel of each class and detecting defects.\n"
        "Each panel is rendered deterministically (no real image, no network)."
    )

    # One clear example of each class, rendered deterministically from fixed seeds.
    samples = [
        (DefectClass.HEALTHY, 1, 0.0, 0.0),
        (DefectClass.HOTSPOT, 2, 0.8, 0.6),
        (DefectClass.CRACK, 3, 0.8, 0.8),
        (DefectClass.DEAD_CELL, 4, 0.8, 0.7),
    ]

    correct = 0
    for cls, seed, severity, size_frac in samples:
        result = svc.assess_sample(cls, seed=seed, severity=severity, size_frac=size_frac)
        a = result.assessment
        img = svc.generate_sample(cls, seed=seed, severity=severity, size_frac=size_frac)
        correct += int(result.correct)

        print(f"\n[{describe(cls).title}]  ground truth: {result.ground_truth}")
        print(_ascii_art(img))
        mark = "OK " if result.correct else "XX "
        print(
            f"   {mark}detected: {a.predicted_class:<10} "
            f"confidence {a.confidence:.2f}  via {a.decided_by}"
            + ("  [NEEDS REVIEW]" if a.needs_review else "")
        )
        print(f"   why: {a.reason}")
        ev = ", ".join(f"{name}={val}" for name, val in a.evidence[:3])
        print(f"   key features: {ev}")

    print("\n" + "=" * 60)
    print(f"Detected {correct}/{len(samples)} clear samples correctly.")
    print(
        "These are deliberately clear instances. The honest, held-out accuracy on\n"
        "UNSEEN defect configurations (faint defects included) is reported by\n"
        "`python -m solarvision.evaluation` and is well below 1.0 -- see the README."
    )


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "demo":
        demo()
        return 0
    print("usage: python -m solarvision.cli demo")
    print(f"classes: {', '.join(ALL_CLASSES)}")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
