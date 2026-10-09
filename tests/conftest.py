from __future__ import annotations

import numpy as np
import pytest

from solarvision.detector import DefectDetector
from solarvision.generate import render
from solarvision.labels import DefectClass
from solarvision.service import SolarVisionService


@pytest.fixture(scope="session")
def detector() -> DefectDetector:
    # Fitting trains the random forest once for the whole session.
    return DefectDetector().fit()


@pytest.fixture(scope="session")
def service() -> SolarVisionService:
    return SolarVisionService()


def make_panel(
    defect: DefectClass | str, seed: int, severity: float = 0.85, size_frac: float = 0.7
) -> np.ndarray:
    """Render one clear, deterministic panel for a test."""
    rng = np.random.default_rng(seed)
    cls = DefectClass(defect)
    if cls is DefectClass.HEALTHY:
        return render(cls, rng)
    return render(cls, rng, severity=severity, size_frac=size_frac)
