"""Application service: ties the renderer, the feature layer and the detector together."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Settings, get_settings
from .detector import Assessment, DefectDetector
from .generate import decode_base64_png, encode_base64_png, render
from .labels import DefectClass


@dataclass
class SampleAssessment:
    """An assessment of a generated sample, paired with its known ground truth."""

    ground_truth: str
    assessment: Assessment
    image_base64: str

    @property
    def correct(self) -> bool:
        return self.assessment.predicted_class == self.ground_truth


class SolarVisionService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.detector = DefectDetector(
            min_confidence=self.settings.min_confidence,
            hotspot_contrast_z=self.settings.hotspot_contrast_z,
            dead_cell_dark_frac=self.settings.dead_cell_dark_frac,
            crack_min_elongation=self.settings.crack_min_elongation,
        ).fit()

    def assess_image(self, img: np.ndarray) -> Assessment:
        return self.detector.assess(img)

    def assess_base64_png(self, data: str) -> Assessment:
        return self.detector.assess(decode_base64_png(data))

    def generate_sample(
        self,
        defect: DefectClass | str,
        seed: int = 0,
        severity: float = 0.7,
        size_frac: float = 0.6,
    ) -> np.ndarray:
        """Render one labelled sample panel deterministically from a seed."""
        rng = np.random.default_rng(seed)
        cls = DefectClass(defect)
        if cls is DefectClass.HEALTHY:
            return render(cls, rng)
        return render(cls, rng, severity=severity, size_frac=size_frac)

    def assess_sample(
        self,
        defect: DefectClass | str,
        seed: int = 0,
        severity: float = 0.7,
        size_frac: float = 0.6,
    ) -> SampleAssessment:
        img = self.generate_sample(defect, seed=seed, severity=severity, size_frac=size_frac)
        return SampleAssessment(
            ground_truth=DefectClass(defect).value,
            assessment=self.assess_image(img),
            image_base64=encode_base64_png(img),
        )
