"""The defect detector: a deterministic rule layer in front of a small classifier.

The design goal is an **explainable** decision, not a black box. Two stages:

1. **Rule layer.** For the unambiguous cases a single hand-written threshold is
   both correct and self-explaining, so we short-circuit the model: a very bright,
   localised blob is a *hotspot*; a long thin dark streak is a *crack*; a large
   compact dark block is a *dead cell*. These rules fire only on clear evidence
   and never invent a defect out of noise.
2. **Classifier.** When no rule fires (the common, ambiguous middle ground) a
   small :class:`RandomForestClassifier` over the classical features decides. Its
   per-class probability is the confidence, and the report names the features that
   carried the most weight, so a reviewer can still see *why*.

The forest is seeded (``random_state``) and the rules are pure functions of the
features, so the whole detector is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .features import FEATURE_NAMES, Features, extract, extract_matrix
from .generate import make_dataset
from .labels import DefectClass, describe

# Confidence reported for a rule-layer hit: the rules fire only on clear evidence,
# so they are treated as high-confidence, but not a dishonest 1.0.
_RULE_CONFIDENCE = 0.95


@dataclass
class Assessment:
    """The explainable result of assessing one panel image."""

    predicted_class: str
    confidence: float
    needs_review: bool
    decided_by: str  # "rule" or "model"
    reason: str
    evidence: list[tuple[str, float]] = field(default_factory=list)
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return describe(self.predicted_class).title


class DefectDetector:
    def __init__(
        self,
        min_confidence: float = 0.55,
        hotspot_contrast_z: float = 6.0,
        dead_cell_dark_frac: float = 0.020,
        crack_min_elongation: float = 4.0,
    ) -> None:
        self.min_confidence = min_confidence
        self.hotspot_contrast_z = hotspot_contrast_z
        self.dead_cell_dark_frac = dead_cell_dark_frac
        self.crack_min_elongation = crack_min_elongation
        self._model: RandomForestClassifier | None = None

    # -- training -------------------------------------------------------------
    def fit(
        self, images: list[np.ndarray] | None = None, labels: list[str] | None = None
    ) -> DefectDetector:
        if images is None or labels is None:
            images, labels = make_dataset()
        x = extract_matrix(images)
        model = RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=0, n_jobs=1
        )
        model.fit(x, labels)
        self._model = model
        return self

    def fit_features(self, x: np.ndarray, labels: list[str]) -> DefectDetector:
        """Fit directly from a precomputed feature matrix (used by the evaluation)."""
        model = RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=0, n_jobs=1
        )
        model.fit(x, labels)
        self._model = model
        return self

    def _require_fitted(self) -> RandomForestClassifier:
        if self._model is None:
            raise RuntimeError("detector is not fitted; call fit() first")
        return self._model

    # -- rule layer -----------------------------------------------------------
    def _apply_rules(self, f: Features) -> tuple[str, str] | None:
        """Return ``(class, reason)`` if a deterministic rule fires, else ``None``.

        Order matters: a hotspot (bright) and a dead cell/crack (dark) are mutually
        exclusive signatures, but we check the strongest, least ambiguous evidence
        first.
        """
        if f.bright_blob_contrast >= self.hotspot_contrast_z and f.bright_blob_area > 0.002:
            return (
                DefectClass.HOTSPOT.value,
                f"bright blob at {f.bright_blob_contrast:.1f} robust SD above background "
                f"(>= {self.hotspot_contrast_z:.1f}) over "
                f"{f.bright_blob_area * 100:.1f}% of the panel",
            )
        if (
            f.dark_blob_area >= self.dead_cell_dark_frac
            and f.dark_blob_fill >= 0.55
            and f.dark_blob_elongation < 3.0
        ):
            return (
                DefectClass.DEAD_CELL.value,
                f"compact dark block over {f.dark_blob_area * 100:.1f}% of the panel "
                f"(>= {self.dead_cell_dark_frac * 100:.1f}%), {f.dark_blob_fill:.2f} box-fill",
            )
        if (
            f.dark_blob_elongation >= self.crack_min_elongation
            and f.dark_blob_fill < 0.5
            and f.dark_blob_area > 0.001
        ):
            return (
                DefectClass.CRACK.value,
                f"thin dark streak, elongation {f.dark_blob_elongation:.1f} "
                f"(>= {self.crack_min_elongation:.1f}), {f.dark_blob_fill:.2f} box-fill",
            )
        return None

    # -- model explanation ----------------------------------------------------
    def _model_evidence(self, f: Features, k: int = 4) -> list[tuple[str, float]]:
        """The features that mattered most, as (name, value) pairs.

        A random forest gives *global* importances; we surface the globally most
        important features together with this panel's values, so the reviewer sees
        which measurements the decision leaned on.
        """
        model = self._require_fitted()
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        values = f.vector()
        return [(FEATURE_NAMES[i], round(float(values[i]), 4)) for i in order[:k]]

    # -- inference ------------------------------------------------------------
    def assess_features(self, f: Features) -> Assessment:
        """Assess an already-extracted feature vector (rule layer, then classifier)."""
        model = self._require_fitted()
        proba = model.predict_proba(f.vector().reshape(1, -1))[0]
        probabilities = {
            str(cls): round(float(p), 4) for cls, p in zip(model.classes_, proba, strict=True)
        }

        rule = self._apply_rules(f)
        if rule is not None:
            cls, reason = rule
            return Assessment(
                predicted_class=cls,
                confidence=_RULE_CONFIDENCE,
                needs_review=False,
                decided_by="rule",
                reason=reason,
                evidence=self._model_evidence(f),
                probabilities=probabilities,
            )

        best = int(np.argmax(proba))
        cls = str(model.classes_[best])
        confidence = float(proba[best])
        info = describe(cls)
        return Assessment(
            predicted_class=cls,
            confidence=confidence,
            needs_review=confidence < self.min_confidence,
            decided_by="model",
            reason=f"classifier: {info.signature}",
            evidence=self._model_evidence(f),
            probabilities=probabilities,
        )

    def assess(self, img: np.ndarray) -> Assessment:
        """Assess a grayscale panel image in ``[0, 1]``."""
        return self.assess_features(extract(img))

    def predict(self, images: list[np.ndarray]) -> list[str]:
        return [self.assess(im).predicted_class for im in images]

    def predict_features_batch(self, feats: list[Features]) -> list[str]:
        """Classify many feature vectors at once (rule layer, then batched model).

        Identical in outcome to calling :meth:`assess_features` per item, but the
        classifier runs once over the whole matrix instead of once per sample, so
        the evaluation over thousands of images stays fast.
        """
        model = self._require_fitted()
        if not feats:
            return []
        matrix = np.vstack([f.vector() for f in feats])
        proba = model.predict_proba(matrix)
        classes = model.classes_
        out: list[str] = []
        for f, row in zip(feats, proba, strict=True):
            rule = self._apply_rules(f)
            out.append(rule[0] if rule is not None else str(classes[int(np.argmax(row))]))
        return out
