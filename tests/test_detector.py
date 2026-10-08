from __future__ import annotations

import pytest

from solarvision.detector import DefectDetector
from solarvision.labels import DefectClass
from tests.conftest import make_panel


def test_unfitted_raises():
    with pytest.raises(RuntimeError):
        DefectDetector().assess(make_panel(DefectClass.HEALTHY, seed=1))


@pytest.mark.parametrize(
    "defect,seed",
    [
        (DefectClass.HEALTHY, 1),
        (DefectClass.HOTSPOT, 2),
        (DefectClass.CRACK, 3),
        (DefectClass.DEAD_CELL, 4),
    ],
)
def test_detects_each_clear_defect(detector: DefectDetector, defect: DefectClass, seed: int):
    # Clear, strong instances of every class must be classified correctly.
    for s in range(seed, seed + 20, 5):
        a = detector.assess(make_panel(defect, seed=s))
        assert a.predicted_class == defect.value, f"{defect} (seed {s}) -> {a.predicted_class}"
        assert a.probabilities  # full probability vector returned
        assert a.evidence  # explanation populated


def test_report_is_explainable(detector: DefectDetector):
    a = detector.assess(make_panel(DefectClass.HOTSPOT, seed=2))
    assert a.decided_by in {"rule", "model"}
    assert a.reason
    assert 0.0 <= a.confidence <= 1.0
    # A strong, bright, localised blob is exactly the hotspot rule's territory.
    assert a.decided_by == "rule"


def test_low_confidence_is_flagged_for_review():
    # An impossibly high threshold routes even a clear model call to review; the
    # rule layer is disabled so the decision goes through the classifier.
    det = DefectDetector(
        min_confidence=0.999,
        hotspot_contrast_z=1e9,
        dead_cell_dark_frac=1e9,
        crack_min_elongation=1e9,
    ).fit()
    a = det.assess(make_panel(DefectClass.CRACK, seed=3))
    assert a.decided_by == "model"
    assert a.needs_review


def test_rule_layer_fires_on_unambiguous_cases(detector: DefectDetector):
    hot = detector.assess(make_panel(DefectClass.HOTSPOT, seed=2))
    dead = detector.assess(make_panel(DefectClass.DEAD_CELL, seed=4))
    assert hot.decided_by == "rule" and hot.predicted_class == "hotspot"
    assert dead.decided_by == "rule" and dead.predicted_class == "dead_cell"
