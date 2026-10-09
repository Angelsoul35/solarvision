"""Tests for the real ELPV electroluminescence benchmark.

They prove the vendored data is the real ELPV subset, that the classical features
are well fit on real EL cells (clearly above chance, small train/test gap, strong
AUC), and that the loader and CLI work.
"""

from __future__ import annotations

import numpy as np
import pytest

from solarvision.real_benchmark import run_benchmark
from solarvision.real_data import load_elpv


@pytest.fixture(scope="module")
def elpv_data():
    return load_elpv()


@pytest.fixture(scope="module")
def report(elpv_data):
    return run_benchmark(elpv_data)


def test_loads_the_real_elpv_images(elpv_data):
    X, y, classes = elpv_data
    assert classes == ["functional", "defective"]
    assert X.shape == (90, 14)  # 45 per class, 14 classical features
    assert int((y == 0).sum()) == 45
    assert int((y == 1).sum()) == 45


def test_classical_features_are_well_fit_on_real_cells(report):
    lg = report.logistic
    # Not underfit / not a coin flip: clearly above the 0.50 binary chance level.
    assert lg.cv_acc_mean >= 0.78, lg
    assert lg.cv_auc >= 0.80, lg
    # Not overfit: train and test accuracy stay close.
    assert abs(lg.gap) <= 0.15, lg
    assert report.verdict.startswith("well-fit")


def test_shuffled_labels_collapse_to_chance(elpv_data):
    # Regression guard: shuffle the labels and the classifier drops to ~0.5.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, _ = elpv_data
    y_shuffled = np.random.default_rng(0).permutation(y)
    pipe = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=3000))])
    acc = cross_val_score(
        pipe, X, y_shuffled, cv=StratifiedKFold(5, shuffle=True, random_state=0)
    ).mean()
    assert acc <= 0.62, acc  # near the 0.50 chance level, nowhere near 0.88


def test_loader_rejects_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_elpv(tmp_path)


def test_format_report_and_main(report, capsys, monkeypatch):
    import solarvision.real_benchmark as rb

    text = rb.format_report(report)
    assert "ELPV" in text
    assert f"{report.logistic.cv_auc:.3f}" in text
    monkeypatch.setattr(rb, "run_benchmark", lambda data=None: report)
    rb.main()
    assert "real-image benchmark" in capsys.readouterr().out


def _report_with(cv_acc: float, train_acc: float, test_acc: float):
    from solarvision.real_benchmark import ModelResult, RealBenchmarkReport

    model = ModelResult("x", cv_acc, 0.02, 0.9, train_acc, test_acc)
    return RealBenchmarkReport(
        n_images=90,
        classes=["functional", "defective"],
        chance=0.5,
        logistic=model,
        forest=model,
        confusion=[[1, 1], [1, 1]],
    )


def test_verdict_classifies_underfit_overfit_and_well_fit():
    assert _report_with(0.55, 0.56, 0.54).verdict.startswith("underfit")
    assert _report_with(0.88, 0.99, 0.80).verdict.startswith("overfit")
    assert _report_with(0.88, 0.90, 0.86).verdict.startswith("well-fit")


def test_fetch_elpv_downloads_a_balanced_subset(tmp_path, monkeypatch):
    import json

    page = {
        "rows": [
            {"row": {"label": 1, "image": {"src": "http://x/f.png"}}},
            {"row": {"label": 0, "image": {"src": "http://x/d.png"}}},
        ]
    }

    class _Resp:
        def __init__(self, body: bytes):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self._body

    def fake_urlopen(url, timeout=0):
        if "rows?" in url:
            return _Resp(json.dumps(page).encode())
        return _Resp(b"\x89PNG\r\n")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    from solarvision.real_data import fetch_elpv

    written = fetch_elpv(n_per_class=1, elpv_dir=tmp_path)
    assert written == 2
    assert (tmp_path / "functional" / "000.png").exists()
    assert (tmp_path / "defective" / "000.png").exists()
