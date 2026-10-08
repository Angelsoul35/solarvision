from __future__ import annotations

from solarvision.evaluation import aggregate_eval, held_out_eval


def test_held_out_generalises_without_memorising():
    # Honest evaluation: whole severity/size bins are held out of training, so the
    # test defect configurations are unseen and a near-1.0 cannot hide
    # memorisation. Floors are set conservatively below the measured numbers.
    report = held_out_eval()
    assert report.accuracy >= 0.80  # four classes; chance is ~0.25
    assert report.macro_f1 >= 0.78
    assert report.train_accuracy >= 0.90
    assert report.gap <= 0.15  # bounded -> learns the signal, not the configs


def test_held_out_accuracy_is_not_a_suspicious_one():
    # The faint end of the severity range genuinely overlaps healthy under noise,
    # so an honest held-out score must leave room below 1.0.
    report = held_out_eval(seed=3)
    assert report.accuracy < 0.995


def test_aggregate_eval_prints_summary():
    text = aggregate_eval(seeds=[7, 8])
    assert "accuracy (mean+/-std over 2 seeds)" in text
    assert "macro-F1" in text
    assert "train-test gap" in text
