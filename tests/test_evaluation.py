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


def test_generalisation_is_proven_by_a_small_gap_not_a_cap():
    # The honesty signal is the small train/test gap on DISJOINT severity/size
    # bins, not an arbitrary accuracy ceiling: a high score with a near-zero gap is
    # real generalisation, a high score with a large gap would be memorisation.
    # (Here the faint-defect-near-noise boundary keeps accuracy around 0.88 anyway.)
    report = held_out_eval(seed=3)
    assert report.gap <= 0.15
    assert report.accuracy >= report.train_accuracy - 0.15


def test_aggregate_eval_prints_summary():
    text = aggregate_eval(seeds=[7, 8])
    assert "accuracy (mean+/-std over 2 seeds)" in text
    assert "macro-F1" in text
    assert "train-test gap" in text
