"""Leakage-free, held-out evaluation of the defect detector.

A random train/test split of synthetic images would report a meaningless
near-1.0: the two sets would share the exact defect configurations, so the model
would only have to recognise contrasts and sizes it had already seen.
:func:`held_out_eval` instead trains on one pool of severity/size bins and tests
on a *disjoint* pool (see :func:`solarvision.generate.make_split_datasets`), so
every test defect is an unseen configuration and the number reflects real
generalisation.

The whole detector is evaluated end-to-end (the deterministic rule layer *and* the
classifier), because that is what runs in production. The report pairs accuracy
with the macro-F1 (so a class the detector is weak on cannot hide behind the
easier ones) and the train/test gap (so memorisation would show up as a large
gap). :func:`aggregate_eval` averages over several seeds, because a single split
can land on a lucky number and reporting that would be the same self-deception as
the random split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .detector import DefectDetector
from .features import extract
from .generate import make_split_datasets
from .labels import ALL_CLASSES


@dataclass
class EvalReport:
    accuracy: float
    macro_f1: float
    train_accuracy: float
    gap: float
    n_train: int
    n_test: int

    def __str__(self) -> str:
        return (
            f"accuracy          : {self.accuracy:.2f}   # defect configs unseen in training\n"
            f"macro-F1          : {self.macro_f1:.2f}   # all four classes weighted equally\n"
            f"train accuracy    : {self.train_accuracy:.2f}\n"
            f"train-test gap    : {self.gap:.2f}   # bounded -> signal, not memorised configs\n"
            f"n_train / n_test  : {self.n_train} / {self.n_test}"
        )


def held_out_eval(seed: int = 7, holdout_frac: float = 0.35) -> EvalReport:
    """Train on one pool of severity/size bins, test on a disjoint pool."""
    tr_x, tr_y, te_x, te_y = make_split_datasets(seed=seed, holdout_frac=holdout_frac)

    tr_feats = [extract(im) for im in tr_x]
    te_feats = [extract(im) for im in te_x]

    detector = DefectDetector().fit_features(np.vstack([f.vector() for f in tr_feats]), tr_y)

    test_pred = detector.predict_features_batch(te_feats)
    train_pred = detector.predict_features_batch(tr_feats)

    acc = float(accuracy_score(te_y, test_pred))
    train_acc = float(accuracy_score(tr_y, train_pred))
    macro = float(f1_score(te_y, test_pred, labels=ALL_CLASSES, average="macro", zero_division=0))
    return EvalReport(
        accuracy=acc,
        macro_f1=macro,
        train_accuracy=train_acc,
        gap=round(train_acc - acc, 4),
        n_train=len(tr_x),
        n_test=len(te_x),
    )


def aggregate_eval(seeds: range | list[int] = range(5), holdout_frac: float = 0.35) -> str:
    """Average the held-out eval over several seeds for a robust headline number."""
    reports = [held_out_eval(seed=s, holdout_frac=holdout_frac) for s in seeds]
    accs = [r.accuracy for r in reports]
    mean = sum(accs) / len(accs)
    std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
    f1 = sum(r.macro_f1 for r in reports) / len(reports)
    train = sum(r.train_accuracy for r in reports) / len(reports)
    gap = sum(r.gap for r in reports) / len(reports)
    n = len(list(seeds))
    return (
        f"accuracy (mean+/-std over {n} seeds): {mean:.2f} +/- {std:.2f}   "
        f"# four classes; chance ~0.25; defect configs unseen in training\n"
        f"macro-F1 (mean over {n} seeds)      : {f1:.2f}\n"
        f"train accuracy                      : {train:.2f}\n"
        f"train-test gap                      : {gap:.2f}   # bounded -> real signal, not configs"
    )


if __name__ == "__main__":  # pragma: no cover
    print(aggregate_eval())
