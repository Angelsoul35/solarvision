"""Benchmark the classical-CV approach on the REAL ELPV solar-cell images.

Extracts the same hand-defined features used on synthetic panels and trains on
real electroluminescence images (functional vs defective), reporting whether the
model is correctly fit on real data: strong on unseen folds and with its training
score close to its test score. It compares a regularised logistic model against a
random forest to show the capacity trade-off explicitly.

Run it with ``python -m solarvision.real_benchmark``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .real_data import load_elpv

_SEED = 42
_N_SPLITS = 5


@dataclass
class ModelResult:
    name: str
    cv_acc_mean: float
    cv_acc_std: float
    cv_auc: float
    train_acc: float
    test_acc: float
    gap: float = field(init=False)

    def __post_init__(self) -> None:
        self.gap = round(self.train_acc - self.test_acc, 4)


@dataclass
class RealBenchmarkReport:
    n_images: int
    classes: list[str]
    chance: float
    logistic: ModelResult
    forest: ModelResult
    confusion: list[list[int]]

    @property
    def verdict(self) -> str:
        lg = self.logistic
        if lg.cv_acc_mean <= self.chance + 0.08:
            return "underfit: barely above chance on real images"
        if lg.gap > 0.15:
            return "overfit: training accuracy runs well ahead of test"
        return "well-fit: clearly above chance with train and test accuracy close"


def _logistic() -> Pipeline:
    return Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, C=1.0))]
    )


def _forest() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=5, random_state=_SEED)),
        ]
    )


def _evaluate(name: str, pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> ModelResult:
    cv = StratifiedKFold(n_splits=_N_SPLITS, shuffle=True, random_state=_SEED)
    acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=_SEED
    )
    pipe.fit(X_train, y_train)
    return ModelResult(
        name=name,
        cv_acc_mean=round(float(acc.mean()), 4),
        cv_acc_std=round(float(acc.std()), 4),
        cv_auc=round(float(auc.mean()), 4),
        train_acc=round(float((pipe.predict(X_train) == y_train).mean()), 4),
        test_acc=round(float((pipe.predict(X_test) == y_test).mean()), 4),
    )


def run_benchmark(
    data: tuple[np.ndarray, np.ndarray, list[str]] | None = None,
) -> RealBenchmarkReport:
    X, y, classes = data if data is not None else load_elpv()
    logistic = _evaluate("Logistic (regularised)", _logistic(), X, y)
    forest = _evaluate("Random forest", _forest(), X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=_SEED
    )
    pipe = _logistic().fit(X_train, y_train)
    cm = confusion_matrix(y_test, pipe.predict(X_test))

    return RealBenchmarkReport(
        n_images=int(len(y)),
        classes=classes,
        chance=0.5,
        logistic=logistic,
        forest=forest,
        confusion=cm.tolist(),
    )


def format_report(r: RealBenchmarkReport) -> str:
    lines = [
        "SolarVision — real-image benchmark (ELPV electroluminescence)",
        "=" * 60,
        f"images: {r.n_images}   task: {r.classes[0]} vs {r.classes[1]}   chance: {r.chance:.2f}",
        "",
        f"{'model':<24} {'CV accuracy':<16} {'CV AUC':>7} {'train':>7} {'test':>7} {'gap':>7}",
    ]
    for m in (r.logistic, r.forest):
        lines.append(
            f"{m.name:<24} {m.cv_acc_mean:.3f} +/- {m.cv_acc_std:.3f}   "
            f"{m.cv_auc:>7.3f} {m.train_acc:>7.3f} {m.test_acc:>7.3f} {m.gap:>7.3f}"
        )
    lines += ["", f"confusion (logistic, rows=true {r.classes}):"]
    lines += [f"  {row}" for row in r.confusion]
    lines += ["", f"verdict: {r.verdict}"]
    return "\n".join(lines)


def main() -> None:
    print(format_report(run_benchmark()))


if __name__ == "__main__":  # pragma: no cover
    main()
