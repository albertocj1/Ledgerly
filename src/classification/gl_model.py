"""
GL account categorizer.

Design decisions worth defending in an interview:

* **Character n-grams, not words.** Transaction memos are truncated,
  concatenated and abbreviated ("AMZN MKTP US*2H4TY", "amazon web se").
  Word tokenisation shatters on those; char n-grams (3-5) degrade
  gracefully and handle the truncation directly.

* **Union of word + char features.** Word features still help on clean
  memos, so the two are combined rather than chosen between.

* **class_weight="balanced".** The ledger is ~15:1 imbalanced. Without
  reweighting, the rare accounts (Payroll_Wages, Legal_Fees) get
  swallowed and macro-F1 collapses even as accuracy looks fine.

* **Macro-F1 is the headline metric,** not accuracy. On an imbalanced
  ledger accuracy is dominated by the two biggest classes; macro-F1 is
  the metric that reflects whether rare accounts actually work.

* **Abstention.** Predictions below a confidence threshold are routed
  to human review rather than auto-posted. For a CPA firm, a wrong
  auto-posted entry costs far more than a flagged one, so the
  precision/coverage trade-off is reported explicitly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.common.ledger import Transaction, read_jsonl, theoretical_ceiling

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def build_features() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
        ]
    )


def build_pipeline(model: str = "logreg") -> Pipeline:
    if model == "logreg":
        clf = LogisticRegression(max_iter=2000, C=8.0, class_weight="balanced")
    elif model == "linsvc":
        # Wrapped for probability estimates so the abstention threshold
        # has something calibrated to threshold on.
        clf = CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced"), cv=3, method="sigmoid"
        )
    else:
        raise ValueError(f"Unknown model {model!r}")
    return Pipeline([("features", build_features()), ("clf", clf)])


@dataclass
class GLEvalResult:
    model: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    bayes_ceiling: float
    gap_to_ceiling: float
    coverage_at_threshold: dict[str, float]
    accuracy_at_threshold: dict[str, float]
    report: str


def evaluate_abstention(
    pipe: Pipeline, X_test: list[str], y_test: list[str], thresholds=(0.5, 0.7, 0.85, 0.95)
) -> tuple[dict[str, float], dict[str, float]]:
    """Measure the coverage / accuracy trade-off under abstention.

    At each confidence threshold: what fraction of transactions does the
    model auto-post (coverage), and how accurate is it on those?
    """
    probs = pipe.predict_proba(X_test)
    preds = pipe.classes_[np.argmax(probs, axis=1)]
    conf = probs.max(axis=1)
    y = np.array(y_test)

    coverage, accuracy = {}, {}
    for t in thresholds:
        mask = conf >= t
        cov = float(mask.mean())
        coverage[str(t)] = round(cov, 4)
        accuracy[str(t)] = round(float((preds[mask] == y[mask]).mean()), 4) if mask.any() else float("nan")
    return coverage, accuracy


def train_and_evaluate(
    txns: list[Transaction], model: str = "logreg", seed: int = 42
) -> tuple[Pipeline, GLEvalResult]:
    X = [t.memo for t in txns]
    y = [t.gl_account for t in txns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    pipe = build_pipeline(model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = float((np.array(y_pred) == np.array(y_test)).mean())
    macro = float(f1_score(y_test, y_pred, average="macro"))
    weighted = float(f1_score(y_test, y_pred, average="weighted"))
    ceiling = theoretical_ceiling()

    coverage, acc_at_t = evaluate_abstention(pipe, X_test, y_test)

    result = GLEvalResult(
        model=model,
        accuracy=round(acc, 4),
        macro_f1=round(macro, 4),
        weighted_f1=round(weighted, 4),
        bayes_ceiling=round(ceiling, 4),
        gap_to_ceiling=round(ceiling - acc, 4),
        coverage_at_threshold=coverage,
        accuracy_at_threshold=acc_at_t,
        report=classification_report(y_test, y_pred, zero_division=0),
    )
    return pipe, result


def save_model(pipe: Pipeline, result: GLEvalResult, name: str = "gl_categorizer") -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"{name}.joblib"
    joblib.dump(pipe, path)
    payload = {k: v for k, v in result.__dict__.items() if k != "report"}
    (ARTIFACT_DIR / f"{name}_metrics.json").write_text(json.dumps(payload, indent=2))
    return path


def load_model(name: str = "gl_categorizer") -> Pipeline:
    path = ARTIFACT_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run: python -m src.classification.gl_model")
    return joblib.load(path)


def predict(pipe: Pipeline, memo: str, threshold: float = 0.7) -> dict:
    """Predict a GL account, abstaining when confidence is low."""
    probs = pipe.predict_proba([memo])[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    account = str(pipe.classes_[idx])

    order = np.argsort(probs)[::-1][:3]
    alternatives = [
        {"account": str(pipe.classes_[i]), "confidence": round(float(probs[i]), 4)}
        for i in order
    ]

    return {
        "memo": memo,
        "predicted_account": account if confidence >= threshold else None,
        "confidence": round(confidence, 4),
        "needs_review": confidence < threshold,
        "alternatives": alternatives,
    }


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[2] / "data" / "transactions.jsonl"
    txns = read_jsonl(data_path)

    for m in ("logreg", "linsvc"):
        pipe, res = train_and_evaluate(txns, model=m)
        print(f"\n=== {m} ===")
        print(f"accuracy      {res.accuracy:.4f}")
        print(f"macro F1      {res.macro_f1:.4f}")
        print(f"weighted F1   {res.weighted_f1:.4f}")
        print(f"Bayes ceiling {res.bayes_ceiling:.4f}   (gap {res.gap_to_ceiling:+.4f})")
        print(f"coverage @thr {res.coverage_at_threshold}")
        print(f"accuracy @thr {res.accuracy_at_threshold}")
        if m == "logreg":
            save_model(pipe, res)
