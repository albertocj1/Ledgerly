"""
Document type classifier.

Two models, deliberately:

1. `tfidf_lr`  - TF-IDF + Logistic Regression. Fast, CPU-only, fully
   interpretable (you can read the learned coefficients), no model
   download required. This is the production default.
2. `sgd`       - linear SVM via SGD, used as a comparison baseline.

The point of having two isn't complexity for its own sake - it's that
model selection should be an evidence-based decision, and the
evaluation harness below produces that evidence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.common.synth import DOC_TYPES, Document, read_jsonl

ModelName = Literal["tfidf_lr", "sgd"]

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def build_pipeline(model: ModelName = "tfidf_lr") -> Pipeline:
    """Build a text-classification pipeline.

    Word-level n-grams up to bigrams capture phrases like "purchase
    order" and "nonemployee compensation" that are strongly indicative
    of a single document type.
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    if model == "tfidf_lr":
        clf = LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced")
    elif model == "sgd":
        clf = SGDClassifier(loss="hinge", alpha=1e-4, max_iter=2000, class_weight="balanced", random_state=42)
    else:
        raise ValueError(f"Unknown model: {model}")

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


@dataclass
class EvalResult:
    model: ModelName
    accuracy: float
    macro_f1: float
    weighted_f1: float
    cv_macro_f1_mean: float
    cv_macro_f1_std: float
    per_class_f1: dict[str, float]
    report: str
    confusion: list[list[int]]

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def train_and_evaluate(
    docs: list[Document],
    model: ModelName = "tfidf_lr",
    test_size: float = 0.25,
    seed: int = 42,
    run_cv: bool = True,
) -> tuple[Pipeline, EvalResult]:
    """Train on a stratified split and report held-out performance."""
    X = [d.text for d in docs]
    y = [d.doc_type for d in docs]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipe = build_pipeline(model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    macro = f1_score(y_test, y_pred, average="macro")
    weighted = f1_score(y_test, y_pred, average="weighted")
    acc = float((np.array(y_pred) == np.array(y_test)).mean())

    per_class = {
        label: float(f1_score(y_test, y_pred, labels=[label], average="macro"))
        for label in DOC_TYPES
    }

    if run_cv:
        cv_scores = cross_val_score(build_pipeline(model), X, y, cv=5, scoring="f1_macro")
        cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
    else:
        cv_mean, cv_std = float("nan"), float("nan")

    result = EvalResult(
        model=model,
        accuracy=acc,
        macro_f1=float(macro),
        weighted_f1=float(weighted),
        cv_macro_f1_mean=cv_mean,
        cv_macro_f1_std=cv_std,
        per_class_f1=per_class,
        report=classification_report(y_test, y_pred, zero_division=0),
        confusion=confusion_matrix(y_test, y_pred, labels=DOC_TYPES).tolist(),
    )
    return pipe, result


def save_model(pipe: Pipeline, result: EvalResult, name: str = "classifier") -> Path:
    """Persist the model alongside its evaluation metrics.

    Storing metrics next to the artifact is a small MLOps habit that
    matters: it means you can always answer "how good was the model
    that's currently in production, and on what data?"
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / f"{name}.joblib"
    joblib.dump(pipe, model_path)
    (ARTIFACT_DIR / f"{name}_metrics.json").write_text(
        json.dumps(
            {k: v for k, v in result.to_dict().items() if k != "report"},
            indent=2,
        )
    )
    return model_path


def load_model(name: str = "classifier") -> Pipeline:
    path = ARTIFACT_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run: python -m src.classification.train"
        )
    return joblib.load(path)


def predict_with_confidence(pipe: Pipeline, text: str) -> tuple[str, float]:
    """Predict a document type and return a calibrated-ish confidence.

    For LogisticRegression we get real probabilities. For SGD/hinge we
    fall back to a normalised decision margin, which is ordinal but not
    a true probability - flagged here so nobody downstream treats it
    as one.
    """
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        probs = pipe.predict_proba([text])[0]
        idx = int(np.argmax(probs))
        return str(pipe.classes_[idx]), float(probs[idx])

    scores = pipe.decision_function([text])[0]
    idx = int(np.argmax(scores))
    exp = np.exp(scores - scores.max())
    return str(pipe.classes_[idx]), float(exp[idx] / exp.sum())


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[2] / "data" / "documents.jsonl"
    documents = read_jsonl(data_path)
    _, res = train_and_evaluate(documents)
    print(res.report)
