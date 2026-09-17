"""
Evaluation harness for the document classifier.

Produces the numbers that actually matter: how the model holds up as
input quality degrades. A single accuracy figure on clean text is
not a result - the degradation curve is.
"""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.classification.model import ARTIFACT_DIR, build_pipeline
from src.common.intake import realistic_intake
from src.common.noise import LEVELS, degrade, degrade_mixed
from src.common.synth import Document, read_jsonl


def _prepare(text: str, level: str, seed: int, hard: bool) -> str:
    """Apply intake variation (if hard mode) then OCR degradation."""
    if hard:
        text = realistic_intake(text, seed=seed)
    if level == "mixed":
        return degrade_mixed(text, seed=seed)
    return degrade(text, level, seed=seed)


def evaluate_robustness(
    docs: list[Document],
    model: str = "tfidf_lr",
    train_on: str = "mixed",
    seed: int = 42,
    hard: bool = True,
) -> dict:
    """Train once, then evaluate across every degradation level.

    `train_on="mixed"` applies the realistic intake mixture during
    training. `train_on="clean"` is the naive baseline - included
    precisely to show how much worse it does on degraded input, which
    is the argument for training-time augmentation.

    `hard=True` additionally applies terminology variation and header
    cropping (see src/common/intake.py). Without it the benchmark
    saturates at 1.00 and carries no information.
    """
    train_docs, test_docs = train_test_split(
        docs, test_size=0.25, random_state=seed, stratify=[d.doc_type for d in docs]
    )

    X_train = [
        _prepare(d.text, train_on, seed=seed + i, hard=hard)
        for i, d in enumerate(train_docs)
    ]
    y_train = [d.doc_type for d in train_docs]
    y_test = [d.doc_type for d in test_docs]

    pipe = build_pipeline(model)  # type: ignore[arg-type]
    pipe.fit(X_train, y_train)

    results: dict[str, float] = {}
    for level in LEVELS:
        X_test = [
            _prepare(d.text, level, seed=seed * 7 + i, hard=hard)
            for i, d in enumerate(test_docs)
        ]
        y_pred = pipe.predict(X_test)
        results[level] = round(float(f1_score(y_test, y_pred, average="macro")), 4)

    return {
        "model": model,
        "trained_on": train_on,
        "hard_mode": hard,
        "macro_f1_by_level": results,
    }


def run_comparison(docs: list[Document]) -> list[dict]:
    """Compare training strategies and model families side by side."""
    configs = [
        ("tfidf_lr", "clean", True),
        ("tfidf_lr", "mixed", True),
        ("sgd", "mixed", True),
    ]
    return [evaluate_robustness(docs, model=m, train_on=t, hard=h) for m, t, h in configs]


def format_table(results: list[dict]) -> str:
    levels = list(LEVELS)
    header = f"{'model':<10} {'trained on':<12} " + " ".join(f"{lv:>8}" for lv in levels)
    lines = [header, "-" * len(header)]
    for r in results:
        row = f"{r['model']:<10} {r['trained_on']:<12} " + " ".join(
            f"{r['macro_f1_by_level'][lv]:>8.4f}" for lv in levels
        )
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[2] / "data" / "documents.jsonl"
    documents = read_jsonl(data_path)
    results = run_comparison(documents)
    table = format_table(results)
    print("\nMacro-F1 by input quality\n")
    print(table)
    print()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "robustness.json").write_text(json.dumps(results, indent=2))
