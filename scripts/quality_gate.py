"""
Model quality gate.

Fails the build when a model regresses below an agreed threshold, the
same way a broken unit test does. Metrics that live only in a notebook
don't protect production.

Thresholds are set a little below current measured performance - tight
enough to catch a real regression, loose enough to absorb ordinary
run-to-run variance. They are floors to defend, not targets to hit.

    python -m scripts.quality_gate
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.classification.gl_model import train_and_evaluate
from src.common.ledger import read_jsonl
from src.rag.evaluate import run_all

# Measured at time of writing:
#   GL macro-F1      0.897   (Bayes ceiling 0.936)
#   GL accuracy      0.927
#   dense recall@3   1.000
#   dense hit@1      0.816
GL_MIN_MACRO_F1 = 0.86
GL_MIN_ACCURACY = 0.90
RAG_MIN_RECALL_AT_3 = 0.92
RAG_MIN_HIT_AT_1 = 0.75


def main() -> int:
    failures: list[str] = []

    # --- GL categorizer ---
    data = Path(__file__).resolve().parents[1] / "data" / "transactions.jsonl"
    if not data.exists():
        print("ERROR: transactions.jsonl missing. Run: python -m src.common.ledger")
        return 2

    _, gl = train_and_evaluate(read_jsonl(data))
    print(f"GL macro-F1   {gl.macro_f1:.4f}  (floor {GL_MIN_MACRO_F1})")
    print(f"GL accuracy   {gl.accuracy:.4f}  (floor {GL_MIN_ACCURACY})")
    print(f"Bayes ceiling {gl.bayes_ceiling:.4f}  gap {gl.gap_to_ceiling:+.4f}")

    if gl.macro_f1 < GL_MIN_MACRO_F1:
        failures.append(f"GL macro-F1 {gl.macro_f1:.4f} < {GL_MIN_MACRO_F1}")
    if gl.accuracy < GL_MIN_ACCURACY:
        failures.append(f"GL accuracy {gl.accuracy:.4f} < {GL_MIN_ACCURACY}")

    # --- Retrieval ---
    results = {r["mode"]: r for r in run_all(k=3)}
    dense = results["dense"]
    print(f"\ndense recall@3 {dense['recall@3']:.4f}  (floor {RAG_MIN_RECALL_AT_3})")
    print(f"dense hit@1    {dense['hit@1']:.4f}  (floor {RAG_MIN_HIT_AT_1})")

    if dense["recall@3"] < RAG_MIN_RECALL_AT_3:
        failures.append(f"recall@3 {dense['recall@3']:.4f} < {RAG_MIN_RECALL_AT_3}")
    if dense["hit@1"] < RAG_MIN_HIT_AT_1:
        failures.append(f"hit@1 {dense['hit@1']:.4f} < {RAG_MIN_HIT_AT_1}")

    print()
    if failures:
        print("QUALITY GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("Quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
