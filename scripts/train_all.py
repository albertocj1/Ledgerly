"""
Train every model and write all artifacts.

    python -m scripts.train_all

Regenerates data if missing, trains the document router and the GL
categorizer, and writes metrics JSON alongside each model so the
artifact is always self-describing.
"""
from __future__ import annotations

from pathlib import Path

from src.classification import gl_model
from src.classification.model import save_model, train_and_evaluate
from src.common import ledger, synth

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def ensure_data() -> None:
    docs_path = DATA / "documents.jsonl"
    txns_path = DATA / "transactions.jsonl"

    if not docs_path.exists():
        print("Generating documents...")
        synth.write_jsonl(synth.generate_dataset(), docs_path)
    if not txns_path.exists():
        print("Generating transactions...")
        ledger.write_jsonl(ledger.generate_transactions(), txns_path)


def main() -> None:
    ensure_data()

    print("\n=== Document router ===")
    docs = synth.read_jsonl(DATA / "documents.jsonl")
    router, router_res = train_and_evaluate(docs, run_cv=False)
    save_model(router, router_res)
    print(f"macro-F1 {router_res.macro_f1:.4f} (clean templates)")
    print("Note: this task is near-saturated by design - it is a cheap")
    print("routing stage, not the project's headline result.")

    print("\n=== GL categorizer ===")
    txns = ledger.read_jsonl(DATA / "transactions.jsonl")
    gl, gl_res = gl_model.train_and_evaluate(txns)
    gl_model.save_model(gl, gl_res)
    print(f"accuracy      {gl_res.accuracy:.4f}")
    print(f"macro-F1      {gl_res.macro_f1:.4f}")
    print(f"Bayes ceiling {gl_res.bayes_ceiling:.4f} (gap {gl_res.gap_to_ceiling:+.4f})")

    print(f"\nArtifacts written to {ROOT / 'artifacts'}")


if __name__ == "__main__":
    main()
