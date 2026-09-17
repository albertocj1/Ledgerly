"""
Extraction evaluation.

Reports accuracy per field per noise level, against the ground truth
carried by the synthetic documents. Aggregate extraction accuracy is a
misleading number - dates and totals fail for different reasons and
need different fixes - so the breakdown is the output.

Scoring is exact match after normalisation. That's strict on purpose:
a total of 28040.79 read as 28040.7 is wrong in an accounting system,
not "close".
"""
from __future__ import annotations

import json
from pathlib import Path

from src.common.noise import LEVELS, degrade
from src.common.synth import read_jsonl
from src.extraction.extractor import SPECS, extract

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def _matches(predicted, truth) -> bool:
    if predicted is None:
        return False
    if isinstance(truth, float) or isinstance(predicted, float):
        try:
            return abs(float(predicted) - float(truth)) < 0.005
        except (TypeError, ValueError):
            return False
    return str(predicted).strip() == str(truth).strip()


def evaluate_level(docs, level: str, seed: int = 11) -> dict:
    # field -> [correct, attempted_total]
    tally: dict[str, list[int]] = {}

    for i, doc in enumerate(docs):
        specs = SPECS.get(doc.doc_type, [])
        if not specs:
            continue
        text = degrade(doc.text, level, seed=seed + i)
        got = extract(text, doc.doc_type)

        for spec in specs:
            truth = doc.entities.get(spec.name)
            if truth is None:
                continue  # field not present in this document's ground truth
            key = f"{doc.doc_type}.{spec.name}"
            slot = tally.setdefault(key, [0, 0])
            slot[1] += 1
            if _matches(got.get(spec.name), truth):
                slot[0] += 1

    per_field = {k: round(v[0] / v[1], 4) for k, v in sorted(tally.items()) if v[1]}
    total_correct = sum(v[0] for v in tally.values())
    total_seen = sum(v[1] for v in tally.values())
    return {
        "level": level,
        "overall": round(total_correct / total_seen, 4) if total_seen else 0.0,
        "per_field": per_field,
    }


def run_all() -> list[dict]:
    path = Path(__file__).resolve().parents[2] / "data" / "documents.jsonl"
    docs = read_jsonl(path)
    return [evaluate_level(docs, lv) for lv in LEVELS]


def format_report(results: list[dict]) -> str:
    lines = ["Overall exact-match accuracy by input quality", ""]
    lines.append("  " + "  ".join(f"{r['level']:>10}" for r in results))
    lines.append("  " + "  ".join(f"{r['overall']:>10.4f}" for r in results))
    lines.append("")
    lines.append("Per-field accuracy")
    lines.append("")

    fields = sorted({f for r in results for f in r["per_field"]})
    header = f"{'field':<40}" + "".join(f"{r['level']:>10}" for r in results)
    lines.append(header)
    lines.append("-" * len(header))
    for f in fields:
        row = f"{f:<40}" + "".join(f"{r['per_field'].get(f, float('nan')):>10.4f}" for r in results)
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    results = run_all()
    print()
    print(format_report(results))
    print()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "extraction_eval.json").write_text(json.dumps(results, indent=2))
