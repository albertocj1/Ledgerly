"""
Validate the out-of-scope guardrail.

Thresholds tuned on a handful of examples aren't trustworthy, so this
script checks them against the full labelled query set (all in-scope)
plus a set of out-of-scope probes, and reports false-refusal and
false-accept rates.

Run after changing the corpus or the embedder:
    python -m scripts.tune_threshold
"""
from __future__ import annotations

from src.rag.evaluate import GOLD
from src.rag.generate import MIN_BM25, MIN_COSINE
from src.rag.retriever import HybridRetriever

# Out-of-scope probes. Deliberately varied: everyday trivia, adjacent
# technical topics, and finance-flavoured questions the corpus does NOT
# cover (the last group is the hard case - topically close but absent).
OUT_OF_SCOPE = [
    "What is the capital of France?",
    "best pizza in Rome",
    "how do I train a neural network",
    "who won the world cup",
    "what is the weather tomorrow",
    "write me a poem about autumn",
    "how do I center a div in CSS",
    "what is the boiling point of water",
    # Topically adjacent but not in the corpus:
    "what is the R&D tax credit calculation",
    "how do I compute depreciation under MACRS",
    "what are the estate tax exemption amounts",
    "explain FBAR filing requirements",
    "what is the QBI deduction phase-out",
]


def is_grounded(retriever: HybridRetriever, query: str) -> tuple[bool, dict]:
    s = retriever.relevance_signals(query)
    return (s["max_bm25"] >= MIN_BM25 or s["max_cosine"] >= MIN_COSINE), s


def main() -> None:
    retriever = HybridRetriever()

    false_refusals = []
    for gq in GOLD:
        ok, sig = is_grounded(retriever, gq.query)
        if not ok:
            false_refusals.append((gq.query, sig))

    false_accepts = []
    for q in OUT_OF_SCOPE:
        ok, sig = is_grounded(retriever, q)
        if ok:
            false_accepts.append((q, sig))

    print(f"Thresholds: BM25 >= {MIN_BM25}, cosine >= {MIN_COSINE}\n")
    print(f"In-scope queries:     {len(GOLD)}")
    print(f"  false refusals:     {len(false_refusals)} "
          f"({len(false_refusals)/len(GOLD):.1%})")
    for q, s in false_refusals:
        print(f"    - {q[:58]:<58} bm25={s['max_bm25']:.2f} cos={s['max_cosine']:.2f}")

    print(f"\nOut-of-scope probes:  {len(OUT_OF_SCOPE)}")
    print(f"  false accepts:      {len(false_accepts)} "
          f"({len(false_accepts)/len(OUT_OF_SCOPE):.1%})")
    for q, s in false_accepts:
        print(f"    - {q[:58]:<58} bm25={s['max_bm25']:.2f} cos={s['max_cosine']:.2f}")


if __name__ == "__main__":
    main()
