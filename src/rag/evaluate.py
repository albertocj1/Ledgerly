"""
Retrieval evaluation.

Most RAG portfolio projects ship a retriever and simply assert it
works. This module measures it against a hand-labelled query set, so
the choice of hybrid over pure vector search is an evidence-backed
decision rather than a claim.

Queries are written the way a tax associate would actually type them -
including paraphrases that share no vocabulary with the passage, and
exact-code lookups that dense retrieval alone tends to miss. That split
is the whole point of the benchmark.

Metrics: Recall@k, MRR, and Hit@1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.rag.retriever import HybridRetriever

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"


@dataclass
class GoldQuery:
    query: str
    relevant_ids: list[str]
    kind: str  # "exact" | "paraphrase"


# Hand-labelled evaluation set.
GOLD: list[GoldQuery] = [
    GoldQuery("what percentage of business meals can we deduct", ["kb_001"], "paraphrase"),
    GoldQuery("IRC 274(n) meal limitation", ["kb_001"], "exact"),
    GoldQuery("can we write off the team lunch with a client", ["kb_001"], "paraphrase"),
    GoldQuery("1099-NEC filing threshold", ["kb_002"], "exact"),
    GoldQuery("when do we have to send a contractor a tax form", ["kb_002"], "paraphrase"),
    GoldQuery("do we report payments made to a law firm", ["kb_002"], "paraphrase"),
    GoldQuery("employee versus independent contractor test", ["kb_003"], "exact"),
    GoldQuery("how do we decide if someone is staff or freelance", ["kb_003"], "paraphrase"),
    GoldQuery("de minimis safe harbor threshold", ["kb_004"], "exact"),
    GoldQuery("should we capitalize or expense a new laptop", ["kb_004"], "paraphrase"),
    GoldQuery("how are SaaS subscriptions treated for tax", ["kb_005"], "paraphrase"),
    GoldQuery("hosted software subscription expense treatment", ["kb_005"], "exact"),
    GoldQuery("what records do we need for a business trip", ["kb_006"], "paraphrase"),
    GoldQuery("travel substantiation amount time place business purpose", ["kb_006"], "exact"),
    GoldQuery("per diem instead of receipts for meals", ["kb_006"], "paraphrase"),
    GoldQuery("cash versus accrual method of accounting", ["kb_007"], "exact"),
    GoldQuery("when do we recognize income if we have not been paid yet", ["kb_007"], "paraphrase"),
    GoldQuery("can our audit team also do the bookkeeping", ["kb_008"], "paraphrase"),
    GoldQuery("independence impairment non-attest services", ["kb_008"], "exact"),
    GoldQuery("ASC 606 five step model", ["kb_009"], "exact"),
    GoldQuery("how do we book revenue on a multi-part contract", ["kb_009"], "paraphrase"),
    GoldQuery("ASC 842 lease classification", ["kb_010"], "exact"),
    GoldQuery("do we have to put the office lease on the balance sheet", ["kb_010"], "paraphrase"),
    GoldQuery("accountable plan reimbursement rules", ["kb_011"], "exact"),
    GoldQuery("are expense reimbursements taxable to the employee", ["kb_011"], "paraphrase"),
    GoldQuery("how long should we keep tax records", ["kb_012"], "paraphrase"),
    GoldQuery("record retention assessment period three years", ["kb_012"], "exact"),
    GoldQuery("are stripe processing fees deductible", ["kb_013"], "paraphrase"),
    GoldQuery("bank service charge wire fee deduction", ["kb_013"], "exact"),
    GoldQuery("is google ad spend deductible this year", ["kb_014"], "paraphrase"),
    GoldQuery("advertising versus goodwill capitalization", ["kb_014"], "exact"),
    GoldQuery("can we deduct a CPA review course", ["kb_015"], "paraphrase"),
    GoldQuery("continuing education deduction new trade or business", ["kb_015"], "exact"),
    GoldQuery("disclosing transactions with an affiliate company", ["kb_016"], "paraphrase"),
    GoldQuery("related party arm's length disclosure", ["kb_016"], "exact"),
    # --- Hard cases, added after they were found failing in manual
    # testing. Kept in the set deliberately: removing queries a system
    # fails on is how benchmarks become decoration. These drag the
    # headline numbers down, which is the honest outcome.
    #
    # "client dinner" is dominated by the token "client", which appears
    # throughout the audit-independence passage. With only 16 documents
    # BM25's IDF estimate for common domain words is unreliable, so a
    # near-stopword outranks the real signal. A larger corpus, or a
    # domain stopword list, is the fix - not a threshold tweak.
    GoldQuery("Can we deduct the full cost of a client dinner?", ["kb_001"], "paraphrase"),
    GoldQuery("client entertainment at a hotel restaurant", ["kb_001"], "paraphrase"),
    GoldQuery("our client paid a contractor 800 dollars", ["kb_002"], "paraphrase"),
]


def evaluate(retriever: HybridRetriever, mode: str, k: int = 3) -> dict:
    recall_hits = 0
    hit_at_1 = 0
    mrr_total = 0.0

    per_kind: dict[str, list[int]] = {"exact": [], "paraphrase": []}

    for gq in GOLD:
        hits = retriever.search(gq.query, top_k=k, mode=mode)
        ids = [h.passage.doc_id for h in hits]

        found = any(i in gq.relevant_ids for i in ids)
        recall_hits += int(found)
        per_kind[gq.kind].append(int(found))

        if ids and ids[0] in gq.relevant_ids:
            hit_at_1 += 1

        rr = 0.0
        for rank, doc_id in enumerate(ids, start=1):
            if doc_id in gq.relevant_ids:
                rr = 1.0 / rank
                break
        mrr_total += rr

    n = len(GOLD)
    return {
        "mode": mode,
        "k": k,
        f"recall@{k}": round(recall_hits / n, 4),
        "hit@1": round(hit_at_1 / n, 4),
        "mrr": round(mrr_total / n, 4),
        f"recall@{k}_exact": round(sum(per_kind["exact"]) / max(len(per_kind["exact"]), 1), 4),
        f"recall@{k}_paraphrase": round(
            sum(per_kind["paraphrase"]) / max(len(per_kind["paraphrase"]), 1), 4
        ),
    }


def run_all(k: int = 3) -> list[dict]:
    retriever = HybridRetriever()
    return [evaluate(retriever, mode, k=k) for mode in ("bm25", "dense", "hybrid")]


def format_table(results: list[dict], k: int = 3) -> str:
    cols = [f"recall@{k}", "hit@1", "mrr", f"recall@{k}_exact", f"recall@{k}_paraphrase"]
    header = f"{'mode':<8}" + "".join(f"{c:>22}" for c in cols)
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(f"{r['mode']:<8}" + "".join(f"{r[c]:>22.4f}" for c in cols))
    return "\n".join(lines)


if __name__ == "__main__":
    k = 3
    results = run_all(k=k)
    print(f"\nRetrieval quality over {len(GOLD)} labelled queries\n")
    print(format_table(results, k=k))
    print()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "retrieval_eval.json").write_text(json.dumps(results, indent=2))
