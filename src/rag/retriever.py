"""
Hybrid retriever: BM25 (lexical) + dense embeddings, fused with
reciprocal rank fusion.

The motivation: tax and accounting queries mix exact terminology that
must match literally ("1099-NEC", "ASC 842", "de minimis safe harbor")
with paraphrased intent ("can we write off the team lunch"). BM25
handles the codes; dense retrieval handles the paraphrase.

What the benchmark actually found (see `evaluate.py`, and the numbers
in the README): on this corpus **dense alone outperforms hybrid** -
recall@3 of 1.00 vs 0.94, driven entirely by paraphrase queries
(1.00 vs 0.89). BM25 is perfect on exact-code lookups but drops to
0.79 on paraphrases, and fusing it in pulls the strong dense ranking
down.

The explanation is specific to the fallback embedder: `SvdEmbedder`
is LSA over word *and character* n-grams, so it already carries the
lexical signal BM25 would contribute, and the fusion adds noise
instead of coverage. With a true sentence-transformer embedder -
which has no character-level lexical matching - the usual hybrid
advantage on exact codes is expected to reappear.

All three modes are kept and benchmarked rather than hard-coding the
winner, because the right choice depends on the embedder in use. Note
also that this corpus is small (16 passages, 35 queries), so these
numbers are indicative, not decisive - a production evaluation needs
a far larger labelled set.

BM25 is implemented directly (~40 lines) rather than pulled in as a
dependency - it keeps the install light and makes the scoring auditable.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.rag.corpus import Passage, get_corpus
from src.rag.embeddings import Embedder, get_embedder

_TOKEN = re.compile(r"[a-z0-9§]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Hit:
    passage: Passage
    score: float
    rank: int


class BM25:
    """Okapi BM25. k1 and b at the usual defaults."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = [tokenize(d) for d in docs]
        self.n = len(self.corpus)
        self.doc_len = [len(d) for d in self.corpus]
        self.avgdl = sum(self.doc_len) / max(self.n, 1)

        df: Counter[str] = Counter()
        for doc in self.corpus:
            df.update(set(doc))
        # Smoothed IDF, floored at zero to avoid negative contributions
        # from terms present in almost every document.
        self.idf = {
            term: max(math.log((self.n - freq + 0.5) / (freq + 0.5) + 1.0), 0.0)
            for term, freq in df.items()
        }
        self.tf = [Counter(doc) for doc in self.corpus]

    def score(self, query: str) -> np.ndarray:
        q_terms = tokenize(query)
        scores = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            dl = self.doc_len[i]
            tf_i = self.tf[i]
            s = 0.0
            for term in q_terms:
                if term not in tf_i:
                    continue
                freq = tf_i[term]
                idf = self.idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                s += idf * (freq * (self.k1 + 1)) / max(denom, 1e-9)
            scores[i] = s
        return scores


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Fuse ranked ID lists. RRF is robust because it uses ranks, not
    raw scores, so BM25 and cosine scales never need calibrating."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


class HybridRetriever:
    def __init__(self, passages: list[Passage] | None = None, embedder: Embedder | None = None):
        self.passages = passages or get_corpus()
        self._texts = [f"{p.title}. {p.text}" for p in self.passages]
        self.bm25 = BM25(self._texts)
        self.embedder = embedder or get_embedder()
        self.embedder.fit(self._texts)
        self.matrix = self.embedder.encode(self._texts)

    def _dense_scores(self, query: str) -> np.ndarray:
        q = self.embedder.encode([query])[0]
        return self.matrix @ q  # vectors are L2-normalized, so this is cosine

    def relevance_signals(self, query: str) -> dict[str, float]:
        """Mode-independent signals for out-of-scope detection.

        RRF fusion scores can't be used for this: they're built from
        ranks, so the top result always scores about the same whether
        the query is on-topic or nonsense. These two raw signals are
        comparable across queries, which is what a threshold needs.
        """
        return {
            "max_cosine": float(self._dense_scores(query).max()),
            "max_bm25": float(self.bm25.score(query).max()),
        }

    def search(self, query: str, top_k: int = 4, mode: str = "hybrid") -> list[Hit]:
        if mode == "bm25":
            scores = self.bm25.score(query)
            order = np.argsort(scores)[::-1][:top_k]
            return [Hit(self.passages[i], float(scores[i]), r) for r, i in enumerate(order)]

        if mode == "dense":
            scores = self._dense_scores(query)
            order = np.argsort(scores)[::-1][:top_k]
            return [Hit(self.passages[i], float(scores[i]), r) for r, i in enumerate(order)]

        if mode == "hybrid":
            bm = self.bm25.score(query)
            dn = self._dense_scores(query)
            depth = min(len(self.passages), max(top_k * 3, 10))
            bm_rank = list(np.argsort(bm)[::-1][:depth])
            dn_rank = list(np.argsort(dn)[::-1][:depth])
            fused = reciprocal_rank_fusion([bm_rank, dn_rank])
            order = sorted(fused, key=lambda i: fused[i], reverse=True)[:top_k]
            return [Hit(self.passages[i], float(fused[i]), r) for r, i in enumerate(order)]

        raise ValueError(f"Unknown mode {mode!r}; expected bm25, dense or hybrid")
