"""
Answer generation over retrieved passages.

Two things matter more than the prompt itself in a tax/advisory
context:

1. **Grounding.** The model must answer only from retrieved passages
   and say so plainly when they don't cover the question. An invented
   deduction threshold is worse than no answer.

2. **Citation.** Every claim carries its source passage. "Where does
   this come from" is the first question any reviewer asks.

The LLM client is provider-agnostic with an Azure OpenAI path first,
since the target environment is Azure. `EchoClient` is a deterministic
offline stub so the pipeline - and its tests - run without credentials.
"""
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import Protocol

from src.rag.retriever import Hit, HybridRetriever

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a research assistant for a CPA and advisory firm.

    Rules, in priority order:
    1. Answer ONLY from the numbered context passages provided. Do not
       use outside knowledge, even if you are confident it is correct.
    2. If the passages do not contain enough information to answer,
       say exactly: "The knowledge base does not cover this." Then
       state what additional source would be needed. Do not guess.
    3. Cite the passage number inline for every factual claim, like [2].
    4. Note the authority shown for each passage you rely on.
    5. Be concise and precise with numbers, thresholds, and dates.
    6. Close with a one-line reminder that this is research support,
       not a filing position, and requires practitioner review.

    Never present a summary of general tax knowledge as if it came
    from the passages.
    """
)


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class EchoClient:
    """Offline deterministic stub.

    Returns an extractive answer built from the retrieved passages so
    the end-to-end pipeline is testable without an API key. It does not
    paraphrase or reason - that is the point: any fluency you see in
    the output of this client came from the corpus, not a model.
    """

    name = "echo-offline"

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002
        return (
            "[offline stub] No LLM provider configured, so no generated answer.\n"
            "The retrieved passages below are the grounding that would be sent "
            "to the model. Set AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY "
            "(or OPENAI_API_KEY) to enable generation."
        )


class AzureOpenAIClient:
    """Azure OpenAI chat completions."""

    def __init__(self, deployment: str | None = None):
        from openai import AzureOpenAI

        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        api_key = os.environ["AZURE_OPENAI_API_KEY"]
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
        self.deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        self._client = AzureOpenAI(
            azure_endpoint=endpoint, api_key=api_key, api_version=api_version
        )
        self.name = f"azure:{self.deployment}"

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,  # deterministic: this is research support, not prose
            max_tokens=700,
        )
        return resp.choices[0].message.content or ""


class OpenAIClient:
    """Direct OpenAI API, for local development."""

    def __init__(self, model: str | None = None):
        from openai import OpenAI

        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.name = f"openai:{self.model}"

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=700,
        )
        return resp.choices[0].message.content or ""


def get_llm_client() -> LLMClient:
    """Pick a provider from the environment, Azure first."""
    import logging

    log = logging.getLogger(__name__)
    if os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY"):
        try:
            return AzureOpenAIClient()
        except Exception as exc:  # noqa: BLE001
            log.warning("Azure OpenAI init failed (%s); trying OpenAI.", exc.__class__.__name__)
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIClient()
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenAI init failed (%s); using offline stub.", exc.__class__.__name__)
    return EchoClient()


SUFFICIENCY_PROMPT = textwrap.dedent(
    """\
    You are a strict relevance judge for a tax research system.

    You will be given a question and numbered context passages. Decide
    whether the passages contain enough information to answer the
    question directly.

    Answer with one word only: SUFFICIENT or INSUFFICIENT.

    Judge strictly. Passages that are merely on the same general topic
    (both about tax, both about deductions) are INSUFFICIENT unless
    they address the specific question asked. If the question asks
    about a rule, threshold, form, or regime that the passages never
    mention, answer INSUFFICIENT.
    """
)


@dataclass
class RagAnswer:
    question: str
    answer: str
    citations: list[dict]
    retrieval_mode: str
    llm: str
    grounded: bool
    guard_stage: str = "passed"  # which stage rejected, if any


def build_context(hits: list[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        p = hit.passage
        blocks.append(
            f"[{i}] {p.title}\nAuthority: {p.authority}\n{p.text}"
        )
    return "\n\n".join(blocks)


# Out-of-scope thresholds.
#
# An earlier version gated on the top hit's ranking score, which was a
# bug: under RRF the top result scores ~1/(k+1) regardless of whether
# the query is on-topic, so "What is the capital of France?" sailed
# through as grounded. These thresholds are set from the measured
# separation between in-scope and out-of-scope queries on this corpus
# (in-scope: BM25 >= 4.6, cosine >= 0.75; out-of-scope: BM25 <= 1.3,
# cosine <= 0.66). Either signal clearing its bar is enough, so an
# exact-code lookup with little semantic overlap still passes.
#
# These are corpus- and embedder-specific. Re-fit them when either
# changes; `scripts/tune_threshold.py` regenerates the numbers.
MIN_BM25 = 2.5
MIN_COSINE = 0.72


class RagPipeline:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        llm: LLMClient | None = None,
        mode: str = "dense",
        check_sufficiency: bool = True,
    ):
        # Default mode is "dense" because that's what the retrieval
        # benchmark actually favoured on this corpus - see
        # src/rag/evaluate.py. Override per-call if your embedder differs.
        self.retriever = retriever or HybridRetriever()
        self.llm = llm or get_llm_client()
        self.mode = mode
        # Costs one extra LLM call per query. Worth it here: the
        # failure it prevents is a confidently wrong tax answer.
        # Disable only for latency-critical, low-stakes paths.
        self.check_sufficiency = check_sufficiency

    def answer(self, question: str, top_k: int = 4, mode: str | None = None) -> RagAnswer:
        mode = mode or self.mode
        hits = self.retriever.search(question, top_k=top_k, mode=mode)

        # ---- Stage 1: cheap retrieval-score gate ----
        # Catches obvious off-topic input at no LLM cost. Checked on raw
        # retrieval signals, not the ranking score - see the threshold
        # comment above for why that distinction matters.
        signals = self.retriever.relevance_signals(question)
        passed_scores = bool(hits) and (
            signals["max_bm25"] >= MIN_BM25 or signals["max_cosine"] >= MIN_COSINE
        )
        if not passed_scores:
            return RagAnswer(
                question=question,
                answer="The knowledge base does not cover this.",
                citations=[],
                retrieval_mode=mode,
                llm=self.llm.name,
                grounded=False,
                guard_stage="score_gate",
            )

        context = build_context(hits)

        # ---- Stage 2: LLM sufficiency check ----
        # Stage 1 alone is not enough. Measured on this corpus it lets
        # through ~38% of topically-adjacent but uncovered questions
        # ("QBI deduction phase-out", "FBAR filing requirements") -
        # they share tax vocabulary with the corpus, so no similarity
        # threshold can separate them. Those are precisely the queries
        # where an ungrounded answer is most damaging, because the
        # output will look plausible.
        if self.check_sufficiency and not isinstance(self.llm, EchoClient):
            verdict = self.llm.complete(
                SUFFICIENCY_PROMPT,
                f"Context passages:\n\n{context}\n\nQuestion: {question}",
            ).strip().upper()
            if verdict.startswith("INSUFFICIENT"):
                return RagAnswer(
                    question=question,
                    answer=(
                        "The knowledge base does not cover this. The retrieved "
                        "passages are related but do not address the specific "
                        "question. A practitioner should consult the firm's "
                        "primary research library."
                    ),
                    citations=[],
                    retrieval_mode=mode,
                    llm=self.llm.name,
                    grounded=False,
                    guard_stage="sufficiency_check",
                )
        user = f"Context passages:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
        text = self.llm.complete(SYSTEM_PROMPT, user)

        citations = [
            {
                "n": i,
                "doc_id": h.passage.doc_id,
                "title": h.passage.title,
                "authority": h.passage.authority,
                "score": round(h.score, 4),
            }
            for i, h in enumerate(hits, start=1)
        ]

        return RagAnswer(
            question=question,
            answer=text,
            citations=citations,
            retrieval_mode=mode,
            llm=self.llm.name,
            grounded=True,
        )


if __name__ == "__main__":
    pipe = RagPipeline()
    for q in [
        "Can we deduct the full cost of a client dinner?",
        "What is the capital of France?",
    ]:
        res = pipe.answer(q)
        print("=" * 70)
        print("Q:", res.question)
        print("grounded:", res.grounded, "| llm:", res.llm)
        print(res.answer[:400])
        for c in res.citations:
            print(f"   [{c['n']}] {c['title']}  ({c['authority']})")
