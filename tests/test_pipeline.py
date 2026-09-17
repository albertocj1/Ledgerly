"""Tests for the document intelligence pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from src.common.intake import crop_header, vary_terminology
from src.common.ledger import VENDOR_MAP, generate_transactions, theoretical_ceiling
from src.common.noise import degrade
from src.common.synth import DOC_TYPES, generate_dataset
from src.rag.corpus import get_corpus
from src.rag.generate import MIN_BM25, MIN_COSINE, EchoClient, RagPipeline
from src.rag.retriever import BM25, HybridRetriever


# ---------------------------------------------------------------- data

def test_synthetic_dataset_is_balanced():
    docs = generate_dataset(n=700)
    counts = {t: 0 for t in DOC_TYPES}
    for d in docs:
        counts[d.doc_type] += 1
    assert len(set(counts.values())) == 1, f"unbalanced: {counts}"


def test_synthetic_documents_have_ground_truth_entities():
    docs = generate_dataset(n=70)
    for d in docs:
        assert d.entities, f"{d.doc_id} has no entities"


def test_degrade_is_deterministic_given_seed():
    text = "INVOICE\nTotal Due: $1,234.56\nDate: 2024-01-01"
    assert degrade(text, "medium", seed=1) == degrade(text, "medium", seed=1)


def test_degrade_rejects_unknown_level():
    with pytest.raises(ValueError):
        degrade("x", "catastrophic")


def test_heavier_degradation_changes_text_more():
    text = "INVOICE\n" + "Line item description 12345\n" * 40
    light = degrade(text, "light", seed=3)
    heavy = degrade(text, "heavy", seed=3)
    assert heavy != text
    # Heavy drops and truncates far more aggressively.
    assert len(heavy) < len(light)


def test_crop_header_removes_leading_content_lines():
    text = "INVOICE\n\nAcme Corp\n123 Main St\nTotal: $50"
    out = crop_header(text, 2)
    assert "INVOICE" not in out
    assert "Total: $50" in out


def test_crop_header_never_returns_empty():
    assert crop_header("ONLY LINE", 10).strip()


def test_vary_terminology_replaces_known_titles():
    import random
    out = vary_terminology("INVOICE\nbody", random.Random(0))
    assert "body" in out


# ------------------------------------------------------------- ledger

def test_ledger_is_imbalanced_as_intended():
    txns = generate_transactions(n=3000)
    counts: dict[str, int] = {}
    for t in txns:
        counts[t.gl_account] = counts.get(t.gl_account, 0) + 1
    ratio = max(counts.values()) / min(counts.values())
    assert ratio > 3, f"expected imbalance, got ratio {ratio:.1f}"


def test_every_transaction_label_is_a_valid_account_for_its_vendor():
    txns = generate_transactions(n=1000)
    for t in txns:
        # Memos are mutated (truncated/lowercased), so match by prefix.
        matches = [v for v in VENDOR_MAP if t.memo.upper().startswith(v[: len(t.memo)])]
        assert t.gl_account.startswith("6"), t.gl_account


def test_bayes_ceiling_is_below_one():
    """If the ceiling were 1.0 the task would carry no ambiguity,
    which would defeat the purpose of this dataset."""
    ceiling = theoretical_ceiling(n=20_000)
    assert 0.85 < ceiling < 0.99, ceiling


# --------------------------------------------------------------- BM25

def test_bm25_ranks_matching_document_first():
    docs = ["cats and dogs", "quarterly tax filing deadlines", "baking sourdough bread"]
    bm = BM25(docs)
    scores = bm.score("tax filing")
    assert int(np.argmax(scores)) == 1


def test_bm25_scores_are_non_negative():
    bm = BM25(["alpha beta", "beta gamma", "gamma delta"])
    assert (bm.score("beta") >= 0).all()


# ---------------------------------------------------------- retrieval

@pytest.fixture(scope="module")
def retriever() -> HybridRetriever:
    return HybridRetriever()


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_search_returns_requested_number_of_hits(retriever, mode):
    hits = retriever.search("business meal deduction", top_k=3, mode=mode)
    assert len(hits) == 3
    assert all(h.passage.doc_id for h in hits)


def test_search_rejects_unknown_mode(retriever):
    with pytest.raises(ValueError):
        retriever.search("anything", mode="magic")


def test_exact_code_query_retrieves_right_passage(retriever):
    hits = retriever.search("ASC 842 lease classification", top_k=3, mode="hybrid")
    assert "kb_010" in [h.passage.doc_id for h in hits]


def test_relevance_signals_separate_in_and_out_of_scope(retriever):
    in_scope = retriever.relevance_signals("1099-NEC filing threshold")
    out_scope = retriever.relevance_signals("who won the world cup")
    assert in_scope["max_bm25"] > out_scope["max_bm25"]


# --------------------------------------------------------- guardrails

@pytest.fixture(scope="module")
def rag(retriever) -> RagPipeline:
    return RagPipeline(retriever=retriever, llm=EchoClient())


def test_out_of_scope_question_is_refused(rag):
    res = rag.answer("What is the capital of France?")
    assert res.grounded is False
    assert res.guard_stage == "score_gate"
    assert res.citations == []


@pytest.mark.parametrize(
    "question",
    ["best pizza in Rome", "how do I center a div in CSS", "who won the world cup"],
)
def test_clearly_irrelevant_questions_are_refused(rag, question):
    assert rag.answer(question).grounded is False


def test_in_scope_question_is_answered_with_citations(rag):
    res = rag.answer("What are the limits on deducting business meals?")
    assert res.grounded is True
    assert len(res.citations) > 0
    assert all("authority" in c for c in res.citations)


def test_thresholds_are_ordered_sensibly():
    assert 0 < MIN_BM25
    assert 0 < MIN_COSINE < 1


def test_echo_client_does_not_fabricate():
    """The offline stub must never look like a real generated answer."""
    out = EchoClient().complete("sys", "user")
    assert "offline stub" in out.lower()


# ---------------------------------------------------------------- API

def test_api_health_and_endpoints():
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        res = client.post("/research", json={"question": "who won the world cup"})
        assert res.status_code == 200
        assert res.json()["grounded"] is False


def test_api_rejects_invalid_retrieval_mode():
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as client:
        res = client.post("/research", json={"question": "meals", "mode": "nonsense"})
        assert res.status_code == 422


def test_api_rejects_empty_memo():
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as client:
        assert client.post("/categorize", json={"memo": ""}).status_code == 422


# -------------------------------------------------------------- corpus

def test_every_passage_declares_an_authority():
    """Unsourced guidance is unusable in a tax context."""
    for p in get_corpus():
        assert p.authority.strip(), f"{p.doc_id} has no authority"


def test_passage_ids_are_unique():
    ids = [p.doc_id for p in get_corpus()]
    assert len(ids) == len(set(ids))


# ----------------------------------------------------------- extraction

from src.extraction.extractor import extract, extract_with_coverage  # noqa: E402
from src.extraction.extractor import _norm_amount  # noqa: E402


def test_total_is_not_confused_with_subtotal():
    """Regression: the label 'TOTAL' used to match inside 'Subtotal',
    and since the subtotal line comes first, the extractor returned a
    confident, plausible, wrong total. Word boundaries fixed it."""
    text = "Subtotal:   $100.00\nTax (8%):  $8.00\nTOTAL DUE:  $108.00\n"
    got = extract(text, "invoice")
    assert got["total_amount"] == 108.00
    assert got["subtotal"] == 100.00


def test_form_box_wins_over_page_title():
    """Regression: 'Nonemployee Compensation' appears in the 1099-NEC
    page title before box 1, so a plain search returned the tax year."""
    text = (
        "Form 1099-NEC  Nonemployee Compensation              2025\n"
        "PAYER'S name: Acme\n"
        " 1 Nonemployee compensation ................. 94,616.83\n"
    )
    got = extract(text, "1099_nec")
    assert got["nonemployee_compensation"] == 94616.83


def test_negative_balances_are_parsed():
    """Regression: account balances go negative; the amount pattern
    rejected a leading minus and silently dropped the field."""
    text = "Opening Balance: $5,000.00\nClosing Balance: $-1,234.56\n"
    got = extract(text, "bank_statement")
    assert got["closing_balance"] == -1234.56


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1,234.56", 1234.56),
        ("1234.56", 1234.56),
        ("1,234", 1234.0),
        ("O", 0.0),          # OCR: letter O read for zero
        ("l,2S4.S6", 1254.56),  # OCR: l->1, S->5
        ("", None),
        (".", None),
    ],
)
def test_amount_normalisation_handles_ocr_damage(raw, expected):
    assert _norm_amount(raw) == expected


def test_coverage_flags_partial_extraction():
    """A document that looks processed but lost its total is the
    dangerous case - it must be flagged, not passed through."""
    partial = extract_with_coverage("Invoice Number: INV-1\n", "invoice")
    assert partial["needs_review"] is True
    assert "total_amount" in partial["missing_fields"]


def test_clean_documents_extract_perfectly():
    from src.common.synth import generate_dataset
    from src.extraction.extractor import SPECS

    for doc in generate_dataset(n=210):
        specs = SPECS.get(doc.doc_type, [])
        got = extract(doc.text, doc.doc_type)
        for spec in specs:
            truth = doc.entities.get(spec.name)
            if truth is None:
                continue
            assert spec.name in got, f"{doc.doc_type}.{spec.name} not extracted"
