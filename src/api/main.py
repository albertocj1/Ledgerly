"""
Serving layer.

Exposes the three capabilities behind one API:
  POST /classify    - document type routing
  POST /categorize  - GL account assignment (with abstention)
  POST /research    - grounded Q&A over the knowledge base

Design notes:
* Models load once at startup via lifespan, not per request.
* `/categorize` returns `needs_review` rather than silently
  auto-posting a low-confidence guess. The threshold is a request
  parameter because the right operating point differs between a
  high-volume AP queue and a year-end close.
* `/research` returns `grounded` and `guard_stage` so a caller can
  distinguish "here is an answer" from "we declined to answer",
  and see which guard fired.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.classification import gl_model
from src.rag.generate import RagPipeline

log = logging.getLogger("acctintel.api")

STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup."""
    try:
        STATE["gl"] = gl_model.load_model()
        log.info("GL categorizer loaded.")
    except FileNotFoundError:
        STATE["gl"] = None
        log.warning("GL categorizer artifact missing; /categorize will 503.")

    try:
        from src.classification.model import load_model as load_router

        STATE["router"] = load_router()
        log.info("Document router loaded.")
    except FileNotFoundError:
        STATE["router"] = None
        log.warning("Router artifact missing; /extract requires explicit doc_type.")

    STATE["rag"] = RagPipeline()
    log.info("RAG pipeline ready (llm=%s).", STATE["rag"].llm.name)
    yield
    STATE.clear()


app = FastAPI(
    title="Accounting Document Intelligence",
    version="1.0.0",
    description="Document routing, GL categorization, and grounded tax research.",
    lifespan=lifespan,
)


class CategorizeRequest(BaseModel):
    memo: str = Field(min_length=1, max_length=500, examples=["AMZN MKTP US*2H4TY"])
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)
    mode: str = Field(default="dense", pattern="^(bm25|dense|hybrid)$")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "gl_model": STATE.get("gl") is not None,
        "router_model": STATE.get("router") is not None,
        "llm": STATE["rag"].llm.name if STATE.get("rag") else None,
    }


@app.post("/categorize")
def categorize(req: CategorizeRequest) -> dict:
    pipe = STATE.get("gl")
    if pipe is None:
        raise HTTPException(
            status_code=503,
            detail="GL model not trained. Run: python -m src.classification.gl_model",
        )
    return gl_model.predict(pipe, req.memo, threshold=req.threshold)


@app.post("/categorize/batch")
def categorize_batch(memos: list[str], threshold: float = 0.7) -> dict:
    pipe = STATE.get("gl")
    if pipe is None:
        raise HTTPException(status_code=503, detail="GL model not trained.")
    if len(memos) > 500:
        raise HTTPException(status_code=413, detail="Max 500 memos per batch.")

    results = [gl_model.predict(pipe, m, threshold=threshold) for m in memos]
    flagged = sum(r["needs_review"] for r in results)
    return {
        "count": len(results),
        "auto_posted": len(results) - flagged,
        "needs_review": flagged,
        "results": results,
    }


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)
    doc_type: str | None = Field(
        default=None,
        description="Document type. If omitted, the router classifies it first.",
    )


@app.post("/extract")
def extract_fields(req: ExtractRequest) -> dict:
    """Extract structured fields, classifying the document first if needed."""
    from src.extraction.extractor import extract_with_coverage

    doc_type = req.doc_type
    if doc_type is None:
        router = STATE.get("router")
        if router is None:
            raise HTTPException(
                status_code=400,
                detail="doc_type omitted and no router model available. "
                       "Pass doc_type explicitly, or train the router.",
            )
        from src.classification.model import predict_with_confidence

        doc_type, confidence = predict_with_confidence(router, req.text)
    else:
        confidence = None

    result = extract_with_coverage(req.text, doc_type)
    result["doc_type"] = doc_type
    if confidence is not None:
        result["doc_type_confidence"] = round(confidence, 4)
    return result


@app.post("/research")
def research(req: ResearchRequest) -> dict:
    rag: RagPipeline = STATE["rag"]
    res = rag.answer(req.question, top_k=req.top_k, mode=req.mode)
    return {
        "question": res.question,
        "answer": res.answer,
        "grounded": res.grounded,
        "guard_stage": res.guard_stage,
        "citations": res.citations,
        "retrieval_mode": res.retrieval_mode,
        "llm": res.llm,
    }
