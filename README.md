# Accounting Document Intelligence Platform

An end-to-end NLP system for a CPA and advisory workflow: it routes
incoming documents by type, extracts structured fields from scanned
text, assigns transactions to general-ledger accounts, and answers
tax questions from a knowledge base with enforced grounding.

Built with scikit-learn, FastAPI, and a provider-agnostic LLM layer
(Azure OpenAI first). Containerised, with a CI pipeline that fails the
build on model regression.

---

## Results

All numbers below are reproducible with `python -m scripts.train_all`
and the evaluation commands in [Reproducing the results](#reproducing-the-results).

### GL account categorization — the core ML task

| Metric | Value |
|---|---|
| Accuracy | **0.927** |
| Macro-F1 | **0.897** |
| Bayes ceiling (estimated) | 0.936 |
| Gap to ceiling | **+0.009** |

20 GL accounts, ~15:1 class imbalance, 6,000 transactions.

**Why the ceiling matters more than the accuracy.** The same vendor
legitimately maps to different accounts — an Amazon charge is office
supplies, computer equipment, or a subscription depending on what was
bought, and the memo text carries no signal about which. The
Bayes-optimal strategy (always predict each vendor's most likely
account) tops out near 0.936. Reporting 0.927 *against that ceiling*
says something a bare accuracy figure cannot: the residual error is
irreducible ambiguity, not a weak model. Chasing it would be wasted
effort; the useful lever is abstention, below.

### Abstention: the business lever

| Confidence threshold | Coverage (auto-posted) | Accuracy on auto-posted |
|---|---|---|
| 0.50 | 98.8% | 93.5% |
| 0.70 | 94.3% | 95.4% |
| **0.85** | **85.6%** | **97.8%** |
| 0.95 | 75.9% | 99.3% |

A wrong auto-posted journal entry costs far more than a flagged one,
so the model routes low-confidence predictions to human review instead
of guessing. Observed behaviour:

```
AMZN MKTP US*2H4TY       -> None                  conf=0.522  needs_review=True
DELTA AIR LINES #4821    -> 6040_Travel_Airfare   conf=0.994  needs_review=False
WELLS FARGO SVC CHG      -> 6180_Bank_Fees        conf=0.998  needs_review=False
ZZQQ UNKNOWN VENDOR 99   -> None                  conf=0.133  needs_review=True
```

The genuinely ambiguous vendor and the unseen vendor both get flagged.
That's the system working, not failing.

### Field extraction, by input quality

Exact-match accuracy, degrading as OCR quality drops:

| clean | light | medium | heavy |
|---|---|---|---|
| **1.000** | 0.848 | 0.569 | 0.330 |

Clean at 1.000 confirms the patterns are correct, so the decline is
genuine OCR difficulty rather than latent bugs. Per-field results are
in `artifacts/extraction_eval.json` — reported per field on purpose,
since dates hold up far better than form-box amounts and those are
different engineering problems.

### Retrieval, over 38 hand-labelled queries

| mode | recall@3 | hit@1 | MRR | recall@3 (exact) | recall@3 (paraphrase) |
|---|---|---|---|---|---|
| BM25 | 0.895 | 0.789 | 0.833 | 1.000 | 0.818 |
| **dense** | **1.000** | **0.816** | **0.899** | 1.000 | **1.000** |
| hybrid | 0.947 | 0.816 | 0.873 | 1.000 | 0.909 |

**Hybrid lost, and that's worth explaining.** The usual argument for
hybrid search is that BM25 catches exact codes ("ASC 842") while dense
catches paraphrase. Here the fallback embedder is LSA over word *and
character* n-grams, so it already carries the lexical signal BM25
would contribute — fusing BM25 in adds noise rather than coverage.
With a true sentence-transformer embedder, which has no character-level
matching, the hybrid advantage is expected to return. All three modes
are kept and benchmarked rather than hard-coding a winner, because the
right choice depends on the embedder.

---

## Three engineering decisions worth discussing

### 1. I discarded the first benchmark

Document-type classification scored a perfect 1.00 macro-F1 — and held
1.00 even under heavy OCR degradation, terminology variation, and
header cropping. That is not an achievement; it means the task is
trivially separable by keyword ("Nonemployee compensation" appears in
exactly one document type).

Reporting it as a headline result would have been misleading. So the
classifier was demoted to a cheap routing stage, and the ML weight
moved to GL categorization, which has real class imbalance and
irreducible ambiguity. The saturated benchmark is still in the repo
(`src/classification/evaluate.py`) because the negative result is part
of the reasoning.

### 2. The out-of-scope guardrail failed, and the fix is two-stage

The first guardrail gated on the retriever's ranking score. That was a
bug: under reciprocal rank fusion the top result scores about the same
regardless of whether the query is on-topic, so *"What is the capital
of France?"* passed as grounded.

Replacing it with thresholds on raw BM25 and cosine signals fixed the
obvious cases. But validating against 13 out-of-scope probes exposed
the real problem:

```
Thresholds: BM25 >= 2.5, cosine >= 0.72
In-scope queries:     38   false refusals:  1 (2.6%)
Out-of-scope probes:  13   false accepts:   5 (38.5%)
    - what is the R&D tax credit calculation      bm25=3.13 cos=0.59
    - how do I compute depreciation under MACRS   bm25=2.16 cos=0.76
    - what are the estate tax exemption amounts   bm25=5.02 cos=0.66
    - explain FBAR filing requirements            bm25=2.56 cos=0.83
    - what is the QBI deduction phase-out         bm25=3.39 cos=0.80
```

Every false accept is a *tax* question the corpus doesn't cover. They
share vocabulary with the corpus, so **no similarity threshold can
separate them** — and they are the most dangerous queries in the
system, because a confabulated answer about the QBI phase-out will
look entirely plausible to a reader.

Hence stage two: an LLM sufficiency check that judges whether the
retrieved passages actually address the question before any answer is
generated. It costs one extra call per query. In a tax context that is
obviously worth it.

### 3. Two extraction bugs that produced *plausible wrong numbers*

Per-field evaluation caught three fields failing on clean input, which
had to be pattern bugs rather than OCR difficulty:

- `TOTAL` matched inside "Sub**total**" — and since the subtotal line
  comes first, the extractor confidently returned the subtotal as the
  invoice total.
- On Form 1099-NEC, "Nonemployee Compensation" appears in the page
  title before box 1, so the extractor returned the *tax year* as the
  compensation amount.
- Account balances go negative; the amount pattern rejected a leading
  minus and silently dropped ~20% of closing balances.

None of these threw an error. All three returned a well-formed number
of the right magnitude — the worst possible failure mode in an
accounting pipeline, and the reason the evaluation is per-field rather
than aggregate. Each is now locked down by a named regression test.

---

## Architecture

```
                   ┌─────────────────────────────┐
  scanned doc ───► │  /extract                   │
                   │  router → field extraction  │ ──► fields + coverage
                   └─────────────────────────────┘     + needs_review

                   ┌─────────────────────────────┐
  txn memo ──────► │  /categorize                │ ──► GL account + confidence
                   │  char+word TF-IDF → LogReg  │     + needs_review
                   └─────────────────────────────┘

                   ┌─────────────────────────────┐
  question ──────► │  /research                  │ ──► answer + citations
                   │  retrieve → score gate →    │     + grounded + guard_stage
                   │  sufficiency check → LLM    │
                   └─────────────────────────────┘
```

```
src/
  common/      synthetic data, OCR noise, intake variation, ledger generator
  classification/  document router + GL categorizer + evaluation
  extraction/  field extraction + per-field evaluation
  rag/         corpus, embedders, hybrid retriever, retrieval eval, generation
  api/         FastAPI service
scripts/       train_all, quality_gate, tune_threshold
tests/         43 tests
```

---

## Running it

```bash
pip install -r requirements.txt

# Generate data and train everything
python -m scripts.train_all

# Serve
uvicorn src.api.main:app --reload --port 8000
# docs at http://localhost:8000/docs
```

```bash
# Docker
docker build -t acctintel .
docker run -p 8000:8000 acctintel
```

### Enabling real models

The system runs fully offline with no API keys — the RAG layer falls
back to TF-IDF+SVD embeddings and returns retrieved passages with an
explicit "no model configured" notice rather than fabricating an
answer. To enable the production path:

```bash
pip install sentence-transformers openai

export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini"
```

The offline fallback is a deliberate engineering choice, not a
limitation: build agents in regulated environments often can't reach a
model hub, and a system that hard-fails there isn't deployable.

### Reproducing the results

```bash
python -m scripts.train_all              # trains, prints GL metrics
python -m src.rag.evaluate               # retrieval table
python -m src.extraction.evaluate        # per-field extraction table
python -m scripts.tune_threshold         # guardrail validation
python -m src.classification.evaluate    # the saturated router benchmark
python -m scripts.quality_gate           # CI gate
pytest tests/ -v                         # 43 tests
```

---

## Example requests

```bash
curl -X POST localhost:8000/categorize \
  -H 'Content-Type: application/json' \
  -d '{"memo": "UBER TRIP 4821 SEATTLE", "threshold": 0.85}'
```

This one returns `predicted_account: null` with `needs_review: true` —
confidence 0.824, just under the threshold, split 0.82 / 0.16 between
Travel_Ground and Meals_Entertainment. That is the intended behaviour,
and a useful check that the model learned the real distribution: Uber
charges genuinely are mostly rides and sometimes meals. `UBER EATS`
resolves to Meals_Entertainment at 0.999, and `LYFT` to Travel_Ground
at 0.999 — the abstention fires on the ambiguous memo specifically,
not on the vendor family.

```bash
curl -X POST localhost:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"question": "What are the limits on deducting business meals?"}'
```

```bash
curl -X POST localhost:8000/extract \
  -H 'Content-Type: application/json' \
  -d '{"text": "Invoice Number: INV-123\nTOTAL DUE: $1,250.00", "doc_type": "invoice"}'
```

---

## MLOps

`scripts/quality_gate.py` runs in CI and **fails the build** when a
model drops below an agreed floor:

```
GL macro-F1   0.8968  (floor 0.86)
GL accuracy   0.9273  (floor 0.90)
dense recall@3 1.0000 (floor 0.92)
dense hit@1    0.8158 (floor 0.75)
Quality gate passed.
```

Floors sit slightly below measured performance — tight enough to catch
a real regression, loose enough to absorb run-to-run variance. Metrics
that are only ever read in a notebook don't protect anything.

The CI pipeline (`.github/workflows/ci.yml`) runs tests, enforces the
quality gate, uploads metrics as build artifacts, builds the container,
and smoke-tests it against `/health`.

Models are persisted with their metrics JSON alongside, so the question
"how good is the model currently deployed, and on what data?" always
has an answer.

### Mapping to an Azure deployment

| This repo | Azure equivalent |
|---|---|
| `joblib` artifacts + metrics JSON | Azure ML model registry (versioned, with metrics) |
| `scripts/train_all.py` | Azure ML pipeline job |
| `scripts/quality_gate.py` | Gate before model registration / staged rollout |
| Dockerfile (non-root, healthcheck) | AKS deployment, or Azure ML managed endpoint |
| `AzureOpenAIClient` | Azure OpenAI deployment |
| `HybridRetriever` | Azure AI Search (hybrid + semantic ranker) |
| Env-var config | Key Vault + App Configuration |
| `data/*.jsonl` | Azure Data Lake / Synapse |

The container trains models at build time to stay self-contained. For
production, pull a versioned artifact from the registry at deploy time
instead — training and serving shouldn't share a lifecycle.

---

## Limitations

Stated plainly, because a portfolio project that hides these is less
useful than one that names them:

- **The knowledge base is not authoritative.** The 16 passages are
  plain-language summaries written for this project, not IRS or
  commercial research text. They demonstrate the RAG mechanics; they
  are not a filing position. A production deployment ingests the
  firm's own licensed library.
- **The retrieval corpus is small.** 16 passages and 38 queries make
  the retrieval numbers indicative, not decisive. With so few
  documents BM25's IDF estimates for common domain words are
  unreliable — which is exactly why "client dinner" ranks the
  audit-independence passage first. Those failing queries were kept in
  the gold set rather than deleted; removing queries a system fails on
  is how benchmarks become decoration.
- **Dense retrieval numbers come from the fallback embedder.**
  `sentence-transformers` wasn't installable in the build environment,
  so the reported figures use TF-IDF+SVD. Install it and re-run
  `python -m src.rag.evaluate` — the hybrid-vs-dense conclusion may
  well flip.
- **All data is synthetic.** Real accounting documents carry PII and
  client-confidential financials. The generators model realistic
  structure, ambiguity, and OCR failure modes, but no synthetic
  distribution fully matches production.
- **Extraction is rule-based.** Deliberate: for rigid fields, regex is
  faster, auditable, and doesn't hallucinate. Party names and
  free-text descriptions are where an LLM earns its place, and that
  fallback is a hook, not yet an implementation.
- **The sufficiency check is unvalidated at scale.** It resolves the
  38% false-accept rate by construction, but measuring its own
  precision and recall needs a labelled adjacent-topic set and an LLM
  provider — the natural next piece of work.
