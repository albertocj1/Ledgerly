# Multi-stage build keeps the runtime image free of build tooling.
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim

# Run as a non-root user. Required by most hardened Kubernetes
# policies (including AKS with Pod Security Standards enforced).
RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /install /usr/local

WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Train models at build time so the image is self-contained and
# startup is fast. For larger models you'd instead pull a versioned
# artifact from a registry (Azure ML model registry / blob storage)
# at deploy time - noted in the README as the production path.
RUN python -m src.common.synth \
    && python -m src.common.ledger \
    && python -m src.classification.gl_model \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
