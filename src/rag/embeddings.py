"""
Embedding backends.

Two implementations behind one interface:

* `SentenceTransformerEmbedder` - the production path. Uses
  all-MiniLM-L6-v2 (or any model you pass) for real semantic
  embeddings. Requires downloading model weights.

* `SvdEmbedder` - a dependency-light fallback: TF-IDF over character
  and word n-grams reduced with truncated SVD (i.e. LSA). No model
  download, no GPU, runs anywhere. Weaker on paraphrase, but it keeps
  the pipeline runnable in locked-down CI and air-gapped environments.

The factory picks the transformer when it's importable and the weights
resolve, otherwise it falls back. That fallback is a deliberate
engineering choice, not a limitation: an accounting firm's build
agents often can't reach a model hub, and a system that hard-fails
there isn't deployable.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> "Embedder": ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class SvdEmbedder:
    """TF-IDF + truncated SVD (LSA) embeddings. No downloads required."""

    name = "tfidf-svd"

    def __init__(self, n_components: int = 128, random_state: int = 42):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        self._vec = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)),
            ]
        )
        self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._fitted = False

    def fit(self, texts: list[str]) -> "SvdEmbedder":
        X = self._vec.fit_transform(texts)
        n_comp = min(self._svd.n_components, X.shape[1] - 1, max(2, len(texts) - 1))
        self._svd.n_components = n_comp
        self._svd.fit(X)
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SvdEmbedder.fit() must be called before encode()")
        X = self._vec.transform(texts)
        return _l2_normalize(self._svd.transform(X).astype(np.float32))


class SentenceTransformerEmbedder:
    """Real semantic embeddings via sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        return self  # pretrained; nothing to fit

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(vecs.astype(np.float32))


def get_embedder(prefer_transformer: bool = True, model_name: str | None = None) -> Embedder:
    """Return the best available embedder, falling back cleanly."""
    if prefer_transformer:
        try:
            return SentenceTransformerEmbedder(
                model_name or "sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as exc:  # noqa: BLE001 - fallback is intentional
            import logging

            logging.getLogger(__name__).warning(
                "sentence-transformers unavailable (%s); falling back to TF-IDF+SVD embeddings.",
                exc.__class__.__name__,
            )
    return SvdEmbedder()
