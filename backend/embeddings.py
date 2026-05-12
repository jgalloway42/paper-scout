"""SentenceTransformer: encode, deduplicate, classify topics."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.ingestors.base import RawItem

_model: SentenceTransformer | None = None
_taxonomy_embeddings: np.ndarray | None = None
_taxonomy_labels: list[str] | None = None


def load_model() -> SentenceTransformer:
    """Load all-MiniLM-L6-v2 once. Module-level cache."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def encode(texts: list[str]) -> np.ndarray:
    """Return (N, 384) float32. Input is a list of pre-built strings."""
    model = load_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.astype(np.float32)


def encode_items(items: list[RawItem]) -> np.ndarray:
    """Encode title + first 200 chars of abstract for each item."""
    texts = [f"{it.title} {it.abstract[:200]}" for it in items]
    return encode(texts)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of a (N, D) and rows of b (M, D).

    Returns (N, M) matrix.
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def deduplicate(
    candidates: list[RawItem],
    existing_vectors: np.ndarray,
    threshold: float,
) -> list[RawItem]:
    """Remove candidates too similar to existing DB vectors or to each other.

    existing_vectors: (M, 384) from repository.get_all_embeddings().
    threshold: cosine similarity cutoff (items above this are dropped).
    Returns filtered list of RawItem.
    """
    if not candidates:
        return []

    cand_vecs = encode_items(candidates)

    # Dedup within the batch: keep first occurrence of near-duplicates
    kept_indices: list[int] = []
    kept_vecs: list[np.ndarray] = []
    for i, vec in enumerate(cand_vecs):
        if kept_vecs:
            sims = _cosine_sim(vec.reshape(1, -1), np.stack(kept_vecs))[0]
            if sims.max() > threshold:
                continue
        kept_indices.append(i)
        kept_vecs.append(vec)

    if not kept_indices:
        return []

    filtered_candidates = [candidates[i] for i in kept_indices]
    filtered_vecs = np.stack([cand_vecs[i] for i in kept_indices])

    # Dedup against existing DB vectors
    if existing_vectors.shape[0] > 0:
        sims = _cosine_sim(filtered_vecs, existing_vectors)
        max_sims = sims.max(axis=1)
        mask = max_sims <= threshold
        filtered_candidates = [c for c, keep in zip(filtered_candidates, mask) if keep]

    return filtered_candidates


def classify_topic(text: str, taxonomy: list[str]) -> str:
    """Return closest taxonomy label by cosine similarity.

    Caches taxonomy label embeddings on first call.
    """
    global _taxonomy_embeddings, _taxonomy_labels

    if _taxonomy_embeddings is None or _taxonomy_labels != taxonomy:
        _taxonomy_labels = list(taxonomy)
        _taxonomy_embeddings = encode(_taxonomy_labels)

    text_vec = encode([text])
    sims = _cosine_sim(text_vec, _taxonomy_embeddings)[0]
    return taxonomy[int(sims.argmax())]
