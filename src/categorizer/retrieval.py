"""Retrieval tier: kNN over the user's labeled history via pgvector.

Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 through
fastembed (ONNX-int8) for CPU efficiency. The model is cached under
`artifacts_dir/embedding_cache/` and loaded lazily the first time an
embedding is requested. (Originally intfloat/multilingual-e5-small — swapped
because fastembed 0.8 dropped that model from its registry. Same 384-dim
vector, so no pgvector migration was required.)

We don't maintain a separate FAISS index — pgvector's IVFFlat on the
`labeled_transactions.embedding` column is plenty at our scale (<100k rows).
Keeping everything in Postgres means backup/restore is a single `pg_dump`
and writes are atomic with the label-insertion transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

from sqlalchemy import func, text
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings

if TYPE_CHECKING:
    from fastembed import TextEmbedding


@dataclass(frozen=True)
class RetrievedNeighbor:
    external_id: str
    normalized_description: str
    category_slug: str
    similarity: float


class EmbeddingModel:
    """Thread-safe singleton around fastembed's TextEmbedding.

    Deferred import: fastembed has a heavy dependency chain (onnxruntime).
    We don't want to pay its import cost in unit tests that don't touch retrieval.
    """

    _instance: EmbeddingModel | None = None
    _lock = Lock()

    def __init__(self) -> None:
        settings = get_settings()
        from fastembed import TextEmbedding

        cache_dir = settings.artifacts_dir / "embedding_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._model: TextEmbedding = TextEmbedding(
            model_name=settings.embedding_model,
            cache_dir=str(cache_dir),
        )

    @classmethod
    def instance(cls) -> EmbeddingModel:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def embed(self, texts: list[str]) -> list[list[float]]:
        # e5-family models want a "query: " / "passage: " prefix; symmetric
        # "query: " on both sides works for classification-by-similarity.
        # Other multilingual encoders (MiniLM, mpnet) don't use that convention.
        if "e5" in get_settings().embedding_model.lower():
            texts = [f"query: {t}" for t in texts]
        return [list(v) for v in self._model.embed(texts)]


async def embed_for_storage(texts: list[str]) -> list[list[float]]:
    return EmbeddingModel.instance().embed(texts)


async def knn(
    session: AsyncSession,
    normalized_query: str,
    k: int = 8,
    account_slug: str | None = None,
) -> list[RetrievedNeighbor]:
    """Top-k nearest neighbors from labeled_transactions.

    Returns at most `k` neighbors sorted by descending cosine similarity.
    """
    if not normalized_query:
        return []

    vector = EmbeddingModel.instance().embed([normalized_query])[0]

    # pgvector's <=> is cosine DISTANCE (0 = identical). We convert to similarity.
    # Branch on account_slug to avoid `:name::type` casts, which SQLAlchemy's
    # text() parser treats as literal (not a bindparam) and forwards to Postgres
    # unbound, causing a syntax error.
    account_filter = "AND account_slug = :account_slug" if account_slug else ""
    sql = text(
        f"""
        SELECT external_id, normalized_description, category_slug,
               1 - (embedding <=> (:q)::vector) AS similarity
        FROM labeled_transactions
        WHERE embedding IS NOT NULL
          {account_filter}
        ORDER BY embedding <=> (:q)::vector
        LIMIT :k
        """
    )
    params: dict[str, object] = {"q": vector, "k": k}
    if account_slug:
        params["account_slug"] = account_slug
    rows = (await session.execute(sql, params)).all()

    return [
        RetrievedNeighbor(
            external_id=r.external_id,
            normalized_description=r.normalized_description,
            category_slug=r.category_slug,
            similarity=float(r.similarity),
        )
        for r in rows
    ]


def vote(neighbors: list[RetrievedNeighbor]) -> tuple[str, float, float] | None:
    """Distance-weighted majority vote over neighbors.

    Returns (winning_slug, top1_similarity, margin) or None if `neighbors` is empty.
    Margin = top-1 similarity minus top-1-different-class similarity. A wide
    margin means the kNN tier is confident even when top1 < 1.0.
    """
    if not neighbors:
        return None

    weights: dict[str, float] = {}
    for n in neighbors:
        weights[n.category_slug] = weights.get(n.category_slug, 0.0) + n.similarity

    winner = max(weights, key=weights.get)  # type: ignore[arg-type]
    top1 = neighbors[0].similarity
    margin = 0.0
    for n in neighbors[1:]:
        if n.category_slug != winner:
            margin = top1 - n.similarity
            break
    return winner, top1, margin


# Helper for the avg() aggregate above so SQLAlchemy type-checking stays happy.
__all__ = ["EmbeddingModel", "RetrievedNeighbor", "embed_for_storage", "func", "knn", "vote"]
