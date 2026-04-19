"""SQLAlchemy 2.0 async models for the categorizer's own Postgres.

Design principle: the categorizer owns its data end-to-end. It ingests a
transaction, stores the raw + normalized versions, emits a prediction, and
appends every user correction as a new row (append-only). The retrieval
index lives in the `labeled_transactions.embedding` column (pgvector).

Tables:

  own_accounts           — the user's money containers (Bancolombia savings,
                            Nequi, credit cards). Populated via /v1/accounts.
  labeled_transactions   — gold-confirmed labels + their embeddings.
                            Retrieval tier queries this table only.
  predictions            — one row per categorization call (audit trail).
  corrections            — user corrections, pointing at the original prediction.
  merchants              — seeded + learned canonical-merchant mapping.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Date, DateTime, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .config import get_settings


class Base(DeclarativeBase):
    pass


def _utc_now() -> datetime:
    return datetime.now(UTC)


class OwnAccount(Base):
    __tablename__ = "own_accounts"

    slug: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    institution: Mapped[str] = mapped_column(String(64), nullable=False)
    account_number_tail: Mapped[str | None] = mapped_column(String(32))
    aliases: Mapped[list[str] | None] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class LabeledTransaction(Base):
    """The gold/user-confirmed corpus. This is the retrieval-tier data."""

    __tablename__ = "labeled_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    account_slug: Mapped[str] = mapped_column(String(64), ForeignKey("own_accounts.slug"), nullable=False)
    tx_date: Mapped[date] = mapped_column(Date, nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="COP")
    transaction_type: Mapped[str] = mapped_column(String(8), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_description: Mapped[str] = mapped_column(Text, nullable=False)
    category_slug: Mapped[str] = mapped_column(String(96), nullable=False)
    source: Mapped[str] = mapped_column(String(16), default="user")  # user | seed
    embedding: Mapped[list[float] | None] = mapped_column(
        # Dimension comes from config; default 384 for paraphrase-multilingual-MiniLM-L12-v2.
        Vector(get_settings().embedding_dim)
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    __table_args__ = (
        # IVFFlat over cosine. For <50k rows on CPU, flat would also be fine;
        # IVFFlat scales better as the corpus grows and the overhead at small
        # sizes is negligible.
        Index(
            "ix_labeled_transactions_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_labeled_transactions_category", "category_slug"),
        Index("ix_labeled_transactions_account", "account_slug"),
    )


class Prediction(Base):
    """Audit trail: one row per /v1/categorize call."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(128), nullable=False)
    account_slug: Mapped[str] = mapped_column(String(64), nullable=False)
    normalized_description: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_slug: Mapped[str] = mapped_column(String(96), nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    source_tier: Mapped[str] = mapped_column(String(16), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    trace: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    __table_args__ = (
        Index("ix_predictions_external_id", "external_id"),
        Index("ix_predictions_created_at", "created_at"),
    )


class Correction(Base):
    """User-originated correction of a prediction. Append-only."""

    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(128), nullable=False)
    old_slug: Mapped[str | None] = mapped_column(String(96))
    new_slug: Mapped[str] = mapped_column(String(96), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    __table_args__ = (Index("ix_corrections_external_id", "external_id"),)


class Merchant(Base):
    """Canonical-merchant mapping used by the lookup_merchant tool.

    Rows are a mix of:
      • Hand-seeded Colombian chains (Éxito, Carulla, Rappi, Didi, ...).
      • User-learned: if a user consistently labels tx matching "BOLD*Macchiato"
        as "comida.cafe_panaderia", the resolver eventually promotes
        "macchiato caffe" → canonical merchant "Macchiato Caffè" with that
        category hint.
    """

    __tablename__ = "merchants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    aliases: Mapped[list[str] | None] = mapped_column(JSONB, default=list)
    mcc_hint: Mapped[str | None] = mapped_column(String(8))
    default_category_slug: Mapped[str | None] = mapped_column(String(96))
    source: Mapped[str] = mapped_column(String(16), default="seed")  # seed | learned
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


# ── Engine / session factory ─────────────────────────────────────────────
_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            str(get_settings().database_url),
            pool_pre_ping=True,
            pool_size=4,
            max_overflow=2,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _session_factory
