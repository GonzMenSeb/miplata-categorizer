from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TransactionType = Literal["debit", "credit"]
TierName = Literal["rules", "knn", "llm_notink", "llm_think", "reject"]
PredictionSource = Literal["rules", "knn", "llm_notink", "llm_think", "user"]


class OwnAccountRef(BaseModel):
    """Pointer to one of the user's own money containers, used by the
    internal-transfer rule to decide whether a recipient / counterparty is
    actually the same user."""

    model_config = ConfigDict(extra="ignore")

    slug: str               # e.g. "bancolombia_ahorros_0810"
    display_name: str       # e.g. "Bancolombia Ahorros 0810"
    institution: str        # e.g. "bancolombia" | "nequi"
    account_number_tail: str | None = None  # last 4 or full number; matched case-insensitively
    aliases: list[str] = Field(default_factory=list)


class TransactionIn(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    external_id: str = Field(..., description="Stable id from the source system (e.g. miplata tx uuid).")
    account_slug: str = Field(..., description="Which own-account this tx belongs to.")
    tx_date: date = Field(..., alias="date", description="Transaction date. Accepts `date` or `tx_date`.")
    amount: Decimal
    currency: str = "COP"
    description: str
    original_description: str | None = None
    transaction_type: TransactionType
    balance_after: Decimal | None = None
    metadata: dict = Field(default_factory=dict)


class RetrievedExample(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    external_id: str
    normalized_description: str
    category_slug: str
    similarity: float
    amount: Decimal | None = None
    tx_date: date | None = Field(default=None, alias="date")


class CategorizationResult(BaseModel):
    category_slug: str
    confidence: float
    source: PredictionSource
    reasoning: str | None = None
    retrieved_examples: list[RetrievedExample] = Field(default_factory=list)
    features: dict = Field(default_factory=dict)
    tier_trace: list[dict] = Field(default_factory=list)


class CategorizeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    transaction: TransactionIn
    allow_reject: bool = True
    return_trace: bool = False


class CategorizeResponse(BaseModel):
    result: CategorizationResult
    latency_ms: float


class LabelIn(BaseModel):
    """User-confirmed or user-corrected label. This is the highest-quality
    training signal — every call here updates the retrieval index + the
    per-user merchant-preference table."""

    model_config = ConfigDict(extra="ignore")

    transaction: TransactionIn
    category_slug: str
    confirmed_at: datetime | None = None
    correction_of: str | None = Field(
        None,
        description="If this is a correction of a prior prediction, the slug the system originally returned.",
    )
