"""Lightweight feature extraction for each transaction.

Intentionally stateless. Calling code passes in recent tx for the user (for
recurring detection) rather than querying the DB here — keeps this module
pure and easy to test.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from statistics import median
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import TransactionIn


def amount_bucket(amount: Decimal) -> str:
    """Coarse log-ish buckets in COP. Tuned to Colombian tx scale: small snacks
    start around $2k, restaurants $15–60k, utilities $50–300k, rent $1–3M."""
    a = abs(amount)
    if a < Decimal("2000"):
        return "tiny"
    if a < Decimal("20000"):
        return "small"
    if a < Decimal("80000"):
        return "medium"
    if a < Decimal("300000"):
        return "large"
    if a < Decimal("1500000"):
        return "xlarge"
    return "huge"


def temporal_features(tx_date: date) -> dict[str, int | str]:
    return {
        "day_of_month": tx_date.day,
        "day_of_week": tx_date.weekday(),  # 0 = Monday
        "is_month_end": 1 if tx_date.day >= 28 else 0,
        "is_month_start": 1 if tx_date.day <= 3 else 0,
        "is_payday_window": 1 if tx_date.day in (15, 30, 31, 1, 2) else 0,
        "month": tx_date.month,
        "weekday_name": tx_date.strftime("%A").lower(),
    }


def detect_recurring(
    tx: TransactionIn,
    history: list[TransactionIn],
    amount_tolerance_pct: float = 0.05,
    day_tolerance: int = 4,
) -> dict[str, int | float | None]:
    """Is this transaction part of a recurring monthly charge?

    Signals:
      • Same merchant/normalized_description appeared at least 3 times
      • Amount within ±amount_tolerance_pct of the median
      • Day-of-month within ±day_tolerance of the median

    `history` is the user's own recent tx (typically last 6 months). Caller
    filters to matching-merchant candidates — this function doesn't do its
    own merchant match.
    """
    if len(history) < 2:
        return {"is_recurring": 0, "confidence": 0.0, "cadence_days": None}

    days_gaps: list[int] = []
    amounts: list[Decimal] = []
    prev: date | None = None
    for h in sorted(history, key=lambda t: t.tx_date):
        if prev is not None:
            days_gaps.append((h.tx_date - prev).days)
        amounts.append(h.amount)
        prev = h.tx_date

    # Cadence: ~30 ± 4 days → monthly recurring.
    med_gap = median(days_gaps) if days_gaps else None
    med_amount = median(amounts)
    amount_match = (
        abs(tx.amount - med_amount) <= abs(med_amount) * Decimal(str(amount_tolerance_pct))
    )
    cadence_match = med_gap is not None and 30 - day_tolerance <= med_gap <= 30 + day_tolerance

    confidence = 0.0
    if amount_match and cadence_match and len(history) >= 3:
        confidence = 0.92
    elif amount_match and cadence_match:
        confidence = 0.75
    elif cadence_match:
        confidence = 0.55

    return {
        "is_recurring": 1 if confidence >= 0.75 else 0,
        "confidence": round(confidence, 3),
        "cadence_days": float(med_gap) if med_gap is not None else None,
    }
