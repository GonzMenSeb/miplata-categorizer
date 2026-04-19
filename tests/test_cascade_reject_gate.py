"""Reject-gate postcondition on the LLM tiers.

The cascade must NEVER accept an LLM-proposed ``movimientos_internos.*``
label without corroborating evidence — either a paired-tx hit (tier 4) or
an own_account textual match against the description. This pins the
behavior regressed by gold-0044 (name + round amount → LLM hallucinated
``pago_tarjeta_propia`` with 0.95 confidence on a plain Nequi debit).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from categorizer import cascade
from categorizer.schemas import OwnAccountRef, TransactionIn
from categorizer.taxonomy import Category, Taxonomy


def _tx(desc: str = "SEBASTIAN MENDOZA", amount: str = "-200000") -> TransactionIn:
    return TransactionIn(
        external_id="probe-reject",
        account_slug="nequi",
        tx_date=date(2025, 6, 17),
        amount=Decimal(amount),
        currency="COP",
        description=desc,
        original_description=desc,
        transaction_type="debit",
    )


def _taxonomy() -> Taxonomy:
    cats = [
        Category("comida", "Comida", None, True),
        Category("comida.domicilios", "Domicilios", "comida", True),
        Category("movimientos_internos", "Mov. internos", None, True),
        Category("movimientos_internos.entre_bancos", "Entre bancos", "movimientos_internos", True),
        Category("movimientos_internos.pago_tarjeta_propia", "Pago TC propia", "movimientos_internos", True),
        Category("sin_clasificar", "Sin clasificar", None, True),
        Category("sin_clasificar.pendiente", "Pendiente", "sin_clasificar", True),
    ]
    return Taxonomy(cats)


def _own_accounts_nequi_only() -> list[OwnAccountRef]:
    return [
        OwnAccountRef(
            slug="nequi",
            display_name="Nequi",
            institution="nequi",
            account_number_tail=None,
            aliases=["nequi"],
        )
    ]


class _Resp:
    def __init__(self, slug: str | None, conf: float, reasoning: str = "r"):
        self.parsed = (
            {"category_slug": slug, "confidence": conf, "reasoning": reasoning}
            if slug is not None
            else None
        )
        self.tool_calls: list[dict] = []
        self.elapsed_ms = 12.0
        self.finish_reason = "stop"


@pytest.fixture(autouse=True)
def _patch_session_deps():
    """Stub out DB I/O and merchant lookup used by cascade.categorize."""
    with (
        patch.object(cascade, "_load_own_accounts", new=AsyncMock(return_value=_own_accounts_nequi_only())),
        patch.object(cascade, "merchant_lookup", new=AsyncMock(return_value=None)),
        patch.object(cascade, "knn", new=AsyncMock(return_value=[])),
    ):
        yield


@pytest.mark.asyncio
async def test_llm_proposes_internal_transfer_without_own_account_match_is_rejected():
    """gold-0044 regression: `SEBASTIAN MENDOZA` is not an own_account alias."""
    tx = _tx(desc="SEBASTIAN MENDOZA")
    session = AsyncMock()
    with (
        patch.object(cascade, "_paired_internal_transfer", new=AsyncMock(return_value=None)),
        patch.object(
            cascade,
            "llm_classify",
            new=AsyncMock(
                side_effect=[
                    _Resp("movimientos_internos.pago_tarjeta_propia", 0.95),
                    _Resp("movimientos_internos.pago_tarjeta_propia", 0.95),
                ]
            ),
        ),
    ):
        result = await cascade.categorize(session, tx, _taxonomy())
    assert result.category_slug == "sin_clasificar.pendiente"
    assert result.source in {"llm_think", "llm_notink"}
    gate_trace = [t for t in result.tier_trace if "reject_gate" in t.get("tier", "")]
    assert gate_trace, "reject-gate trace entry should be emitted"


@pytest.mark.asyncio
async def test_llm_internal_transfer_with_alias_in_description_passes():
    """`nequi` appears in the description (not in an internal-transfer phrase
    that the rule tier would catch) → own_account text match, LLM accepted."""
    tx = _tx(desc="ABONO NEQUI WALLET")
    session = AsyncMock()
    with (
        patch.object(cascade, "_paired_internal_transfer", new=AsyncMock(return_value=None)),
        patch.object(
            cascade,
            "llm_classify",
            new=AsyncMock(return_value=_Resp("movimientos_internos.entre_bancos", 0.90)),
        ),
    ):
        result = await cascade.categorize(session, tx, _taxonomy())
    assert result.category_slug == "movimientos_internos.entre_bancos"
    assert result.source == "llm_notink"


@pytest.mark.asyncio
async def test_paired_tx_hit_wins_over_llm():
    """Paired-tx tier (4) fires before the LLM ever runs; reject-gate inactive."""
    tx = _tx(desc="SEBASTIAN MENDOZA")
    session = AsyncMock()
    paired = (
        "movimientos_internos.entre_bancos",
        0.93,
        "Paired tx found on own_account 'bancolombia' (same amount, opposite direction).",
    )
    llm_mock = AsyncMock(return_value=_Resp("comida.domicilios", 0.99))
    with (
        patch.object(cascade, "_paired_internal_transfer", new=AsyncMock(return_value=paired)),
        patch.object(cascade, "llm_classify", new=llm_mock),
    ):
        result = await cascade.categorize(session, tx, _taxonomy())
    assert result.category_slug == "movimientos_internos.entre_bancos"
    assert result.source == "rules"
    llm_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_internal_transfer_llm_prediction_passes_through():
    """The reject-gate only guards `movimientos_internos.*`; others flow freely."""
    tx = _tx(desc="SEBASTIAN MENDOZA")
    session = AsyncMock()
    with (
        patch.object(cascade, "_paired_internal_transfer", new=AsyncMock(return_value=None)),
        patch.object(
            cascade,
            "llm_classify",
            new=AsyncMock(return_value=_Resp("comida.domicilios", 0.90)),
        ),
    ):
        result = await cascade.categorize(session, tx, _taxonomy())
    assert result.category_slug == "comida.domicilios"
    assert result.source == "llm_notink"


def test_own_account_text_match_matches_alias_in_normalized():
    assert cascade._own_account_text_match(
        normalized="transferencia_desde nequi",
        raw="TRANSFERENCIA DESDE NEQUI",
        own_accounts=_own_accounts_nequi_only(),
    )


def test_own_account_text_match_no_alias_in_plain_name():
    assert not cascade._own_account_text_match(
        normalized="sebastian mendoza",
        raw="SEBASTIAN MENDOZA",
        own_accounts=_own_accounts_nequi_only(),
    )
