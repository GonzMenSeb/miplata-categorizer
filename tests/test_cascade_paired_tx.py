"""Paired-tx tier tests.

Exercises the UUID → friendly-slug resolution and the paired-tx lookup
with a mocked AsyncSession. Real DB integration is covered live via the
deploy-time probe; these tests pin the logic contract.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from categorizer.cascade import _paired_internal_transfer, _resolve_account_slug
from categorizer.schemas import TransactionIn
from categorizer.storage import LabeledTransaction


def _tx(
    *,
    account_slug: str = "bancolombia_ahorros_0810",
    amount: str = "-50000",
    ttype: str = "debit",
    desc: str = "TRASLADO A CUENTA PROPIA",
) -> TransactionIn:
    return TransactionIn(
        external_id="probe-paired",
        account_slug=account_slug,
        tx_date=date(2026, 1, 15),
        amount=Decimal(amount),
        currency="COP",
        description=desc,
        original_description=desc,
        transaction_type=ttype,  # type: ignore[arg-type]
    )


class _FakeSession:
    """Minimal AsyncSession stand-in for cascade internals."""

    def __init__(self, *, uuid_resolves_to: str | None, paired_hit: object | None):
        self._uuid_resolves_to = uuid_resolves_to
        self._paired_hit = paired_hit
        self.scalar = AsyncMock(side_effect=self._scalar_side_effect)
        self._call_count = 0

    async def _scalar_side_effect(self, _stmt):
        # First call inside _paired_internal_transfer is the UUID resolver
        # (via _resolve_account_slug); the second is the paired-tx lookup.
        self._call_count += 1
        if self._call_count == 1:
            return self._uuid_resolves_to
        return self._paired_hit


@pytest.mark.asyncio
async def test_resolve_account_slug_passthrough_for_friendly_slug():
    session = AsyncMock()
    out = await _resolve_account_slug(session, "bancolombia_ahorros_0810")
    assert out == "bancolombia_ahorros_0810"
    session.scalar.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_account_slug_translates_known_uuid():
    session = AsyncMock()
    session.scalar = AsyncMock(return_value="bancolombia_ahorros_0810")
    out = await _resolve_account_slug(
        session, "11111111-2222-3333-4444-555555555555"
    )
    assert out == "bancolombia_ahorros_0810"
    session.scalar.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_account_slug_unknown_uuid_falls_through():
    session = AsyncMock()
    session.scalar = AsyncMock(return_value=None)
    uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    out = await _resolve_account_slug(session, uuid)
    assert out == uuid


@pytest.mark.asyncio
async def test_paired_internal_transfer_friendly_slug_no_match():
    session = _FakeSession(uuid_resolves_to="ignored", paired_hit=None)
    tx = _tx(account_slug="bancolombia_ahorros_0810")
    # No UUID, so only one scalar call (paired-tx lookup returns None).
    session._uuid_resolves_to = None
    result = await _paired_internal_transfer(session, tx)
    assert result is None


@pytest.mark.asyncio
async def test_paired_internal_transfer_uuid_resolves_and_finds_pair():
    pair = LabeledTransaction(
        external_id="other",
        account_slug="nequi",
        tx_date=date(2026, 1, 15),
        amount=Decimal("50000"),
        currency="COP",
        transaction_type="credit",
        description="RECARGA",
        normalized_description="recarga",
        category_slug="movimientos_internos.entre_bancos",
    )
    session = _FakeSession(
        uuid_resolves_to="bancolombia_ahorros_0810", paired_hit=pair
    )
    tx = _tx(account_slug="11111111-2222-3333-4444-555555555555")
    result = await _paired_internal_transfer(session, tx)
    assert result is not None
    slug, conf, reason = result
    assert slug == "movimientos_internos.entre_bancos"
    assert conf >= 0.9
    assert "nequi" in reason


@pytest.mark.asyncio
async def test_paired_internal_transfer_uuid_unresolved_no_crash():
    # UUID doesn't resolve → fall-through path treats input as opaque slug;
    # if the paired-tx lookup returns None we simply return None.
    session = _FakeSession(uuid_resolves_to=None, paired_hit=None)
    tx = _tx(account_slug="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    result = await _paired_internal_transfer(session, tx)
    assert result is None
