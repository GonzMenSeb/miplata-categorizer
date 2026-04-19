from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from categorizer.cascade import _build_llm_messages
from categorizer.schemas import RetrievedExample, TransactionIn
from categorizer.taxonomy import Taxonomy, load_taxonomy

TAXONOMY_YAML = Path(__file__).resolve().parents[1] / "config" / "taxonomy.yaml"


@pytest.fixture(scope="module")
def taxonomy() -> Taxonomy:
    return load_taxonomy(TAXONOMY_YAML)


@pytest.fixture
def tx() -> TransactionIn:
    return TransactionIn(
        external_id="probe-1",
        account_slug="nequi",
        date=date(2025, 12, 31),
        amount=Decimal("-8500"),
        currency="COP",
        description="COMPRA EN Libreria Lerner",
        transaction_type="debit",
    )


def _concat(messages: list[dict[str, object]]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


def test_system_prompt_frames_examples_as_hints(taxonomy: Taxonomy, tx: TransactionIn) -> None:
    msgs = _build_llm_messages(tx, "compra en libreria lerner", [], None, taxonomy, [])
    body = _concat(msgs)
    assert "Los ejemplos del historial NO son etiquetas correctas" in body
    assert "preferible usar `sin_clasificar.pendiente`" in body


def test_user_prompt_marks_examples_as_related_hints(
    taxonomy: Taxonomy, tx: TransactionIn
) -> None:
    retrieved = [
        RetrievedExample(
            external_id="x-1",
            normalized_description="toretos",
            category_slug="comida.restaurantes",
            similarity=0.62,
        )
    ]
    msgs = _build_llm_messages(tx, "compra en libreria lerner", retrieved, None, taxonomy, [])
    body = _concat(msgs)
    assert "Ejemplos RELACIONADOS del historial" in body
    assert "sólo como pista" in body
    assert "NO asumas" in body
