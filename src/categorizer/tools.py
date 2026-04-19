"""Tool implementations exposed to the LLM tier.

The model sees four tools; each maps to a deterministic Python call.

  lookup_merchant(raw)           → canonical name + MCC hint + category hint
  query_user_history(merchant)   → what labels has the user used for similar tx
  get_recurring_tx_signal(...)   → is this a monthly recurring pattern
  ask_user_clarification(reason) → explicit escape hatch — do not guess

We expose the tool SCHEMAS here so llm.py can pass them as `tools=[...]`
to the chat completion. The RUNTIME dispatch (calling the Python function
when the model emits a tool_call) is driven by cascade.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from .merchant import MerchantMatch
from .merchant import lookup as merchant_lookup
from .storage import LabeledTransaction


@dataclass(frozen=True)
class ToolCallResult:
    name: str
    ok: bool
    payload: dict[str, Any]


# ── OpenAI-format tool schemas (passed to llama-server with Hermes parser) ──
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_merchant",
            "description": (
                "Resuelve un texto de transacción ruidoso a un comerciante canónico "
                "(Éxito, Rappi, Didi, Cinemark, Claude.ai, etc.) con un posible MCC "
                "y una categoría sugerida."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "normalized_text": {
                        "type": "string",
                        "description": "Descripción de la transacción ya normalizada.",
                    }
                },
                "required": ["normalized_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_user_history",
            "description": (
                "Devuelve las 10 etiquetas más recientes que el usuario ha usado para "
                "transacciones cuya descripción normalizada coincide o contiene la cadena dada."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "normalized_text": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 30, "default": 10},
                },
                "required": ["normalized_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recurring_tx_signal",
            "description": (
                "Indica si la transacción actual parece ser parte de un cargo mensual "
                "recurrente para el mismo comerciante (misma tienda, monto similar, "
                "cadencia ~30 días)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "normalized_text": {"type": "string"},
                    "amount": {"type": "number"},
                },
                "required": ["normalized_text", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user_clarification",
            "description": (
                "Indica que la transacción es genuinamente ambigua y debe enviarse al "
                "usuario para revisión manual. Usar sólo como último recurso cuando las "
                "otras herramientas no aportan señal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "maxLength": 200,
                        "description": "Por qué la transacción es ambigua.",
                    }
                },
                "required": ["reason"],
            },
        },
    },
]


# ── Runtime dispatch ────────────────────────────────────────────────────
async def dispatch(
    session: AsyncSession, name: str, arguments: dict[str, Any]
) -> ToolCallResult:
    """Invoke a tool by name. Each branch returns a ToolCallResult whose
    `payload` goes straight into the OpenAI-format tool-call response."""
    if name == "lookup_merchant":
        match: MerchantMatch | None = await merchant_lookup(
            session, arguments.get("normalized_text", "")
        )
        return ToolCallResult(
            name=name,
            ok=match is not None,
            payload=(
                {
                    "canonical_name": match.canonical_name,
                    "mcc_hint": match.mcc_hint,
                    "default_category_slug": match.default_category_slug,
                    "confidence": match.match_confidence,
                    "source": match.source,
                }
                if match is not None
                else {"match": None}
            ),
        )

    if name == "query_user_history":
        q = arguments.get("normalized_text", "")
        limit = int(arguments.get("limit", 10))
        if not q:
            return ToolCallResult(name=name, ok=True, payload={"history": []})
        stmt = (
            select(LabeledTransaction)
            .where(LabeledTransaction.normalized_description.ilike(f"%{q}%"))
            .order_by(desc(LabeledTransaction.tx_date))
            .limit(limit)
        )
        rows = (await session.scalars(stmt)).all()
        return ToolCallResult(
            name=name,
            ok=True,
            payload={
                "history": [
                    {
                        "category_slug": r.category_slug,
                        "normalized_description": r.normalized_description,
                        "amount": float(r.amount),
                        "date": r.tx_date.isoformat(),
                    }
                    for r in rows
                ]
            },
        )

    if name == "get_recurring_tx_signal":
        # Cheap DB-driven variant of features.detect_recurring. We don't
        # reuse that helper because it expects a history list from the caller
        # — here we go straight to the source.
        q = arguments.get("normalized_text", "")
        amount = float(arguments.get("amount", 0))
        if not q:
            return ToolCallResult(name=name, ok=True, payload={"is_recurring": False})
        stmt = (
            select(LabeledTransaction)
            .where(LabeledTransaction.normalized_description.ilike(f"%{q}%"))
            .order_by(desc(LabeledTransaction.tx_date))
            .limit(12)
        )
        rows = (await session.scalars(stmt)).all()
        amounts = [float(r.amount) for r in rows]
        matches = sum(1 for a in amounts if abs(a - amount) <= abs(amount) * 0.05)
        is_recurring = matches >= 3
        return ToolCallResult(
            name=name,
            ok=True,
            payload={
                "is_recurring": is_recurring,
                "match_count": matches,
                "median_amount": sorted(amounts)[len(amounts) // 2] if amounts else None,
            },
        )

    if name == "ask_user_clarification":
        return ToolCallResult(
            name=name,
            ok=True,
            payload={
                "escalated_to_user": True,
                "reason": arguments.get("reason", "ambiguous"),
            },
        )

    return ToolCallResult(name=name, ok=False, payload={"error": f"unknown tool: {name}"})
