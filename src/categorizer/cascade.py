"""Cascade orchestrator.

Order of tiers (each writes to tier_trace; later tiers see earlier evidence):

  1. normalize                 (deterministic)
  2. features                  (deterministic, inline)
  3. rules                     (deterministic; text-based internal-transfer + merchant patterns)
  4. paired-tx internal-transfer  (DB-driven; catches "Recarga desde" with no tail)
  5. merchant lookup           (seeded dictionary)
  6. kNN retrieval             (pgvector + fastembed)
  7. LLM /no_think with tools  (Qwen3-4B via llama-server, JSON-schema constrained)
  8. LLM /think (uncertainty branch)
  9. reject → sin_clasificar.pendiente  (self-hosted policy: never guess)

Each tier emits (slug, confidence, reasoning). We stop at the first tier
that meets its configured threshold.
"""

from __future__ import annotations

import json
import time
from datetime import timedelta

import structlog
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from . import rules
from .config import get_settings
from .llm import classify as llm_classify
from .merchant import lookup as merchant_lookup
from .normalize import normalize
from .retrieval import knn, vote
from .schemas import (
    CategorizationResult,
    OwnAccountRef,
    RetrievedExample,
    TransactionIn,
)
from .storage import LabeledTransaction, OwnAccount
from .taxonomy import Taxonomy
from .tools import TOOL_SCHEMAS
from .tools import dispatch as tool_dispatch

log = structlog.get_logger("cascade")


async def _load_own_accounts(session: AsyncSession) -> list[OwnAccountRef]:
    rows = (await session.scalars(select(OwnAccount))).all()
    return [
        OwnAccountRef(
            slug=r.slug,
            display_name=r.display_name,
            institution=r.institution,
            account_number_tail=r.account_number_tail,
            aliases=list(r.aliases or []),
        )
        for r in rows
    ]


async def _paired_internal_transfer(
    session: AsyncSession, tx: TransactionIn
) -> tuple[str, float, str] | None:
    """Detect internal transfer via paired tx on another own_account.

    Looks for a labeled transaction on a DIFFERENT own_account with:
      • opposite transaction_type
      • same absolute amount
      • within ±2 days of the current tx date

    Carcass variant: currently only matches against already-labeled
    transactions. In a future version we also query miplata_ro_database_url
    for unlabeled candidate pairs — marked TODO below.
    """
    opposite_type = "credit" if tx.transaction_type == "debit" else "debit"
    stmt = select(LabeledTransaction).where(
        and_(
            LabeledTransaction.account_slug != tx.account_slug,
            LabeledTransaction.transaction_type == opposite_type,
            LabeledTransaction.amount == abs(tx.amount),
            LabeledTransaction.tx_date >= tx.tx_date - timedelta(days=2),
            LabeledTransaction.tx_date <= tx.tx_date + timedelta(days=2),
        )
    )
    hit = await session.scalar(stmt)
    if hit is None:
        # TODO(carcass): extend to miplata_ro DB once we have that client.
        return None
    return (
        "movimientos_internos.entre_bancos",
        0.93,
        f"Paired tx found on own_account '{hit.account_slug}' "
        f"(same amount, opposite direction, within ±2 days).",
    )


def _build_llm_messages(
    tx: TransactionIn,
    normalized: str,
    retrieved: list[RetrievedExample],
    merchant_hint: str | None,
    taxonomy: Taxonomy,
    own_accounts: list[OwnAccountRef],
) -> list[dict[str, object]]:
    system = (
        "Eres un categorizador de transacciones bancarias colombianas para una app "
        "de finanzas personales. Elige EXACTAMENTE un slug del enum provisto. "
        "Privilegia las categorías en las que los ejemplos recuperados coinciden. "
        "Si la transacción parece moverse entre cuentas propias del usuario (listadas abajo), "
        "usa la rama `movimientos_internos`. Si la transacción es genuinamente ambigua "
        "y ninguna herramienta ayuda, usa `sin_clasificar.pendiente` — NO adivines."
    )

    own_accts_block = "\n".join(
        f"- {a.display_name} (slug={a.slug}, institution={a.institution}, "
        f"aliases={a.aliases})"
        for a in own_accounts
    ) or "(ninguna cuenta propia registrada aún)"

    examples_block = (
        "\n".join(
            f"- [{e.similarity:.2f}] {e.normalized_description!r} → {e.category_slug}"
            for e in retrieved[:8]
        )
        or "(sin ejemplos previos del usuario)"
    )

    taxonomy_block = "\n".join(
        f"- {c.slug}: {c.name}" for c in sorted(taxonomy.children_of(None), key=lambda x: x.slug)
    )

    user = (
        f"Cuentas propias del usuario:\n{own_accts_block}\n\n"
        f"Taxonomía (niveles raíz):\n{taxonomy_block}\n\n"
        f"Transacción:\n"
        f"  descripción original: {tx.description!r}\n"
        f"  descripción normalizada: {normalized!r}\n"
        f"  monto: {tx.amount} {tx.currency}\n"
        f"  tipo: {tx.transaction_type}\n"
        f"  fecha: {tx.tx_date.isoformat()}\n"
        f"  cuenta: {tx.account_slug}\n\n"
        f"Pista de comerciante (si se resolvió): {merchant_hint or 'ninguna'}\n\n"
        f"Ejemplos del historial del usuario:\n{examples_block}\n\n"
        "Responde SÓLO con el JSON que cumple el schema."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


async def categorize(
    session: AsyncSession,
    tx: TransactionIn,
    taxonomy: Taxonomy,
) -> CategorizationResult:
    settings = get_settings()
    trace: list[dict] = []

    normalized = normalize(tx.description)
    trace.append({"tier": "normalize", "output": normalized})

    own_accounts = await _load_own_accounts(session)

    # ── Tier 3: deterministic rules (text-based) ────────────────────────
    rule_hit = rules.match(tx, normalized, own_accounts)
    if rule_hit is not None and rule_hit.confidence >= settings.rule_min_confidence:
        trace.append(
            {"tier": "rules", "slug": rule_hit.category_slug, "confidence": rule_hit.confidence}
        )
        return CategorizationResult(
            category_slug=rule_hit.category_slug,
            confidence=rule_hit.confidence,
            source="rules",
            reasoning=rule_hit.reasoning,
            tier_trace=trace,
        )

    # ── Tier 4: paired-tx internal transfer ─────────────────────────────
    paired = await _paired_internal_transfer(session, tx)
    if paired is not None:
        slug, conf, reason = paired
        if conf >= settings.rule_min_confidence - 0.05:   # slightly more lenient: DB evidence
            trace.append({"tier": "paired_internal", "slug": slug, "confidence": conf})
            return CategorizationResult(
                category_slug=slug, confidence=conf, source="rules",
                reasoning=reason, tier_trace=trace,
            )

    # ── Tier 5: merchant hint (soft — not itself a prediction) ──────────
    merchant_match = await merchant_lookup(session, normalized)
    merchant_hint = (
        f"{merchant_match.canonical_name} (MCC {merchant_match.mcc_hint}, "
        f"sugerida {merchant_match.default_category_slug}, conf {merchant_match.match_confidence})"
        if merchant_match is not None
        else None
    )
    if merchant_match is not None and merchant_match.default_category_slug in taxonomy:
        trace.append(
            {
                "tier": "merchant",
                "canonical_name": merchant_match.canonical_name,
                "suggested": merchant_match.default_category_slug,
                "confidence": merchant_match.match_confidence,
            }
        )
        # High-confidence seed matches are emitted directly.
        if (
            merchant_match.source == "seed"
            and merchant_match.match_confidence >= settings.knn_min_confidence
        ):
            return CategorizationResult(
                category_slug=merchant_match.default_category_slug,   # type: ignore[arg-type]
                confidence=merchant_match.match_confidence,
                source="rules",
                reasoning=f"Coincidencia en diccionario de comerciantes: {merchant_match.canonical_name}.",
                tier_trace=trace,
            )

    # ── Tier 6: kNN retrieval (carcass-safe: returns no prediction if index empty) ─
    retrieved_raw: list[RetrievedExample] = []
    try:
        neighbors = await knn(session, normalized, k=8)
        retrieved_raw = [
            RetrievedExample(
                external_id=n.external_id,
                normalized_description=n.normalized_description,
                category_slug=n.category_slug,
                similarity=n.similarity,
            )
            for n in neighbors
        ]
    except Exception as exc:  # pragma: no cover — retrieval is best-effort
        log.warning("knn_failed", error=str(exc))
        neighbors = []

    if neighbors:
        voted = vote(neighbors)
        if voted is not None:
            winner, top1, margin = voted
            trace.append(
                {
                    "tier": "knn",
                    "top1": top1,
                    "margin": margin,
                    "winner": winner,
                    "n_neighbors": len(neighbors),
                }
            )
            if (
                top1 >= settings.knn_min_confidence
                and margin >= settings.knn_min_margin
                and winner in taxonomy
            ):
                return CategorizationResult(
                    category_slug=winner,
                    confidence=min(0.99, top1),
                    source="knn",
                    reasoning=f"kNN: top-1 sim={top1:.3f}, margen={margin:.3f}.",
                    retrieved_examples=retrieved_raw,
                    tier_trace=trace,
                )

    # ── Tier 7: LLM no-think with tools ────────────────────────────────
    messages = _build_llm_messages(
        tx, normalized, retrieved_raw, merchant_hint, taxonomy, own_accounts
    )
    allowed = list(taxonomy.implemented_slugs)
    llm_resp = await llm_classify(
        messages=messages, allowed_slugs=allowed, tools=TOOL_SCHEMAS, thinking=False
    )
    trace.append(
        {
            "tier": "llm_nothink",
            "elapsed_ms": llm_resp.elapsed_ms,
            "tool_calls": [tc["name"] for tc in llm_resp.tool_calls],
            "finish_reason": llm_resp.finish_reason,
        }
    )

    # Tool-call loop. Single round for v1 — Qwen3-4B at this ISA tier
    # deteriorates past one tool round on this hardware.
    if llm_resp.tool_calls:
        tool_outputs: list[dict[str, object]] = []
        for tc in llm_resp.tool_calls:
            try:
                args = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            res = await tool_dispatch(session, tc["name"], args)
            tool_outputs.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(res.payload, ensure_ascii=False),
                }
            )
        follow = await llm_classify(
            messages=[*messages, *tool_outputs],
            allowed_slugs=allowed,
            tools=None,    # no more tool calls in the second round
            thinking=False,
        )
        llm_resp = follow
        trace.append({"tier": "llm_nothink_followup", "elapsed_ms": llm_resp.elapsed_ms})

    parsed = llm_resp.parsed or {}
    slug = parsed.get("category_slug")
    conf = float(parsed.get("confidence", 0.0))
    reasoning = str(parsed.get("reasoning", "") or "")

    if (
        slug
        and slug in taxonomy
        and conf >= settings.llm_min_confidence
    ):
        return CategorizationResult(
            category_slug=slug,
            confidence=conf,
            source="llm_notink",
            reasoning=reasoning,
            retrieved_examples=retrieved_raw,
            tier_trace=trace,
        )

    # ── Tier 8: LLM /think (uncertainty branch) ────────────────────────
    if conf < settings.think_trigger_confidence:
        think_resp = await llm_classify(
            messages=messages, allowed_slugs=allowed, tools=None, thinking=True,
        )
        trace.append({"tier": "llm_think", "elapsed_ms": think_resp.elapsed_ms})
        tparsed = think_resp.parsed or {}
        tslug = tparsed.get("category_slug")
        tconf = float(tparsed.get("confidence", 0.0))
        treas = str(tparsed.get("reasoning", "") or "")
        if tslug and tslug in taxonomy and tconf >= settings.llm_min_confidence:
            return CategorizationResult(
                category_slug=tslug,
                confidence=tconf,
                source="llm_think",
                reasoning=treas,
                retrieved_examples=retrieved_raw,
                tier_trace=trace,
            )

    # ── Tier 9: reject ──────────────────────────────────────────────────
    trace.append({"tier": "reject", "reason": "no tier met its confidence threshold"})
    return CategorizationResult(
        category_slug="sin_clasificar.pendiente",
        confidence=conf if conf > 0 else 0.0,
        source="llm_notink",
        reasoning="Ninguna tier alcanzó el umbral de confianza; la transacción queda pendiente de revisión.",
        retrieved_examples=retrieved_raw,
        tier_trace=trace,
    )


def total_latency_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000
