"""HTTP surface.

All business endpoints live under `/v1/`. Traefik enforces bearer auth on
`/v1/*`; the internal health probe uses `/healthz` unauthenticated (routed
only inside the Docker network).
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from .cascade import _resolve_account_slug
from .cascade import categorize as run_cascade
from .metrics import (
    categorize_latency_seconds,
    corrections_total,
    predictions_total,
)
from .metrics import (
    render as render_metrics,
)
from .retrieval import embed_for_storage
from .schemas import (
    CategorizeRequest,
    CategorizeResponse,
    LabelIn,
)
from .storage import (
    Correction,
    LabeledTransaction,
    OwnAccount,
    Prediction,
    get_session_factory,
)
from .taxonomy import Taxonomy

router = APIRouter()


async def _get_session() -> AsyncSession:
    factory = get_session_factory()
    async with factory() as session:
        yield session


def _get_taxonomy(request: Request) -> Taxonomy:
    tax = request.app.state.taxonomy
    if tax is None:
        raise HTTPException(503, "taxonomy not loaded")
    return tax


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/metrics")
async def metrics_endpoint() -> Response:
    body, ct = render_metrics()
    return Response(content=body, media_type=ct)


@router.post("/v1/categorize", response_model=CategorizeResponse)
async def categorize_endpoint(
    payload: CategorizeRequest,
    request: Request,
    session: AsyncSession = Depends(_get_session),  # noqa: B008 — FastAPI DI idiom; Depends is immutable
) -> CategorizeResponse:
    taxonomy: Taxonomy = request.app.state.taxonomy
    t0 = time.perf_counter()
    result = await run_cascade(session, payload.transaction, taxonomy)
    latency_ms = (time.perf_counter() - t0) * 1000
    categorize_latency_seconds.observe(latency_ms / 1000.0)

    predictions_total.labels(
        tier=result.source, status="ok" if result.category_slug != "sin_clasificar.pendiente" else "reject"
    ).inc()

    # Audit-log the prediction.
    session.add(
        Prediction(
            external_id=payload.transaction.external_id,
            account_slug=payload.transaction.account_slug,
            normalized_description=(result.tier_trace[0]["output"] if result.tier_trace else ""),
            predicted_slug=result.category_slug,
            confidence=result.confidence,
            source_tier=result.source,
            latency_ms=latency_ms,
            trace=result.tier_trace if payload.return_trace else None,
        )
    )
    await session.commit()

    if not payload.return_trace:
        result = result.model_copy(update={"tier_trace": []})
    return CategorizeResponse(result=result, latency_ms=round(latency_ms, 2))


@router.post("/v1/label")
async def label_endpoint(
    payload: LabelIn,
    session: AsyncSession = Depends(_get_session),  # noqa: B008 — FastAPI DI idiom; Depends is immutable
) -> dict[str, object]:
    """Ingest a user-confirmed label. Appends to labeled_transactions with an
    embedding, and records a `corrections` row if the user is overriding a
    prior prediction."""
    from .normalize import normalize

    normalized = normalize(payload.transaction.description)
    embedding = (await embed_for_storage([normalized]))[0]

    # Resolve miplata UUID → friendly slug so the FK on labeled_transactions
    # is satisfied (labeled_transactions.account_slug → own_accounts.slug).
    # No-op for already-friendly slugs.
    resolved_account_slug = await _resolve_account_slug(session, payload.transaction.account_slug)

    session.add(
        LabeledTransaction(
            external_id=payload.transaction.external_id,
            account_slug=resolved_account_slug,
            tx_date=payload.transaction.tx_date,
            amount=payload.transaction.amount,
            currency=payload.transaction.currency,
            transaction_type=payload.transaction.transaction_type,
            description=payload.transaction.description,
            normalized_description=normalized,
            category_slug=payload.category_slug,
            source="user",
            embedding=embedding,
        )
    )

    if payload.correction_of and payload.correction_of != payload.category_slug:
        session.add(
            Correction(
                external_id=payload.transaction.external_id,
                old_slug=payload.correction_of,
                new_slug=payload.category_slug,
            )
        )
        # Infer whether parent was already right — helps us weight severity.
        parent_was_correct = payload.correction_of.split(".")[0] == payload.category_slug.split(".")[0]
        corrections_total.labels(parent_was_correct=str(parent_was_correct).lower()).inc()

    await session.commit()
    return {"ok": True, "external_id": payload.transaction.external_id}


@router.get("/v1/uncertain")
async def list_uncertain(
    limit: int = 20,
    session: AsyncSession = Depends(_get_session),  # noqa: B008 — FastAPI DI idiom; Depends is immutable
) -> dict[str, list[dict[str, object]]]:
    """Return the N most recent predictions that rejected or came in below
    the LLM confidence threshold. Powers the active-learning review UI."""
    from sqlalchemy import and_, desc, select

    stmt = (
        select(Prediction)
        .where(
            and_(
                Prediction.predicted_slug.in_(
                    ["sin_clasificar.pendiente", "sin_clasificar.desconocido"]
                )
                | (Prediction.confidence < 0.70)
            )
        )
        .order_by(desc(Prediction.created_at))
        .limit(limit)
    )
    rows = (await session.scalars(stmt)).all()
    return {
        "items": [
            {
                "external_id": r.external_id,
                "predicted_slug": r.predicted_slug,
                "confidence": float(r.confidence),
                "source_tier": r.source_tier,
                "normalized_description": r.normalized_description,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]
    }


@router.get("/v1/taxonomy")
async def get_taxonomy(request: Request) -> dict[str, object]:
    """Dump the in-memory taxonomy for external consumers (e.g. miplata's
    correction UI). `emittable_slugs` is the leaves-only list the LLM may
    actually emit; `by_parent` groups leaves under their root for UI pickers.
    """
    tax = _get_taxonomy(request)
    by_parent: dict[str, dict[str, object]] = {}
    for root in tax.roots():
        children = [c.slug for c in tax.children_of(root.slug)] or [root.slug]
        by_parent[root.slug] = {"name": root.name, "children": children}
    return {
        "emittable_slugs": list(tax.emittable_slugs),
        "by_parent": by_parent,
    }


@router.get("/v1/own-accounts")
async def list_own_accounts(
    session: AsyncSession = Depends(_get_session),  # noqa: B008 — FastAPI DI idiom; Depends is immutable
) -> dict[str, list[dict]]:
    from sqlalchemy import select

    rows = (await session.scalars(select(OwnAccount))).all()
    return {
        "items": [
            {
                "slug": r.slug,
                "display_name": r.display_name,
                "institution": r.institution,
                "account_number_tail": r.account_number_tail,
                "aliases": r.aliases or [],
            }
            for r in rows
        ]
    }
