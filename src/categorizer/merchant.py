"""Merchant resolver.

Three tiers:

  1. **Seeded dictionary** — hand-curated Colombian chains (Éxito, Carulla,
     Rappi, Didi variants, Nequi, etc.). Populated by a startup task (carcass)
     and augmented over time.

  2. **Learned aliases** — when users consistently correct a merchant to a
     canonical name, the DB row grows. Not fully implemented — the carcass
     exposes the interface.

  3. **Fuzzy match** (carcass) — RapidFuzz against the dictionary for novel
     merchant strings that don't literally match a seeded alias.

For v1 we rely on the seeded dict + exact-alias lookup. The carcass pieces
are clearly marked with TODO — they're real future work, not cruft.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .storage import Merchant


@dataclass(frozen=True)
class MerchantMatch:
    canonical_name: str
    mcc_hint: str | None
    default_category_slug: str | None
    match_confidence: float
    source: str  # "seed" | "learned" | "fuzzy"


# ── Seeded dictionary ────────────────────────────────────────────────────
# Minimal hand-curated seed. Kept deliberately small — the DB is authoritative
# after first migration; this dict is only for bootstrap / tests.
_SEED: tuple[dict[str, object], ...] = (
    {"canonical_name": "Rappi",
     "aliases": ["rappi", "rappi colombia", "rappi sas"],
     "mcc_hint": "5812",
     "default_category_slug": "comida.domicilios"},
    {"canonical_name": "Didi",
     "aliases": ["didi", "dlo didi", "dl didi rides co", "didi co ride", "didi rides"],
     "mcc_hint": "4121",
     "default_category_slug": "transporte.taxi_rideshare"},
    {"canonical_name": "Didi Food",
     "aliases": ["didi food", "dlo didi food", "dlo didi food co payin"],
     "mcc_hint": "5812",
     "default_category_slug": "comida.domicilios"},
    {"canonical_name": "Éxito",
     "aliases": ["exito", "pv dogger exito poblad", "exito poblad"],
     "mcc_hint": "5411",
     "default_category_slug": "comida.mercado"},
    {"canonical_name": "Tienda D1",
     "aliases": ["tienda d1", "tienda d1 automotriz", "d1"],
     "mcc_hint": "5411",
     "default_category_slug": "comida.mercado"},
    {"canonical_name": "Cinemark",
     "aliases": ["cinemark"],
     "mcc_hint": "7832",
     "default_category_slug": "ocio.cine_teatro"},
    {"canonical_name": "EPM",
     "aliases": ["epm", "pago factura epm", "empresas publicas"],
     "mcc_hint": "4900",
     "default_category_slug": "hogar.servicios_publicos"},
    {"canonical_name": "Compensar",
     "aliases": ["compensar"],
     "mcc_hint": "8099",
     "default_category_slug": "salud.eps_seguros"},
    {"canonical_name": "Claude.ai",
     "aliases": ["claude.ai", "claude.ai subscription", "claude ai"],
     "mcc_hint": "5817",
     "default_category_slug": "suscripciones.software_saas"},
    {"canonical_name": "Google One",
     "aliases": ["google one", "dlo google google one"],
     "mcc_hint": "5817",
     "default_category_slug": "suscripciones.software_saas"},
    {"canonical_name": "Steam",
     "aliases": ["steam", "steamgames.com"],
     "mcc_hint": "5816",
     "default_category_slug": "suscripciones.videojuegos"},
    {"canonical_name": "Patreon",
     "aliases": ["patreon", "patreon membership"],
     "mcc_hint": "5817",
     "default_category_slug": "suscripciones.apoyo_creadores"},
    {"canonical_name": "Cívica",
     "aliases": ["civica", "recarga civica"],
     "mcc_hint": "4111",
     "default_category_slug": "transporte.transporte_publico"},
)


async def seed_merchants(session: AsyncSession) -> int:
    """Idempotent seed — called once on first boot + whenever the dict grows."""
    inserted = 0
    for entry in _SEED:
        exists = await session.scalar(
            select(Merchant).where(Merchant.canonical_name == entry["canonical_name"])
        )
        if exists is not None:
            continue
        session.add(Merchant(**entry, source="seed"))
        inserted += 1
    if inserted:
        await session.commit()
    return inserted


async def lookup(session: AsyncSession, normalized_text: str) -> MerchantMatch | None:
    """Exact-alias lookup against the merchants table.

    Returns the highest-confidence match or None if nothing hits.

    TODO(carcass): add a RapidFuzz-based fallback that handles typos,
    abbreviations, and the long tail of unseeded Colombian merchants. For now
    that traffic falls through to the retrieval + LLM tier, which is fine.
    """
    if not normalized_text:
        return None

    # Exact-alias match: naive but fast for small dict. DB has GIN index on
    # aliases (added in migration). Below ~10k merchants this is plenty.
    stmt = select(Merchant).where(Merchant.aliases.op("?")(normalized_text))
    hit = await session.scalar(stmt)
    if hit is not None:
        return MerchantMatch(
            canonical_name=hit.canonical_name,
            mcc_hint=hit.mcc_hint,
            default_category_slug=hit.default_category_slug,
            match_confidence=0.95,
            source="seed" if hit.source == "seed" else "learned",
        )

    # Token-contains match: any alias appears as a substring of the normalized text.
    # Slower than the ? op but handles "didi" inside "dlo didi rides co".
    all_merchants = (await session.scalars(select(Merchant))).all()
    best: MerchantMatch | None = None
    for m in all_merchants:
        for alias in m.aliases or []:
            if alias and alias in normalized_text:
                conf = 0.85 if len(alias) >= 5 else 0.75
                if best is None or conf > best.match_confidence:
                    best = MerchantMatch(
                        canonical_name=m.canonical_name,
                        mcc_hint=m.mcc_hint,
                        default_category_slug=m.default_category_slug,
                        match_confidence=conf,
                        source="seed" if m.source == "seed" else "learned",
                    )
                break
    return best
