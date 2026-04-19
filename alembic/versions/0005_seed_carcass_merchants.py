"""seed merchants for carcass fill-ins

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-19

Adds merchant dictionary entries for categories that were declared but
unimplemented in taxonomy.yaml (`implemented: false`). The rule tier got
matching regexes in the same commit series; this migration seeds the
merchant-resolver tier so both layers can fire on the same input.

Grounded in Sebas's real statements: only merchants with ≥ 2 real-world
occurrences earn an entry here, plus well-known Colombian chains so the
dictionary is useful beyond Sebas's personal history.

Follows 0002's `on_conflict_do_nothing(index_elements=["canonical_name"])`
pattern so the migration is idempotent across re-runs.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


_MERCHANTS = (
    # (canonical_name, aliases, mcc_hint, default_category_slug)
    # ── salud.farmacia ─────────────────────────────────────────────────
    ("Droguería", ["drogueria", "compra en drogueria"], "5912", "salud.farmacia"),
    ("Droguería Alemana", ["drogueria alemana", "alemana 274"], "5912", "salud.farmacia"),
    ("Droguería Dromarin", ["drogueria dromarin", "cac drogueria dromarin"], "5912", "salud.farmacia"),
    ("Farmatodo", ["farmatodo"], "5912", "salud.farmacia"),
    ("Cruz Verde", ["cruz verde"], "5912", "salud.farmacia"),
    ("Locatel", ["locatel"], "5912", "salud.farmacia"),
    ("Copidrogas", ["copidrogas"], "5912", "salud.farmacia"),
    ("Drogas La Rebaja", ["drogas la rebaja", "la rebaja"], "5912", "salud.farmacia"),
    # ── hogar.internet_telefonia (prepaid PTM + WOM) ──────────────────
    ("Claro PTM", ["paquete ptm", "paquete ptm claro", "recarga ptm", "recarga ptm claro", "ptm claro"], "4814", "hogar.internet_telefonia"),
    ("WOM", ["wom"], "4814", "hogar.internet_telefonia"),
    # ── compras.ropa ──────────────────────────────────────────────────
    ("Arturo Calle", ["arturo calle"], "5651", "compras.ropa"),
)


def upgrade() -> None:
    conn = op.get_bind()
    metadata = sa.MetaData()
    merchants = sa.Table("merchants", metadata, autoload_with=conn)

    for canonical_name, aliases, mcc, slug_cat in _MERCHANTS:
        stmt = (
            sa.dialects.postgresql.insert(merchants)
            .values(
                canonical_name=canonical_name,
                aliases=aliases,
                mcc_hint=mcc,
                default_category_slug=slug_cat,
                source="seed",
            )
            .on_conflict_do_nothing(index_elements=["canonical_name"])
        )
        conn.execute(stmt)


def downgrade() -> None:
    conn = op.get_bind()
    names = [x[0] for x in _MERCHANTS]
    conn.execute(sa.text("DELETE FROM merchants WHERE canonical_name = ANY(:n)"), {"n": names})
