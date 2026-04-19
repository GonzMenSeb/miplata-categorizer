"""seed own_accounts and merchants

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-18

Seeds the six own_accounts that Sebas currently has (two Bancolombia
savings, Bancolombia MasterCard, Bancolombia AmEx, Nequi, plus an
external wallet slot) and a starter merchants dict expanded from the
in-code SEED in merchant.py.

The own_account slugs are friendly / human-readable. If miplata's own
Account.id UUIDs need to align with these later, add them as additional
aliases via a follow-up migration or via POST /v1/own-accounts.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSON

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


_OWN_ACCOUNTS = (
    # slug, display_name, institution, account_number_tail, aliases
    ("bancolombia_ahorros_0810", "Bancolombia Ahorros 0810", "bancolombia", "0810",
     ["bancolombia", "bancolombia_ahorros", "ahorros"]),
    ("bancolombia_ahorros_9855", "Bancolombia Ahorros 9855", "bancolombia", "9855",
     ["bancolombia", "bancolombia_ahorros", "ahorros"]),
    ("bancolombia_mastercard_1194", "Bancolombia MasterCard 1194", "bancolombia", "1194",
     ["bancolombia", "mastercard", "tc_master", "tc"]),
    ("bancolombia_amex_1916", "Bancolombia AmEx 1916", "bancolombia", "1916",
     ["bancolombia", "amex", "americanexpress"]),
    ("nequi", "Nequi", "nequi", None,
     ["nequi"]),
    ("coink_wallet", "Coink", "coink", None,
     ["coink"]),
)

# Expanded merchants seed — covers what was in merchant.py's SEED plus a
# wider set of Colombian chains that appeared in real extractos.
_MERCHANTS = (
    # (canonical_name, aliases, mcc_hint, default_category_slug)
    ("Rappi", ["rappi", "rappi colombia", "rappi sas"], "5812", "comida.domicilios"),
    ("Didi", ["didi", "dlo didi", "dl didi rides co", "didi co ride", "didi rides"], "4121", "transporte.taxi_rideshare"),
    ("Didi Food", ["didi food", "dlo didi food", "dlo didi food co payin"], "5812", "comida.domicilios"),
    ("Éxito", ["exito", "pv dogger exito poblad", "exito poblad", "almacenes exito"], "5411", "comida.mercado"),
    ("Carulla", ["carulla"], "5411", "comida.mercado"),
    ("Jumbo", ["jumbo"], "5411", "comida.mercado"),
    ("Olímpica", ["olimpica", "sao"], "5411", "comida.mercado"),
    ("Ara", ["ara"], "5411", "comida.mercado"),
    ("Makro", ["makro"], "5411", "comida.mercado"),
    ("Tienda D1", ["tienda d1", "tienda d1 automotriz", "d1"], "5411", "comida.mercado"),
    ("Cinemark", ["cinemark"], "7832", "ocio.cine_teatro"),
    ("Cine Colombia", ["cine colombia"], "7832", "ocio.cine_teatro"),
    ("Cinépolis", ["cinepolis"], "7832", "ocio.cine_teatro"),
    ("EPM", ["epm", "pago factura epm", "empresas publicas"], "4900", "hogar.servicios_publicos"),
    ("Compensar", ["compensar"], "8099", "salud.eps_seguros"),
    ("Sura", ["sura"], "6300", "salud.eps_seguros"),
    ("Nueva EPS", ["nueva eps"], "8099", "salud.eps_seguros"),
    ("Sanitas", ["sanitas"], "8099", "salud.eps_seguros"),
    ("Claro", ["claro", "claro colombia", "comcel"], "4814", "hogar.internet_telefonia"),
    ("Movistar", ["movistar", "telefonica"], "4814", "hogar.internet_telefonia"),
    ("Tigo", ["tigo"], "4814", "hogar.internet_telefonia"),
    ("ETB", ["etb"], "4814", "hogar.internet_telefonia"),
    ("Claude.ai", ["claude ai", "claude ai subscription", "claude"], "5817", "suscripciones.software_saas"),
    ("Google One", ["google one", "dlo google google one"], "5817", "suscripciones.software_saas"),
    ("Google Play", ["google play"], "5816", "suscripciones.videojuegos"),
    ("GitHub", ["github"], "5817", "suscripciones.software_saas"),
    ("Steam", ["steam", "steamgames", "steamgames com"], "5816", "suscripciones.videojuegos"),
    ("PlayStation", ["playstation", "psn", "sony playstation network"], "5816", "suscripciones.videojuegos"),
    ("Xbox", ["xbox"], "5816", "suscripciones.videojuegos"),
    ("Patreon", ["patreon", "patreon membership"], "5817", "suscripciones.apoyo_creadores"),
    ("Netflix", ["netflix"], "4899", "suscripciones.streaming"),
    ("Spotify", ["spotify"], "4899", "suscripciones.streaming"),
    ("Disney+", ["disney", "disney+", "disney plus"], "4899", "suscripciones.streaming"),
    ("HBO Max", ["hbo max", "hbo"], "4899", "suscripciones.streaming"),
    ("Prime Video", ["prime video", "amazon prime video"], "4899", "suscripciones.streaming"),
    ("Cívica", ["civica", "recarga civica"], "4111", "transporte.transporte_publico"),
    ("TransMilenio", ["transmilenio", "tullave"], "4111", "transporte.transporte_publico"),
    ("Uber", ["uber"], "4121", "transporte.taxi_rideshare"),
    ("Cabify", ["cabify"], "4121", "transporte.taxi_rideshare"),
    ("MercadoLibre", ["mercadolibre", "mercado libre"], "5399", "compras.tiendas_online"),
    ("Amazon", ["amazon", "amzn mktp", "amzn"], "5399", "compras.tiendas_online"),
    ("EBANX", ["ebanx"], "6012", "compras.tiendas_online"),
    ("PayPal", ["paypal"], "6012", "compras.tiendas_online"),
)


def upgrade() -> None:
    conn = op.get_bind()
    metadata = sa.MetaData()

    own_accounts = sa.Table("own_accounts", metadata, autoload_with=conn)
    merchants = sa.Table("merchants", metadata, autoload_with=conn)

    for slug, display_name, institution, tail, aliases in _OWN_ACCOUNTS:
        stmt = (
            sa.dialects.postgresql.insert(own_accounts)
            .values(
                slug=slug,
                display_name=display_name,
                institution=institution,
                account_number_tail=tail,
                aliases=aliases,
            )
            .on_conflict_do_nothing(index_elements=["slug"])
        )
        conn.execute(stmt)

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
    slugs = [x[0] for x in _OWN_ACCOUNTS]
    names = [x[0] for x in _MERCHANTS]
    conn.execute(sa.text("DELETE FROM own_accounts WHERE slug = ANY(:s)"), {"s": slugs})
    conn.execute(sa.text("DELETE FROM merchants WHERE canonical_name = ANY(:n)"), {"n": names})
