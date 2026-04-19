"""Rule-tier tests. These define the contract for the deterministic tier."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from categorizer.normalize import normalize
from categorizer.rules import match, resolve_internal_transfer_by_text
from categorizer.schemas import OwnAccountRef, TransactionIn


def _tx(desc: str, *, amount: str = "-10000", ttype: str = "debit") -> TransactionIn:
    return TransactionIn(
        external_id="test",
        account_slug="bancolombia_ahorros_0810",
        tx_date=date(2025, 12, 31),
        amount=Decimal(amount),
        currency="COP",
        description=desc,
        original_description=desc,
        transaction_type=ttype,  # type: ignore[arg-type]
    )


def _own_accounts() -> list[OwnAccountRef]:
    return [
        OwnAccountRef(
            slug="bancolombia_ahorros_0810",
            display_name="Bancolombia Ahorros 0810",
            institution="bancolombia",
            account_number_tail="0810",
            aliases=["bancolombia"],
        ),
        OwnAccountRef(
            slug="nequi",
            display_name="Nequi",
            institution="nequi",
            aliases=["nequi"],
        ),
        OwnAccountRef(
            slug="coink_wallet",
            display_name="Coink",
            institution="coink",
            aliases=["coink"],
        ),
    ]


# ── Internal-transfer detection ──────────────────────────────────────────
@pytest.mark.parametrize(
    "description",
    [
        "TRANSFERENCIA DESDE NEQUI",
        "Recarga desde Bancolombia",
        "Recarga desde: COINK",
    ],
)
def test_internal_transfer_matched_by_text(description: str) -> None:
    hit = resolve_internal_transfer_by_text(normalize(description), _own_accounts())
    assert hit is not None
    assert hit.category_slug == "movimientos_internos.entre_bancos"
    assert hit.confidence >= 0.95


def test_internal_transfer_requires_own_account_match() -> None:
    # "Recarga desde XXX" where XXX is NOT one of the user's own_accounts
    # falls through to the next tier.
    hit = resolve_internal_transfer_by_text(
        normalize("Recarga desde: AMIGO_AJENO"), _own_accounts()
    )
    assert hit is None


def test_internal_transfer_requires_counterparty() -> None:
    # No tail after "Recarga desde" → must fall through (will be caught by the
    # paired-tx tier in cascade.py, not by this text-only check).
    hit = resolve_internal_transfer_by_text(normalize("RECARGA DESDE"), _own_accounts())
    assert hit is None


# ── Specific deterministic categories ────────────────────────────────────
@pytest.mark.parametrize(
    ("description", "expected_slug"),
    [
        ("IMPTO GOBIERNO 4X1000", "finanzas.impuestos"),
        ("CXC IMPTO GOBIERNO 4X1000 MON", "finanzas.impuestos"),
        ("CUOTA MANEJO TRJ DEB 10 25", "finanzas.comisiones_bancarias"),
        ("MORA TARJETA MASTER PESOS", "finanzas.mora"),
        ("INTERESES MORA", "finanzas.mora"),
        ("INTERESES CORRIENTES", "finanzas.intereses_tarjeta"),
        ("ABONO INTERESES AHORROS", "ingresos.intereses_bancarios"),
        ("Pago de Intereses", "ingresos.intereses_bancarios"),
        ("RETIRO EN CAJERO", "movimientos_internos.retiro_efectivo"),
        ("PAGO SUC VIRT TC MASTER PESOS", "movimientos_internos.pago_tarjeta_propia"),
        ("ABONO DEBITO POR MORA", "movimientos_internos.pago_tarjeta_propia"),
        ("REVERSO COMPRA EN EBANX", "ingresos.reembolsos"),
        ("PAGO FACTURA EPM", "hogar.servicios_publicos"),
        ("COMPRA PSE EN COMPENSAR-", "salud.eps_seguros"),
        ("DLO*Didi", "transporte.taxi_rideshare"),
        ("DL*DIDI RIDES CO", "transporte.taxi_rideshare"),
        ("RAPPI COLOMBIA*DL", "comida.domicilios"),
        ("DLO*DiDi Food CO Payin", "comida.domicilios"),
        ("CLAUDE.AI SUBSCRIPTION", "suscripciones.software_saas"),
        ("DLO*GOOGLE Google One", "suscripciones.software_saas"),
        ("STEAMGAMES.COM 4259522", "suscripciones.videojuegos"),
        ("COMPRA EN Patreon Membership", "suscripciones.apoyo_creadores"),
        ("CINEMARK", "ocio.cine_teatro"),
        ("TIENDA D1 AUTOMOTRIZ", "comida.mercado"),
        ("PV DOGGER EXITO POBLAD", "comida.mercado"),
        ("Recarga Cívica", "transporte.transporte_publico"),
        # salud.farmacia — generic "droguería" plus named chains.
        ("COMPRA EN  DROGUERIA", "salud.farmacia"),
        ("DROGUERIA ALEMANA 142", "salud.farmacia"),
        ("PAGO QR Drogueria Moontiny", "salud.farmacia"),
        ("FARMATODO CC POBLADO", "salud.farmacia"),
        ("CRUZ VERDE 123", "salud.farmacia"),
        ("LOCATEL ENVIGADO", "salud.farmacia"),
        # hogar.internet_telefonia — PTM prepaid packages (Claro) plus existing telco patterns.
        ("COMPRA PAQUETE PTM CLARO", "hogar.internet_telefonia"),
        ("COMPRA PAQUETE PTM", "hogar.internet_telefonia"),
        ("RECARGA PTM CLARO", "hogar.internet_telefonia"),
        ("MOVISTAR", "hogar.internet_telefonia"),
        ("COMPRA PSE EN CLARO", "hogar.internet_telefonia"),
    ],
)
def test_rule_tier_emits_expected_category(description: str, expected_slug: str) -> None:
    hit = match(_tx(description), normalize(description), _own_accounts())
    assert hit is not None, f"No rule matched {description!r}"
    assert hit.category_slug == expected_slug


# ── Negatives: these MUST fall through (no rule match) ──────────────────
@pytest.mark.parametrize(
    "description",
    [
        # Anonymized Nequi counterparty — can't decide without more signal.
        "MAR*** ELI*** GON*** DIA***",
        # Person-to-person outgoing — the text tier emits `envio_persona`
        # but the rule library doesn't classify it deterministically
        # (could be legitimate P2P or a friend paying you back). Downstream
        # tiers decide.
        "Para LUISA FERNANDA",
        # Bare QR payment with no tail.
        "PAGO EN QR BRE-B:",
        # "farmacéutica" is a lab/manufacturer, not a pharmacy.
        "PAGO A FARMACEUTICA SA",
        # "droga" without the -ueria suffix is ambiguous.
        "COMPRA EN DROGA XYZ",
        # "ptm" embedded in an unrelated token should not trigger the telco rule.
        "COMPRA EN SEPTMBRE SAS",
    ],
)
def test_rule_tier_falls_through(description: str) -> None:
    hit = match(_tx(description), normalize(description), _own_accounts())
    assert hit is None, f"Unexpected rule match for {description!r}"
