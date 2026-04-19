"""Normalization tests against REAL bank-statement strings from Sebas's extractos."""

from __future__ import annotations

import pytest

from categorizer.normalize import normalize


@pytest.mark.parametrize(
    ("raw", "expected_tokens"),
    [
        # Nequi QR payments — strips channel, keeps tail.
        ("PAGO EN QR BRE-B: Mi Dulce", {"mi", "dulce"}),
        ("PAGO EN QR BRE-B:", set()),
        ("PAGO EN QR BRE-B: Toretos", {"toretos"}),
        # Nequi recipient flows — emits a semantic placeholder token.
        ("ENVIO CON BRE-B A: JOHN", {"envio_persona", "john"}),
        ("Para LUISA FERNANDA", {"envio_persona", "luisa", "fernanda"}),
        ("De CRISTIAN DAVID CANO", {"recibido_persona", "cristian", "david", "cano"}),
        ("RECIBÍ A MI LLAVE DE: CRISTIAN", {"recibido_persona", "cristian"}),
        # Bancolombia savings account movements.
        ("TRANSFERENCIA DESDE NEQUI", {"transferencia_desde", "nequi"}),
        ("PAGO SUC VIRT TC MASTER PESOS", {"pago_suc_virt", "tc", "master", "pesos"}),
        ("CUOTA MANEJO TRJ DEB 10 25", {"cuota_manejo"}),
        ("IMPTO GOBIERNO 4X1000", {"gmf_4x1000"}),
        ("CXC IMPTO GOBIERNO 4X1000 MON", {"gmf_4x1000_cxc"}),
        ("ABONO INTERESES AHORROS", {"abono_intereses_ahorros"}),
        ("MORA TARJETA MASTER PESOS", {"mora_tarjeta"}),
        ("ABONO DEBITO POR MORA", {"abono_debito_mora"}),
        # Credit-card statement patterns. Dots are stripped by normalize — the
        # rule tier matches "claude ai" / "steamgames" without the dot.
        ("CLAUDE.AI SUBSCRIPTION", {"claude", "ai", "subscription"}),
        ("STEAMGAMES.COM 4259522", {"steamgames", "com"}),
        ("DLO*Didi", {"didi"}),
        ("DL*DIDI RIDES CO", {"didi", "rides", "co"}),
        ("BOLD*Macchiato Caffe", {"macchiato", "caffe"}),
        ("TIENDA D1 AUTOMOTRIZ", {"tienda", "d1", "automotriz"}),
        ("PV DOGGER EXITO POBLAD", {"pv", "dogger", "exito", "poblad"}),
        ("RAPPI COLOMBIA*DL", {"rappi", "colombia"}),
        ("DLO*GOOGLE Google One", {"google", "one"}),
        # Mixed patterns.
        ("COMPRA PSE EN COMPENSAR-$801,600.00", {"pse", "compensar"}),
        ("COMPRA EN EBANX", {"ebanx"}),
        ("REVERSO COMPRA EN EBANX", {"reverso", "ebanx"}),
        ("RETIRO EN CAJERO", {"retiro_cajero"}),
        ("Recarga desde Bancolombia", {"recarga_desde", "bancolombia"}),
        ("Recarga desde: COINK", {"recarga_desde", "coink"}),
        ("Recarga Cívica", {"recarga", "civica"}),
        # Anonymized Nequi counterparties.
        ("MAR*** ELI*** GON*** DIA***", {"nombre_anonimo"}),
        ("DEI*** JHO*** GAR*** GOM***", {"nombre_anonimo"}),
    ],
)
def test_normalize_contains_expected_tokens(raw: str, expected_tokens: set[str]) -> None:
    tokens = set(normalize(raw).split())
    missing = expected_tokens - tokens
    assert not missing, f"{raw!r} normalized to {tokens!r}, missing {missing!r}"


def test_normalize_is_lowercase() -> None:
    assert normalize("PAGO FACTURA EPM") == normalize("pago factura epm")


def test_normalize_preserves_ñ() -> None:
    out = normalize("PAGO EN QR BRE-B: Baño")
    assert "baño" in out.split()


def test_normalize_idempotent() -> None:
    for raw in (
        "COMPRA EN DLO Didi",
        "Pago recibido de SOLUCIONES",
        "PAGO EN QR BRE-B: Barberia",
    ):
        once = normalize(raw)
        twice = normalize(once)
        assert once == twice
