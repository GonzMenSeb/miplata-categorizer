"""Deterministic rule tier.

This is the first line of the cascade and the one that should absorb the
biggest chunk of traffic with the highest confidence. Rules fire on
normalized descriptions (see normalize.py), and a rule match is gated on
EVERY condition being satisfied — we don't return a rule-based prediction
unless we're certain.

Special cases handled here:

  • **Internal-transfer detection** — by far the most important rule. A
    "Transferencia desde Nequi" is *only* an internal transfer if the user's
    Nequi account is actually one of THEIR own_accounts. We never hard-code
    that; it's checked against the `own_accounts` set at runtime.

  • **Colombian-specific deterministic patterns** — GMF 4x1000, cuota manejo,
    mora, intereses (both ways), Cívica reload, EPM, Compensar, etc.

  • **High-confidence merchants** — when the normalized token IS the merchant
    (e.g. "rappi colombia", "cinemark", "claude.ai"), we can emit directly
    without consulting the LLM.

Anything ambiguous (anon'd Nequi recipient names `nombre_anonimo`, bare
`PAGO EN QR BRE-B:` with no tail, unknown merchants) falls through to the
next tier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import OwnAccountRef, TransactionIn


@dataclass(frozen=True)
class RuleHit:
    category_slug: str
    confidence: float
    reasoning: str


# ───── Pattern library (applied over the NORMALIZED description) ─────

# Patterns below expect the already-normalized text. The tokens
# transferencia_desde / recarga_desde / envio_persona / etc. are emitted by
# normalize.py. Raw-text pattern mistakes are NOT acceptable — fix normalize.py
# instead of adding a workaround here.

_MERCHANT_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    # ── Finanzas ────────────────────────────────────────────────────────
    (re.compile(r"\bgmf_4x1000\b"), "finanzas.impuestos", "GMF (4x1000) es impuesto financiero."),
    (re.compile(r"\bgmf_4x1000_cxc\b"), "finanzas.impuestos", "Cuenta por cobrar del GMF 4x1000."),
    (re.compile(r"\bcuota_manejo\b"), "finanzas.comisiones_bancarias", "Cuota manejo de tarjeta débito."),
    (re.compile(r"\bmora_tarjeta\b"), "finanzas.mora", "Mora en tarjeta de crédito."),
    (re.compile(r"\bintereses_mora\b"), "finanzas.mora", "Intereses de mora."),
    (re.compile(r"\bintereses_corrientes\b"), "finanzas.intereses_tarjeta", "Intereses corrientes de TC."),
    (re.compile(r"\babono_intereses_ahorros\b"), "ingresos.intereses_bancarios", "Intereses pagados sobre saldo de ahorros."),

    # ── Ingresos estructurales ──────────────────────────────────────────
    (re.compile(r"\breverso\b"), "ingresos.reembolsos", "Reverso de compra."),

    # ── Movimientos internos (marca la rama; `resolve_internal_transfer` ya
    #    validó el match de own_account antes de llegar aquí para estos casos).
    # Los patrones genéricos `transferencia_desde` / `recarga_desde` pasan
    # por resolve_internal_transfer para confirmar que la contraparte es
    # dueño; este bloque sólo cubre los determinísticos puros.
    (re.compile(r"\bpago_suc_virt\b"), "movimientos_internos.pago_tarjeta_propia", "Pago de TC propia desde sucursal virtual."),
    (re.compile(r"\babono_suc_virt\b"), "movimientos_internos.pago_tarjeta_propia", "Abono a TC propia desde sucursal virtual."),
    (re.compile(r"\babono_debito_mora\b"), "movimientos_internos.pago_tarjeta_propia", "Abono a TC propia por cobro de mora."),
    (re.compile(r"\bretiro_cajero\b"), "movimientos_internos.retiro_efectivo", "Retiro de efectivo en cajero."),
    (re.compile(r"\brecarga\s+civica\b"), "transporte.transporte_publico", "Recarga de tarjeta Cívica (Metro Medellín)."),

    # ── Servicios públicos / salud ─────────────────────────────────────
    (re.compile(r"\bepm\b"), "hogar.servicios_publicos", "Factura EPM (energía, acueducto, gas, internet)."),
    (re.compile(r"\bcompensar\b"), "salud.eps_seguros", "Pago EPS Compensar."),
    (re.compile(r"\b(sura|sanitas|nueva\s+eps)\b"), "salud.eps_seguros", "EPS / seguro de salud."),
    (re.compile(r"\b(claro|movistar|tigo|\betb\b)\b"), "hogar.internet_telefonia", "Operador de telefonía/internet."),

    # ── Transporte ──────────────────────────────────────────────────────
    # `didi food` FIRST so the more-specific pattern wins over generic `didi`.
    (re.compile(r"\bdidi\s+food\b"), "comida.domicilios", "Didi Food."),
    (re.compile(r"\bdidi\b|\bdidi\s+rides\b|\bdidi\s+co\s+ride\b"), "transporte.taxi_rideshare", "Didi ride."),
    (re.compile(r"\buber\b"), "transporte.taxi_rideshare", "Uber."),
    (re.compile(r"\bcabify\b"), "transporte.taxi_rideshare", "Cabify."),

    # ── Comida ──────────────────────────────────────────────────────────
    (re.compile(r"\brappi\s+colombia\b|\brappi\b"), "comida.domicilios", "Rappi (domicilios)."),
    (re.compile(r"\btienda\s+d1\b|\bd1\s+"), "comida.mercado", "Tienda D1."),
    (re.compile(r"\bpv\s+dogger\s+exito\b|\bexito\s+poblad\b|\bexito\b"), "comida.mercado", "Éxito."),
    (re.compile(r"\b(carulla|jumbo|olimpica|ara)\b"), "comida.mercado", "Cadena de mercado."),
    (re.compile(r"\bpasteleria|panaderia|pan\s+dulce\b"), "comida.cafe_panaderia", "Panadería / pastelería."),

    # ── Suscripciones / digital ────────────────────────────────────────
    # Dots are stripped by normalize.py — patterns must not rely on them.
    (re.compile(r"\bclaude\s+ai\s+subscription\b|\bclaude\s+ai\b"), "suscripciones.software_saas", "Claude.ai suscripción."),
    (re.compile(r"\bgoogle\s+one\b"), "suscripciones.software_saas", "Google One."),
    (re.compile(r"\bgoogle\s+play\b|\bgoogle\b"), "suscripciones.videojuegos", "Google Play (apps/juegos)."),
    (re.compile(r"\bsteamgames\b|\bsteam\b"), "suscripciones.videojuegos", "Steam."),
    (re.compile(r"\bnetflix\b"), "suscripciones.streaming", "Netflix."),
    (re.compile(r"\bspotify\b"), "suscripciones.streaming", "Spotify."),
    (re.compile(r"\b(disney|hbo|prime\s+video)\b"), "suscripciones.streaming", "Streaming."),
    (re.compile(r"\bpatreon\b"), "suscripciones.apoyo_creadores", "Patreon."),

    # ── Ocio ────────────────────────────────────────────────────────────
    (re.compile(r"\bcinemark\b"), "ocio.cine_teatro", "Cinemark."),
    (re.compile(r"\b(cine\s+colombia|cinepolis)\b"), "ocio.cine_teatro", "Cine."),

    # ── Tiendas online ──────────────────────────────────────────────────
    (re.compile(r"\bebanx\b"), "compras.tiendas_online", "Pago vía EBANX (gateway)."),
    (re.compile(r"\bmercadolibre\b"), "compras.tiendas_online", "Mercado Libre."),
    (re.compile(r"\bamazon\b"), "compras.tiendas_online", "Amazon."),
)


# ── Internal-transfer resolver ─────────────────────────────────────────
#
# Invariant: an internal transfer must mention an own_account on one side of
# the description (the "counterparty" side) OR appear with a paired tx on
# another own_account within a short window. The second check requires DB
# access and is done in cascade.py / storage.py; this module handles the
# pure-text case only.
#
# Examples matched HERE (pure text, cheap):
#   "recarga_desde bancolombia"  → matches own_account slug=bancolombia_*
#   "transferencia_desde nequi"  → matches own_account slug=nequi_*
#   "recarga_desde coink"        → coink is an aliased channel, configured per user
#
# Examples NOT matched here (paired-tx check in DB):
#   "recarga_desde" with no tail    → must be paired with a same-amount debit on
#                                      another own_account within ±5 min / ±2 days.

_INTERNAL_HINT_PATTERN = re.compile(
    r"\b(transferencia_desde|transferencia_a|recarga_desde|recarga_a)\b\s*([a-z0-9_\-]*)"
)


def resolve_internal_transfer_by_text(
    normalized: str, own_accounts: list[OwnAccountRef]
) -> RuleHit | None:
    """Pure text-based internal-transfer detection.

    If the description names a counterparty that matches any of the user's
    own_accounts (by institution or alias), classify as `movimientos_internos.entre_bancos`.
    """
    if not own_accounts:
        return None

    match = _INTERNAL_HINT_PATTERN.search(normalized)
    if not match:
        return None

    channel, raw_counterparty = match.group(1), (match.group(2) or "").strip()
    if not raw_counterparty:
        # "Recarga desde" with no tail → needs paired-tx lookup in DB.
        return None

    aliases_to_match = {raw_counterparty}
    # strip "desde " / "a " prefix residues that may leak through
    aliases_to_match.update(
        {raw_counterparty.replace("_", " "), raw_counterparty.replace("_", "-")}
    )

    for acct in own_accounts:
        candidates = {acct.institution.lower(), acct.slug.lower(), *[a.lower() for a in acct.aliases]}
        if aliases_to_match & candidates:
            return RuleHit(
                category_slug="movimientos_internos.entre_bancos",
                confidence=0.97,
                reasoning=(
                    f"Transferencia {channel} identificada contra cuenta propia "
                    f"'{acct.display_name}' por alias '{raw_counterparty}'."
                ),
            )
    return None


# ── Public entry point ─────────────────────────────────────────────────
def match(
    tx: TransactionIn,
    normalized: str,
    own_accounts: list[OwnAccountRef],
) -> RuleHit | None:
    """Return a RuleHit with confidence ≥ 0.95 or None.

    Callers should not feed through predictions below 0.95 — the cascade
    escalates to the next tier instead. Keep it binary.
    """

    internal = resolve_internal_transfer_by_text(normalized, own_accounts)
    if internal is not None:
        return internal

    for pat, slug, reason in _MERCHANT_RULES:
        if pat.search(normalized):
            return RuleHit(
                category_slug=slug,
                confidence=0.96,
                reasoning=reason,
            )

    return None
