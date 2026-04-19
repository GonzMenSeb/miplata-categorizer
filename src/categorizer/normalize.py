"""Colombian-Spanish bank/wallet transaction string normalization.

The goal is NOT to lossy-simplify the description — it's to strip the
*noise tokens* that every Colombian bank adds around the real merchant or
recipient name. Tokens we strip:

  • Processor / channel prefixes:  COMPRA, CARGO, PAGO (when followed by channel),
    POS, PSE, BRE-B (Bancolombia's low-value transfer), TDC/TC, QR, SUC VIRT.
  • Date fragments:                04/17, 17ABR, 2025/12/31.
  • Auth / terminal codes:         #A9C82, REF 9988, trailing number runs.
  • Currency noise:                $, COP, USD.
  • Bank-statement boilerplate:    "COMPRA EN ...", "PAGO EN QR BRE-B: ..." prefixes.

What remains is roughly the merchant or recipient identifier — which is what
embedding, rule-matching, and merchant-resolution care about.

The regex list deliberately order-dependent: strip the most specific patterns
first, then fall back to generic cleanup.
"""

from __future__ import annotations

import re
import unicodedata

# Order matters: these regexes are applied in sequence.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Bancolombia / Nequi channel prefixes — strip them AND keep whatever follows.
    (re.compile(r"\bPAGO\s+EN\s+QR\s+BRE-?B\s*:?\s*", re.IGNORECASE), " "),
    (re.compile(r"\bENVIO\s+CON\s+BRE-?B\s+A\s*:?\s*", re.IGNORECASE), " envio_persona "),
    (re.compile(r"\bRECIBI\s+POR\s+BRE-?B\s+DE\s*:?\s*", re.IGNORECASE), " recibido_persona "),
    (re.compile(r"\bRECIB[IÍ]\s+A\s+MI\s+LLAVE\s+DE\s*:?\s*", re.IGNORECASE), " recibido_persona "),
    (re.compile(r"\bCOMPRA\s+EN\s+", re.IGNORECASE), " "),
    (re.compile(r"\bCOMPRA\s+PSE\s+EN\s+", re.IGNORECASE), " pse "),
    (re.compile(r"\bCOMPRA\s+PSE\b", re.IGNORECASE), " pse "),
    (re.compile(r"\bCOMPRA\b", re.IGNORECASE), " "),
    (re.compile(r"\bCARGO\b", re.IGNORECASE), " "),
    (re.compile(r"\bPAGO\s+FACTURA\s+", re.IGNORECASE), " "),
    (re.compile(r"\bPAGO\s+SUC\s+VIRT\s+", re.IGNORECASE), " pago_suc_virt "),
    (re.compile(r"\bABONO\s+SUCURSAL\s+VIRTUAL\b", re.IGNORECASE), " abono_suc_virt "),
    (re.compile(r"\bABONO\s+DEBITO\s+POR\s+MORA\b", re.IGNORECASE), " abono_debito_mora "),
    (re.compile(r"\bABONO\s+INTERESES\s+AHORROS\b", re.IGNORECASE), " abono_intereses_ahorros "),
    (re.compile(r"\bPAGO\s+DE\s+INTERESES\b", re.IGNORECASE), " abono_intereses_ahorros "),
    (re.compile(r"\bTRANSFERENCIA\s+DESDE\s+", re.IGNORECASE), " transferencia_desde "),
    (re.compile(r"\bTRANSFERENCIA\s+A\s+", re.IGNORECASE), " transferencia_a "),
    (re.compile(r"\bRECARGA\s+DESDE\s*:?\s*", re.IGNORECASE), " recarga_desde "),
    (re.compile(r"\bRECARGA\s+A\s+", re.IGNORECASE), " recarga_a "),
    (re.compile(r"\bRETIRO\s+EN\s+CAJERO\b", re.IGNORECASE), " retiro_cajero "),
    (re.compile(r"\bPARA\s+(?=[A-ZÁÉÍÓÚÑ])", re.IGNORECASE), " envio_persona "),  # "Para LUISA…"
    (re.compile(r"\bDE\s+(?=[A-ZÁÉÍÓÚÑ]{3,})", re.IGNORECASE), " recibido_persona "),
    # More specific first so the generic gmf_4x1000 rule doesn't swallow its prefix/suffix.
    (re.compile(r"\bCXC\s+IMPT?O?\.?\s+GOBIERNO\s+4\s*X\s*1000\s+MON\b", re.IGNORECASE), " gmf_4x1000_cxc "),
    (re.compile(r"\bIMPT?O?\.?\s+GOBIERNO\s+4\s*X\s*1000\b", re.IGNORECASE), " gmf_4x1000 "),
    (re.compile(r"\b4\s*X\s*1000\b", re.IGNORECASE), " gmf_4x1000 "),
    (re.compile(r"\bCUOTA\s+MANEJO\s+TRJ\s+DEB\b.*$", re.IGNORECASE), " cuota_manejo "),
    (re.compile(r"\bCUOTA\s+MANEJO\b", re.IGNORECASE), " cuota_manejo "),
    (re.compile(r"\bMORA\s+TARJETA\s+MASTER\s+PESOS\b", re.IGNORECASE), " mora_tarjeta "),
    (re.compile(r"\bINTERESES\s+MORA\b", re.IGNORECASE), " intereses_mora "),
    (re.compile(r"\bINTERESES\s+CORRIENTES\b", re.IGNORECASE), " intereses_corrientes "),
    (re.compile(r"\bREVERSO\s+COMPRA\s+EN\b", re.IGNORECASE), " reverso "),
    (re.compile(r"\bREVERSO\b", re.IGNORECASE), " reverso "),
    # Payment-processor / aggregator prefixes common in Bancolombia CC statements.
    (re.compile(r"\bDLO\*\s*", re.IGNORECASE), " "),
    (re.compile(r"\bDL\*\s*", re.IGNORECASE), " "),
    (re.compile(r"\bBOLD\*\s*", re.IGNORECASE), " "),
    (re.compile(r"\bSQ\s*\*\s*", re.IGNORECASE), " "),
    (re.compile(r"\bTST\s*\*\s*", re.IGNORECASE), " "),
    (re.compile(r"\bAMZ\*\s*", re.IGNORECASE), " "),
    (re.compile(r"\bAMZN\s+MKTP\s+US\*?\s*", re.IGNORECASE), " amazon "),
    (re.compile(r"\bPAYPAL\s*\*\s*", re.IGNORECASE), " paypal "),
    # Dates in common Colombian formats.
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b"), " "),
    (re.compile(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"), " "),
    (re.compile(r"\b\d{1,2}[A-Z]{3,}\b", re.IGNORECASE), " "),   # 17ABR, 04NOV
    # Trailing / standalone digit runs (auth codes, terminal ids, monetary values).
    (re.compile(r"#\s*\d{3,}\b"), " "),
    (re.compile(r"\b\d{5,}\b"), " "),
    # Currency markers.
    (re.compile(r"\$[\d.,]+"), " "),
    (re.compile(r"\bCOP\b|\bUSD\b", re.IGNORECASE), " "),
    # Anonymized Nequi recipient names like "MAR*** ELI*** GON*** DIA***" → placeholder.
    # Boundary uses (?=\s|$) rather than \b because `*` is a non-word char and
    # \b after it is not a valid word boundary.
    (
        re.compile(r"(?<!\w)[A-ZÁÉÍÓÚÑ]{2,4}\*{2,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,4}\*{2,}){1,4}(?=\s|$)"),
        " nombre_anonimo ",
    ),
    # Residual asterisks / punctuation. KEEP underscores — normalize.py emits
    # sentinel tokens like `gmf_4x1000` that rule patterns depend on.
    (re.compile(r"[:*|/\\]+"), " "),
    (re.compile(r"[^\w\s\-ÁÉÍÓÚÑáéíóúñ]+"), " "),
)


def _strip_accents(text: str) -> str:
    # Preserve ñ/Ñ (distinguishes real Spanish words: "baño" vs "bano").
    # NFKD decomposition would turn ñ into `n` + combining tilde and then drop
    # the tilde, so we shield ñ/Ñ with non-printable sentinels first.
    shielded = text.replace("ñ", "\x01").replace("Ñ", "\x02")
    stripped = "".join(
        ch for ch in unicodedata.normalize("NFKD", shielded) if not unicodedata.combining(ch)
    )
    return stripped.replace("\x01", "ñ").replace("\x02", "Ñ")


def normalize(description: str) -> str:
    """Deterministic normalization for categorization-facing text.

    Safe to cache on (raw_description) — pure function, no randomness.
    """
    if not description:
        return ""
    text = _strip_accents(description.strip())
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    # Strip leading/trailing hyphens that survived character-class cleanup.
    text = re.sub(r"(^[\s\-]+|[\s\-]+$)", "", text)
    return text.lower() or description.strip().lower()
