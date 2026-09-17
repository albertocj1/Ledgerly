"""
Structured field extraction from accounting documents.

Approach: deterministic patterns first, LLM fallback second.

That ordering is deliberate and worth defending. For fields with rigid
surface forms - invoice numbers, dates, totals - regex is faster,
free, auditable, and doesn't hallucinate. Sending every document to an
LLM to read a number off a line is expensive and less reliable. The
LLM earns its place on the fields regex genuinely can't reach (party
names, free-text service descriptions), where the fallback hook below
plugs in.

The patterns are built to survive OCR damage, because the input is
scanned paper:
  - digits and their visual twins are interchangeable (0/O, 1/l/I, 5/S, 8/B)
  - whitespace between a label and its value may collapse or expand
  - currency symbols are frequently misread as 'S'
  - thousands separators and decimal points get swapped

Accuracy is reported per field by `evaluate.py`, not as one aggregate
number - "extraction is 87% accurate" hides that dates might be at 98%
and vendor names at 60%, which are different engineering problems.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# Characters OCR confuses with digits, mapped back for numeric parsing.
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1",
                            "S": "5", "B": "8", "Z": "2"})

# A currency amount, tolerant of OCR noise: optional $ (or a misread S),
# digits with confusable substitutes, and separators that may be either
# commas or periods.
# Note the optional leading minus: account balances legitimately go
# negative, and omitting it silently dropped ~20% of closing balances.
_AMOUNT = r"[\$S]?\s*(-?\s?[0-9OoIlSBZ][0-9OoIlSBZ.,]{0,15})"

# ISO-style date, allowing confusable characters in the digits.
_DATE = r"([0-9OoIlSBZ]{4}-[0-9OoIlSBZ]{2}-[0-9OoIlSBZ]{2})"


def _norm_amount(raw: str) -> float | None:
    """Parse an OCR-damaged currency string into a float."""
    s = raw.translate(_DIGIT_FIX).strip()
    s = s.rstrip(".,")
    negative = s.startswith("-")
    s = s.lstrip("-").strip()
    if not s:
        return None

    # Decide which separator is decimal. A group of exactly two digits
    # after the last separator means it's the decimal point; otherwise
    # every separator is a thousands mark.
    if "." in s and "," in s:
        last = max(s.rfind("."), s.rfind(","))
        intpart = re.sub(r"[.,]", "", s[:last])
        frac = re.sub(r"[^0-9]", "", s[last + 1:])
        s = f"{intpart}.{frac}" if frac else intpart
    else:
        sep = "." if "." in s else ("," if "," in s else None)
        if sep:
            tail = s.rsplit(sep, 1)[1]
            if len(tail) == 2 and tail.isdigit():
                s = s.rsplit(sep, 1)[0].replace(sep, "") + "." + tail
            else:
                s = s.replace(sep, "")

    s = re.sub(r"[^0-9.]", "", s)
    if not s or s == ".":
        return None
    try:
        value = round(float(s), 2)
    except ValueError:
        return None
    return -value if negative else value


def _norm_date(raw: str) -> str | None:
    s = raw.translate(_DIGIT_FIX)
    return s if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s) else None


def _label_pattern(
    labels: list[str], value: str, require_leader: bool = False
) -> re.Pattern[str]:
    """Build a 'Label: value' matcher tolerant of collapsed whitespace.

    Two details that were silent bugs before being fixed, both worth
    keeping in mind for any label-matching extractor:

    * **Word boundaries are mandatory.** Without `\\b`, the label
      "TOTAL" matches inside "Sub*total*", so the subtotal line - which
      appears first - gets returned as the invoice total. The extractor
      reported a confident, plausible, wrong number. In an accounting
      pipeline that is far worse than extracting nothing.

    * **First match is not always the right match.** On Form 1099-NEC
      the phrase "Nonemployee Compensation" appears in the page title
      before it appears as box 1, so a plain search grabbed the tax
      year instead of the amount. `require_leader` forces a dotted
      leader ("......") between label and value, which is present on
      numbered form boxes and absent from title lines.
    """
    alts = "|".join(re.escape(lbl) for lbl in labels)
    leader = r"\.{3,}\s*" if require_leader else r"[:.]*\s*\.{0,40}\s*"
    return re.compile(rf"\b(?:{alts})\b\s*{leader}{value}", re.IGNORECASE)


@dataclass
class FieldSpec:
    name: str
    pattern: re.Pattern[str]
    normalizer: Callable[[str], object | None]


# Field specs per document type. Labels include the realistic variants
# a vendor might use, since terminology isn't standardised.
SPECS: dict[str, list[FieldSpec]] = {
    "invoice": [
        FieldSpec("document_number",
                  _label_pattern(["Invoice Number", "Invoice No", "Invoice #"],
                                 r"([A-Z0-9\-]{3,20})"),
                  lambda s: s.strip()),
        FieldSpec("issue_date",
                  _label_pattern(["Invoice Date", "Issue Date", "Date"], _DATE),
                  _norm_date),
        FieldSpec("due_date", _label_pattern(["Due Date"], _DATE), _norm_date),
        FieldSpec("total_amount",
                  _label_pattern(["TOTAL DUE", "Total Due", "Amount Due", "TOTAL"], _AMOUNT),
                  _norm_amount),
        FieldSpec("subtotal", _label_pattern(["Subtotal"], _AMOUNT), _norm_amount),
    ],
    "receipt": [
        FieldSpec("document_number",
                  _label_pattern(["Transaction", "Txn", "Trans"], r"([A-Z0-9]{4,20})"),
                  lambda s: s.strip()),
        FieldSpec("issue_date", _label_pattern(["Date"], _DATE), _norm_date),
        FieldSpec("total_amount", _label_pattern(["TOTAL"], _AMOUNT), _norm_amount),
        FieldSpec("subtotal", _label_pattern(["Subtotal"], _AMOUNT), _norm_amount),
    ],
    "purchase_order": [
        FieldSpec("document_number",
                  _label_pattern(["PO Number", "P.O. Number", "Order Number"],
                                 r"([A-Z0-9\-]{3,20})"),
                  lambda s: s.strip()),
        FieldSpec("issue_date", _label_pattern(["PO Date", "Order Date", "Date"], _DATE),
                  _norm_date),
        FieldSpec("total_amount", _label_pattern(["ORDER TOTAL", "Total"], _AMOUNT),
                  _norm_amount),
    ],
    "bank_statement": [
        FieldSpec("opening_balance", _label_pattern(["Opening Balance"], _AMOUNT), _norm_amount),
        FieldSpec("closing_balance", _label_pattern(["Closing Balance"], _AMOUNT), _norm_amount),
    ],
    "w2": [
        FieldSpec("wages",
                  _label_pattern(["Wages, tips, other compensation"], _AMOUNT,
                                 require_leader=True), _norm_amount),
        FieldSpec("federal_tax_withheld",
                  _label_pattern(["Federal income tax withheld"], _AMOUNT,
                                 require_leader=True), _norm_amount),
    ],
    "1099_nec": [
        FieldSpec("nonemployee_compensation",
                  _label_pattern(["Nonemployee compensation"], _AMOUNT,
                                 require_leader=True), _norm_amount),
    ],
    "engagement_letter": [
        FieldSpec("engagement_fee",
                  _label_pattern(["estimated professional fee for this engagement is",
                                  "professional fee is", "fee is"], _AMOUNT),
                  _norm_amount),
    ],
}


def extract(text: str, doc_type: str) -> dict:
    """Extract structured fields for a known document type.

    Returns only fields that matched. A missing key means "not found",
    which downstream code must treat differently from "found and empty" -
    the first routes to human review, the second is a real value.
    """
    out: dict[str, object] = {}
    for spec in SPECS.get(doc_type, []):
        m = spec.pattern.search(text)
        if not m:
            continue
        value = spec.normalizer(m.group(1))
        if value is not None:
            out[spec.name] = value
    return out


def extract_with_coverage(text: str, doc_type: str) -> dict:
    """Extract plus a coverage figure for routing decisions."""
    expected = [s.name for s in SPECS.get(doc_type, [])]
    found = extract(text, doc_type)
    coverage = len(found) / len(expected) if expected else 0.0
    return {
        "fields": found,
        "expected_fields": expected,
        "missing_fields": [f for f in expected if f not in found],
        "coverage": round(coverage, 4),
        # Partial extractions are the dangerous case: a document that
        # looks processed but silently lost its total. Flag rather than
        # pass through.
        "needs_review": coverage < 1.0,
    }
