"""
Realistic-intake transformations.

The OCR noise in `noise.py` wasn't enough to make classification a real
problem: every generated type carries a unique title line, so the model
was keyword-matching at ~1.00 macro-F1 even under heavy degradation.
That is a benchmark artifact, not a result.

Two transformations close the gap to production:

1. Terminology variation. Vendors don't standardise. An invoice may
   be titled "Tax Invoice", "Bill", or "Statement of Charges"; a
   receipt may say "Sales Slip". The model shouldn't depend on one
   canonical string.

2. Header cropping. Scans routinely lose the top of the page, so the
   title block is simply absent and the type has to be inferred from
   body structure (line items vs withholding boxes vs running
   balance). This is where the task gets genuinely hard, and where
   invoice / purchase_order / receipt become truly confusable.
"""
from __future__ import annotations

import random

# Real-world title variants per document type.
TITLE_VARIANTS: dict[str, list[str]] = {
    "INVOICE": ["INVOICE", "TAX INVOICE", "BILL", "STATEMENT OF CHARGES",
                "SALES INVOICE", "REQUEST FOR PAYMENT"],
    "*** CUSTOMER RECEIPT ***": ["*** CUSTOMER RECEIPT ***", "SALES SLIP",
                                 "PAYMENT RECEIPT", "-- RECEIPT --", "PROOF OF PURCHASE"],
    "STATEMENT OF ACCOUNT": ["STATEMENT OF ACCOUNT", "ACCOUNT STATEMENT",
                             "MONTHLY STATEMENT", "PERIODIC STATEMENT"],
    "PURCHASE ORDER": ["PURCHASE ORDER", "P.O.", "ORDER FORM",
                       "PROCUREMENT ORDER", "SUPPLY ORDER"],
    "ENGAGEMENT LETTER": ["ENGAGEMENT LETTER", "LETTER OF ENGAGEMENT",
                          "TERMS OF ENGAGEMENT", "SERVICE AGREEMENT LETTER"],
}


def vary_terminology(text: str, rng: random.Random) -> str:
    """Swap canonical titles for real-world synonyms."""
    for canonical, variants in TITLE_VARIANTS.items():
        if canonical in text:
            return text.replace(canonical, rng.choice(variants), 1)
    return text


def crop_header(text: str, n_lines: int) -> str:
    """Drop the leading lines, simulating a scan that lost the page top."""
    lines = text.split("\n")
    # Skip leading blanks so we crop actual content, not whitespace.
    start = 0
    dropped = 0
    while start < len(lines) and dropped < n_lines:
        if lines[start].strip():
            dropped += 1
        start += 1
    remainder = "\n".join(lines[start:])
    return remainder if remainder.strip() else text


def realistic_intake(text: str, seed: int | None = None, crop_rate: float = 0.35) -> str:
    """Apply terminology variation and probabilistic header cropping.

    `crop_rate` is the fraction of documents arriving without a usable
    title block. 0.35 is deliberately aggressive - it forces the model
    to learn body structure instead of leaning on the header, which is
    what makes the resulting metric trustworthy.
    """
    rng = random.Random(seed)
    text = vary_terminology(text, rng)
    if rng.random() < crop_rate:
        text = crop_header(text, rng.randint(2, 6))
    return text
