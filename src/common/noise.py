"""
OCR degradation simulator.

Why this module exists: on clean generated templates the classifier
scores 100% macro-F1, which tells you nothing - the templates are
trivially separable by keyword. Real documents at a CPA firm arrive
as scanned PDFs pushed through OCR, so they carry character
substitutions, dropped lines, broken layout, and truncation.

Evaluating on degraded text is what makes the benchmark meaningful.
The degradation levels below are modelled on common OCR failure
modes rather than random character noise:

  - visually-confusable substitutions (0/O, 1/l/I, 5/S, 8/B, rn/m)
  - whitespace collapse and column misalignment
  - dropped lines (page skew, faint scan regions)
  - truncation (partial fax / cut page)
  - stray artifacts from speckle
"""
from __future__ import annotations

import random

# Visually confusable pairs that real OCR engines actually mix up.
CONFUSIONS: list[tuple[str, str]] = [
    ("0", "O"), ("O", "0"), ("1", "l"), ("l", "1"), ("I", "1"),
    ("5", "S"), ("S", "5"), ("8", "B"), ("B", "8"), ("2", "Z"),
    ("rn", "m"), ("m", "rn"), ("cl", "d"), ("vv", "w"),
    ("$", "S"), (",", "."), (".", ","),
]

SPECKLE = ["`", "'", ".", ",", "~", "^", "|"]


def _substitute(text: str, rate: float, rng: random.Random) -> str:
    """Apply visually-confusable character substitutions."""
    if rate <= 0:
        return text
    chars = list(text)
    out: list[str] = []
    i = 0
    while i < len(chars):
        applied = False
        if rng.random() < rate:
            for src, dst in CONFUSIONS:
                if text.startswith(src, i):
                    out.append(dst)
                    i += len(src)
                    applied = True
                    break
        if not applied:
            out.append(chars[i])
            i += 1
    return "".join(out)


def _drop_lines(text: str, rate: float, rng: random.Random) -> str:
    """Simulate lines lost to skew or faint scanning."""
    lines = text.split("\n")
    kept = [ln for ln in lines if rng.random() >= rate]
    return "\n".join(kept) if kept else text


def _collapse_whitespace(text: str, rate: float, rng: random.Random) -> str:
    """Simulate lost column alignment in OCR output."""
    lines = []
    for ln in text.split("\n"):
        if rng.random() < rate:
            ln = " ".join(ln.split())
        lines.append(ln)
    return "\n".join(lines)


def _truncate(text: str, rate: float, rng: random.Random) -> str:
    """Simulate a partially captured page."""
    if rng.random() >= rate:
        return text
    keep = rng.uniform(0.35, 0.8)
    cut = int(len(text) * keep)
    return text[:cut]


def _speckle(text: str, rate: float, rng: random.Random) -> str:
    """Insert scanner speckle artifacts."""
    if rate <= 0:
        return text
    out = []
    for ch in text:
        out.append(ch)
        if rng.random() < rate:
            out.append(rng.choice(SPECKLE))
    return "".join(out)


# Degradation presets. "light" is a good modern scanner; "heavy" is a
# faxed photocopy, which CPA firms genuinely still receive.
LEVELS: dict[str, dict[str, float]] = {
    "clean": {"sub": 0.0, "drop": 0.0, "collapse": 0.0, "truncate": 0.0, "speckle": 0.0},
    "light": {"sub": 0.02, "drop": 0.02, "collapse": 0.15, "truncate": 0.05, "speckle": 0.002},
    "medium": {"sub": 0.06, "drop": 0.08, "collapse": 0.40, "truncate": 0.15, "speckle": 0.008},
    "heavy": {"sub": 0.12, "drop": 0.18, "collapse": 0.70, "truncate": 0.30, "speckle": 0.020},
}


def degrade(text: str, level: str = "medium", seed: int | None = None) -> str:
    """Apply OCR-style degradation at the named level."""
    if level not in LEVELS:
        raise ValueError(f"Unknown level {level!r}; expected one of {list(LEVELS)}")
    cfg = LEVELS[level]
    rng = random.Random(seed)

    text = _drop_lines(text, cfg["drop"], rng)
    text = _substitute(text, cfg["sub"], rng)
    text = _collapse_whitespace(text, cfg["collapse"], rng)
    text = _speckle(text, cfg["speckle"], rng)
    text = _truncate(text, cfg["truncate"], rng)
    return text


def degrade_mixed(text: str, seed: int | None = None) -> str:
    """Sample a degradation level to mimic a realistic intake queue.

    Most documents are decent; a meaningful minority are bad. The
    mixture below is the default for training and evaluation because
    a model that only ever saw clean text degrades badly in production.
    """
    rng = random.Random(seed)
    level = rng.choices(
        ["clean", "light", "medium", "heavy"],
        weights=[0.15, 0.40, 0.30, 0.15],
    )[0]
    return degrade(text, level, seed=rng.randint(0, 10**9))
