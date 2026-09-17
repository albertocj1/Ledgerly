"""
General-ledger account categorization.

This is the project's core ML task, and the reason the document-type
classifier is treated as a cheap routing stage rather than the
headline result: classifying a W-2 vs a bank statement is nearly
free (distinct boilerplate, ~1.00 F1 even under heavy OCR noise).
Saturated benchmarks aren't worth reporting as achievements.

Mapping a transaction line to a GL account is the task that's
actually hard, and it's the one accounting firms spend real money on:

  - many classes (20 accounts across the standard chart)
  - terse, abbreviated memo text ("AMZN MKTP US*2H4TY", "UBER TRIP")
  - genuine ambiguity: the same vendor maps to different accounts
    depending on context (Amazon -> office supplies OR computer
    equipment OR books/subscriptions)
  - heavy class imbalance, mirroring a real ledger
  - overlapping vocabulary between related accounts (Travel vs Meals,
    Software vs Computer Equipment)

That ambiguity is intentional and irreducible - it puts a ceiling on
achievable accuracy well below 100%, which is what makes the metric
informative.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

# Standard small-business chart of accounts.
GL_ACCOUNTS = [
    "6010_Office_Supplies",
    "6020_Computer_Equipment",
    "6030_Software_Subscriptions",
    "6040_Travel_Airfare",
    "6050_Travel_Lodging",
    "6060_Travel_Ground",
    "6070_Meals_Entertainment",
    "6080_Professional_Fees",
    "6090_Legal_Fees",
    "6100_Marketing_Advertising",
    "6110_Rent_Facilities",
    "6120_Utilities",
    "6130_Telecom",
    "6140_Insurance",
    "6150_Payroll_Wages",
    "6160_Payroll_Taxes",
    "6170_Training_Education",
    "6180_Bank_Fees",
    "6190_Shipping_Postage",
    "6200_Repairs_Maintenance",
]

# Vendor -> plausible accounts, with weights. Multi-account vendors are
# the source of irreducible ambiguity.
VENDOR_MAP: dict[str, list[tuple[str, float]]] = {
    "AMZN MKTP US": [("6010_Office_Supplies", 0.5), ("6020_Computer_Equipment", 0.3), ("6030_Software_Subscriptions", 0.2)],
    "AMAZON WEB SERVICES": [("6030_Software_Subscriptions", 1.0)],
    "STAPLES": [("6010_Office_Supplies", 0.9), ("6020_Computer_Equipment", 0.1)],
    "BEST BUY": [("6020_Computer_Equipment", 0.8), ("6010_Office_Supplies", 0.2)],
    "APPLE STORE": [("6020_Computer_Equipment", 0.85), ("6030_Software_Subscriptions", 0.15)],
    "MICROSOFT 365": [("6030_Software_Subscriptions", 1.0)],
    "ADOBE CREATIVE CLOUD": [("6030_Software_Subscriptions", 1.0)],
    "SLACK TECHNOLOGIES": [("6030_Software_Subscriptions", 1.0)],
    "GITHUB": [("6030_Software_Subscriptions", 1.0)],
    "DELTA AIR LINES": [("6040_Travel_Airfare", 1.0)],
    "UNITED AIRLINES": [("6040_Travel_Airfare", 1.0)],
    "PHILIPPINE AIRLINES": [("6040_Travel_Airfare", 1.0)],
    "MARRIOTT": [("6050_Travel_Lodging", 0.9), ("6070_Meals_Entertainment", 0.1)],
    "HILTON HOTELS": [("6050_Travel_Lodging", 0.9), ("6070_Meals_Entertainment", 0.1)],
    "AIRBNB": [("6050_Travel_Lodging", 1.0)],
    "UBER TRIP": [("6060_Travel_Ground", 0.85), ("6070_Meals_Entertainment", 0.15)],
    "UBER EATS": [("6070_Meals_Entertainment", 1.0)],
    "LYFT": [("6060_Travel_Ground", 1.0)],
    "GRAB": [("6060_Travel_Ground", 0.7), ("6070_Meals_Entertainment", 0.3)],
    "HERTZ RENT A CAR": [("6060_Travel_Ground", 1.0)],
    "STARBUCKS": [("6070_Meals_Entertainment", 1.0)],
    "PANERA BREAD": [("6070_Meals_Entertainment", 1.0)],
    "DOORDASH": [("6070_Meals_Entertainment", 1.0)],
    "DELOITTE ADVISORY": [("6080_Professional_Fees", 1.0)],
    "KPMG CONSULTING": [("6080_Professional_Fees", 1.0)],
    "BAKER MCKENZIE LLP": [("6090_Legal_Fees", 1.0)],
    "JONES DAY LLP": [("6090_Legal_Fees", 1.0)],
    "GOOGLE ADS": [("6100_Marketing_Advertising", 1.0)],
    "META PLATFORMS ADS": [("6100_Marketing_Advertising", 1.0)],
    "LINKEDIN ADS": [("6100_Marketing_Advertising", 0.8), ("6170_Training_Education", 0.2)],
    "WEWORK": [("6110_Rent_Facilities", 1.0)],
    "REGUS OFFICE": [("6110_Rent_Facilities", 1.0)],
    "PACIFIC GAS ELECTRIC": [("6120_Utilities", 1.0)],
    "MERALCO": [("6120_Utilities", 1.0)],
    "COMCAST BUSINESS": [("6130_Telecom", 0.7), ("6120_Utilities", 0.3)],
    "VERIZON WIRELESS": [("6130_Telecom", 1.0)],
    "PLDT ENTERPRISE": [("6130_Telecom", 1.0)],
    "STATE FARM INSURANCE": [("6140_Insurance", 1.0)],
    "CHUBB LIMITED": [("6140_Insurance", 1.0)],
    "ADP PAYROLL": [("6150_Payroll_Wages", 0.6), ("6160_Payroll_Taxes", 0.4)],
    "GUSTO PAYROLL": [("6150_Payroll_Wages", 0.6), ("6160_Payroll_Taxes", 0.4)],
    "IRS EFTPS": [("6160_Payroll_Taxes", 1.0)],
    "COURSERA": [("6170_Training_Education", 1.0)],
    "UDEMY": [("6170_Training_Education", 1.0)],
    "AICPA MEMBERSHIP": [("6170_Training_Education", 0.7), ("6080_Professional_Fees", 0.3)],
    "WELLS FARGO SVC CHG": [("6180_Bank_Fees", 1.0)],
    "WIRE TRANSFER FEE": [("6180_Bank_Fees", 1.0)],
    "STRIPE FEE": [("6180_Bank_Fees", 1.0)],
    "FEDEX": [("6190_Shipping_Postage", 1.0)],
    "UPS STORE": [("6190_Shipping_Postage", 0.8), ("6010_Office_Supplies", 0.2)],
    "USPS POSTAGE": [("6190_Shipping_Postage", 1.0)],
    "HOME DEPOT": [("6200_Repairs_Maintenance", 0.7), ("6010_Office_Supplies", 0.3)],
    "ACME HVAC SERVICE": [("6200_Repairs_Maintenance", 1.0)],
}

# Memo fragments appended to vendor strings, the way card processors do.
MEMO_SUFFIXES = [
    "", "", "", "  #{n}", " *{code}", " {city}", " RECURRING", " AUTOPAY",
    " POS DEBIT", " CARD {last4}", " REF{n}", " {city} {state}",
]

CITIES = ["SEATTLE", "AUSTIN", "ATLANTA", "DENVER", "BOSTON", "MANILA", "MAKATI", "CEBU"]
STATES = ["WA", "TX", "GA", "CO", "MA", "PH"]
CODES = ["2H4TY", "9KLM1", "QQ72B", "8XZP4", "7TT19"]

# Class frequency weights - real ledgers are heavily imbalanced.
ACCOUNT_FREQUENCY = {
    "6010_Office_Supplies": 1.4, "6020_Computer_Equipment": 0.7,
    "6030_Software_Subscriptions": 1.6, "6040_Travel_Airfare": 0.6,
    "6050_Travel_Lodging": 0.5, "6060_Travel_Ground": 1.1,
    "6070_Meals_Entertainment": 1.8, "6080_Professional_Fees": 0.5,
    "6090_Legal_Fees": 0.3, "6100_Marketing_Advertising": 0.9,
    "6110_Rent_Facilities": 0.3, "6120_Utilities": 0.5,
    "6130_Telecom": 0.6, "6140_Insurance": 0.3,
    "6150_Payroll_Wages": 0.4, "6160_Payroll_Taxes": 0.35,
    "6170_Training_Education": 0.5, "6180_Bank_Fees": 1.2,
    "6190_Shipping_Postage": 0.8, "6200_Repairs_Maintenance": 0.4,
}


@dataclass
class Transaction:
    memo: str
    amount: float
    gl_account: str


def _render_memo(vendor: str, rng: random.Random) -> str:
    suffix = rng.choice(MEMO_SUFFIXES)
    suffix = (
        suffix.replace("{n}", str(rng.randint(1000, 99999)))
        .replace("{code}", rng.choice(CODES))
        .replace("{city}", rng.choice(CITIES))
        .replace("{state}", rng.choice(STATES))
        .replace("{last4}", str(rng.randint(1000, 9999)))
    )
    memo = vendor + suffix
    # Card networks truncate; so do bank feeds.
    if rng.random() < 0.15:
        memo = memo[: rng.randint(12, len(memo))] if len(memo) > 12 else memo
    if rng.random() < 0.1:
        memo = memo.lower()
    return memo.strip()


def generate_transactions(n: int = 6000, seed: int = 7) -> list[Transaction]:
    """Generate an imbalanced, ambiguous transaction ledger."""
    rng = random.Random(seed)

    # Weight vendor sampling by the frequency of the accounts it maps to.
    vendors = list(VENDOR_MAP)
    weights = []
    for v in vendors:
        w = sum(ACCOUNT_FREQUENCY.get(acct, 1.0) * p for acct, p in VENDOR_MAP[v])
        weights.append(w)

    txns: list[Transaction] = []
    for _ in range(n):
        vendor = rng.choices(vendors, weights=weights)[0]
        options = VENDOR_MAP[vendor]
        accounts = [a for a, _ in options]
        probs = [p for _, p in options]
        account = rng.choices(accounts, weights=probs)[0]
        txns.append(
            Transaction(
                memo=_render_memo(vendor, rng),
                amount=round(rng.uniform(4.0, 9500.0), 2),
                gl_account=account,
            )
        )
    return txns


def write_jsonl(txns: list[Transaction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in txns:
            f.write(json.dumps(asdict(t)) + "\n")


def read_jsonl(path: Path) -> list[Transaction]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            out.append(Transaction(**json.loads(line)))
    return out


def theoretical_ceiling(seed: int = 7, n: int = 200_000) -> float:
    """Estimate the Bayes-optimal accuracy given vendor ambiguity.

    Even a perfect model can't beat always predicting each vendor's
    most likely account, because the memo text carries no further
    signal about which one applies. Reporting this alongside model
    accuracy is what turns a bare number into an interpretable one.
    """
    rng = random.Random(seed)
    vendors = list(VENDOR_MAP)
    weights = [
        sum(ACCOUNT_FREQUENCY.get(a, 1.0) * p for a, p in VENDOR_MAP[v]) for v in vendors
    ]
    hits = 0
    for _ in range(n):
        v = rng.choices(vendors, weights=weights)[0]
        options = VENDOR_MAP[v]
        probs = [p for _, p in options]
        total = sum(probs)
        hits += max(probs) / total
    return hits / n


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "data" / "transactions.jsonl"
    data = generate_transactions()
    write_jsonl(data, out)
    print(f"Wrote {len(data)} transactions to {out}")
    print(f"Estimated Bayes ceiling (accuracy): {theoretical_ceiling():.4f}")
