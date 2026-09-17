"""
Synthetic accounting document generator.

Real accounting documents contain PII and client-confidential financial
data, so a portfolio project can't use them. This generator produces
labelled, realistic-enough documents to train and evaluate the
classifier and extraction models end-to-end.

Document types mirror what a CPA/advisory firm actually processes:
invoices, receipts, bank statements, W-2s, 1099-NECs, purchase orders,
and engagement letters.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

DocType = Literal[
    "invoice",
    "receipt",
    "bank_statement",
    "w2",
    "1099_nec",
    "purchase_order",
    "engagement_letter",
]

DOC_TYPES: list[DocType] = [
    "invoice",
    "receipt",
    "bank_statement",
    "w2",
    "1099_nec",
    "purchase_order",
    "engagement_letter",
]

VENDORS = [
    "Northwind Traders", "Contoso Ltd", "Fabrikam Inc", "Adventure Works",
    "Tailspin Toys", "Litware Inc", "Proseware Systems", "Wingtip Toys",
    "Blue Yonder Airlines", "Coho Vineyard", "Alpine Ski House", "Lucerne Publishing",
]

CLIENTS = [
    "Redmond Holdings LLC", "Harborview Capital", "Sierra Madre Foods",
    "Pinecrest Dental Group", "Cascade Logistics Co", "Beacon Hill Partners",
]

CITIES = [
    ("Seattle", "WA", "98101"), ("Austin", "TX", "78701"), ("Atlanta", "GA", "30303"),
    ("Denver", "CO", "80202"), ("Boston", "MA", "02108"), ("Phoenix", "AZ", "85004"),
]

LINE_ITEMS = [
    ("Consulting services", 150.0, 350.0),
    ("Software license (annual)", 800.0, 4500.0),
    ("Cloud hosting", 200.0, 1800.0),
    ("Office supplies", 25.0, 400.0),
    ("Professional training", 500.0, 2500.0),
    ("Equipment rental", 300.0, 1500.0),
    ("Marketing retainer", 1000.0, 6000.0),
    ("Legal review", 400.0, 3000.0),
]


@dataclass
class Document:
    """A generated document plus its ground-truth labels."""
    doc_id: str
    doc_type: DocType
    text: str
    entities: dict  # ground truth for the extraction task


def _rand_date(rng: random.Random, start_year: int = 2023) -> date:
    start = date(start_year, 1, 1)
    return start + timedelta(days=rng.randint(0, 900))


def _money(rng: random.Random, low: float, high: float) -> float:
    return round(rng.uniform(low, high), 2)


def _ein(rng: random.Random) -> str:
    return f"{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}"


def _addr(rng: random.Random) -> str:
    city, state, zc = rng.choice(CITIES)
    return f"{rng.randint(100, 9999)} {rng.choice(['Main', 'Oak', 'Pine', 'Cedar', 'Elm'])} St, {city}, {state} {zc}"


def _gen_invoice(rng: random.Random, idx: int) -> Document:
    vendor = rng.choice(VENDORS)
    client = rng.choice(CLIENTS)
    issued = _rand_date(rng)
    due = issued + timedelta(days=rng.choice([15, 30, 45, 60]))
    inv_no = f"INV-{rng.randint(10000, 99999)}"

    items, subtotal = [], 0.0
    for _ in range(rng.randint(1, 4)):
        desc, lo, hi = rng.choice(LINE_ITEMS)
        qty = rng.randint(1, 5)
        unit = _money(rng, lo, hi)
        amt = round(qty * unit, 2)
        subtotal += amt
        items.append(f"  {desc:<32} {qty:>3} x ${unit:>10,.2f}   ${amt:>12,.2f}")

    subtotal = round(subtotal, 2)
    tax_rate = rng.choice([0.0, 0.06, 0.0825, 0.095])
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)

    text = f"""INVOICE

{vendor}
{_addr(rng)}
EIN: {_ein(rng)}

Bill To:
{client}
{_addr(rng)}

Invoice Number: {inv_no}
Invoice Date: {issued.isoformat()}
Due Date: {due.isoformat()}
Payment Terms: Net {(due - issued).days}

Description                          Qty        Unit Price        Amount
{chr(10).join(items)}

                                       Subtotal:   ${subtotal:>12,.2f}
                                    Tax ({tax_rate:.2%}):   ${tax:>12,.2f}
                                    TOTAL DUE:      ${total:>12,.2f}

Remit payment to the address above. Late payments accrue 1.5% monthly interest.
"""
    return Document(
        doc_id=f"doc_{idx:05d}",
        doc_type="invoice",
        text=text,
        entities={
            "vendor_name": vendor,
            "client_name": client,
            "document_number": inv_no,
            "issue_date": issued.isoformat(),
            "due_date": due.isoformat(),
            "subtotal": subtotal,
            "tax_amount": tax,
            "total_amount": total,
        },
    )


def _gen_receipt(rng: random.Random, idx: int) -> Document:
    vendor = rng.choice(VENDORS)
    issued = _rand_date(rng)
    subtotal = _money(rng, 8.0, 480.0)
    tax = round(subtotal * rng.choice([0.06, 0.0825, 0.095]), 2)
    total = round(subtotal + tax, 2)
    txn = f"TXN{rng.randint(100000, 999999)}"

    text = f"""{vendor.upper()}
{_addr(rng)}

*** CUSTOMER RECEIPT ***

Date: {issued.isoformat()}  Time: {rng.randint(8,20):02d}:{rng.randint(0,59):02d}
Transaction: {txn}
Register: {rng.randint(1, 12)}   Cashier: {rng.choice(['A. Reyes','M. Cruz','J. Tan','L. Santos'])}

Subtotal              ${subtotal:>8,.2f}
Sales Tax             ${tax:>8,.2f}
TOTAL                 ${total:>8,.2f}

Paid by: {rng.choice(['VISA ****4412', 'MASTERCARD ****8890', 'AMEX ****3001', 'CASH'])}

Thank you for your business!
Returns accepted within 30 days with receipt.
"""
    return Document(
        doc_id=f"doc_{idx:05d}",
        doc_type="receipt",
        text=text,
        entities={
            "vendor_name": vendor,
            "document_number": txn,
            "issue_date": issued.isoformat(),
            "subtotal": subtotal,
            "tax_amount": tax,
            "total_amount": total,
        },
    )


def _gen_bank_statement(rng: random.Random, idx: int) -> Document:
    client = rng.choice(CLIENTS)
    bank = rng.choice(["First Cascade Bank", "Meridian National", "Summit Trust Bank"])
    period_end = _rand_date(rng)
    period_start = period_end - timedelta(days=30)
    opening = _money(rng, 5000.0, 200000.0)

    lines, balance = [], opening
    for _ in range(rng.randint(5, 10)):
        d = period_start + timedelta(days=rng.randint(0, 30))
        desc = rng.choice(["ACH DEPOSIT", "WIRE TRANSFER OUT", "CHECK #" + str(rng.randint(1000, 9999)),
                           "CARD PURCHASE", "PAYROLL DEBIT", "VENDOR PAYMENT"])
        amt = _money(rng, 50.0, 15000.0) * (1 if "DEPOSIT" in desc else -1)
        balance = round(balance + amt, 2)
        lines.append(f"{d.isoformat()}  {desc:<24} {amt:>12,.2f}  {balance:>14,.2f}")

    text = f"""{bank}
STATEMENT OF ACCOUNT

Account Holder: {client}
Account Number: ****{rng.randint(1000, 9999)}
Statement Period: {period_start.isoformat()} to {period_end.isoformat()}

Opening Balance: ${opening:,.2f}

Date        Description              Amount        Balance
{chr(10).join(lines)}

Closing Balance: ${balance:,.2f}

Please report any discrepancies within 60 days of the statement date.
Member FDIC.
"""
    return Document(
        doc_id=f"doc_{idx:05d}",
        doc_type="bank_statement",
        text=text,
        entities={
            "client_name": client,
            "vendor_name": bank,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "opening_balance": opening,
            "closing_balance": balance,
        },
    )


def _gen_w2(rng: random.Random, idx: int) -> Document:
    employer = rng.choice(VENDORS)
    employee = rng.choice(["Maria Santos", "James Delgado", "Priya Nair", "Tomas Reyes", "Ana Villanueva"])
    year = rng.choice([2023, 2024, 2025])
    wages = _money(rng, 42000.0, 185000.0)
    fed = round(wages * rng.uniform(0.10, 0.24), 2)
    ss = round(min(wages, 168600) * 0.062, 2)
    medicare = round(wages * 0.0145, 2)

    text = f"""Form W-2  Wage and Tax Statement                    {year}
Department of the Treasury - Internal Revenue Service

a  Employee's social security number: ***-**-{rng.randint(1000,9999)}
b  Employer identification number (EIN): {_ein(rng)}

c  Employer's name, address, and ZIP code:
   {employer}
   {_addr(rng)}

e  Employee's name: {employee}
f  Employee's address: {_addr(rng)}

 1 Wages, tips, other compensation .......... {wages:>12,.2f}
 2 Federal income tax withheld .............. {fed:>12,.2f}
 3 Social security wages .................... {min(wages, 168600):>12,.2f}
 4 Social security tax withheld ............. {ss:>12,.2f}
 5 Medicare wages and tips .................. {wages:>12,.2f}
 6 Medicare tax withheld .................... {medicare:>12,.2f}

Copy B - To Be Filed With Employee's FEDERAL Tax Return.
This information is being furnished to the Internal Revenue Service.
"""
    return Document(
        doc_id=f"doc_{idx:05d}",
        doc_type="w2",
        text=text,
        entities={
            "employer_name": employer,
            "employee_name": employee,
            "tax_year": year,
            "wages": wages,
            "federal_tax_withheld": fed,
        },
    )


def _gen_1099(rng: random.Random, idx: int) -> Document:
    payer = rng.choice(VENDORS)
    recipient = rng.choice(CLIENTS)
    year = rng.choice([2023, 2024, 2025])
    comp = _money(rng, 1200.0, 95000.0)

    text = f"""Form 1099-NEC  Nonemployee Compensation              {year}
Department of the Treasury - Internal Revenue Service

PAYER'S name, address, ZIP:
  {payer}
  {_addr(rng)}
PAYER'S TIN: {_ein(rng)}

RECIPIENT'S name:
  {recipient}
  {_addr(rng)}
RECIPIENT'S TIN: **-***{rng.randint(1000,9999)}

 1 Nonemployee compensation ................. {comp:>12,.2f}
 4 Federal income tax withheld .............. {0.00:>12,.2f}

This is important tax information and is being furnished to the IRS.
If you are required to file a return, a negligence penalty or other
sanction may be imposed on you if this income is taxable and the IRS
determines that it has not been reported.
"""
    return Document(
        doc_id=f"doc_{idx:05d}",
        doc_type="1099_nec",
        text=text,
        entities={
            "payer_name": payer,
            "recipient_name": recipient,
            "tax_year": year,
            "nonemployee_compensation": comp,
        },
    )


def _gen_purchase_order(rng: random.Random, idx: int) -> Document:
    buyer = rng.choice(CLIENTS)
    supplier = rng.choice(VENDORS)
    issued = _rand_date(rng)
    po_no = f"PO-{rng.randint(10000, 99999)}"
    items, total = [], 0.0
    for _ in range(rng.randint(1, 4)):
        desc, lo, hi = rng.choice(LINE_ITEMS)
        qty = rng.randint(1, 20)
        unit = _money(rng, lo, hi)
        amt = round(qty * unit, 2)
        total += amt
        items.append(f"  {desc:<32} {qty:>3}   ${unit:>10,.2f}   ${amt:>12,.2f}")
    total = round(total, 2)

    text = f"""PURCHASE ORDER

Buyer: {buyer}
       {_addr(rng)}

Supplier: {supplier}
          {_addr(rng)}

PO Number: {po_no}
PO Date: {issued.isoformat()}
Requested Delivery: {(issued + timedelta(days=rng.randint(7, 45))).isoformat()}
Ship Via: {rng.choice(['Ground', 'Air Freight', 'LTL Carrier'])}

Item                                 Qty    Unit Price         Amount
{chr(10).join(items)}

                                          ORDER TOTAL:  ${total:>12,.2f}

This purchase order is subject to the terms and conditions agreed between
the parties. Reference the PO number on all invoices and packing slips.
"""
    return Document(
        doc_id=f"doc_{idx:05d}",
        doc_type="purchase_order",
        text=text,
        entities={
            "client_name": buyer,
            "vendor_name": supplier,
            "document_number": po_no,
            "issue_date": issued.isoformat(),
            "total_amount": total,
        },
    )


def _gen_engagement_letter(rng: random.Random, idx: int) -> Document:
    client = rng.choice(CLIENTS)
    issued = _rand_date(rng)
    fee = _money(rng, 8000.0, 120000.0)
    service = rng.choice([
        "annual financial statement audit",
        "federal and state income tax preparation",
        "quarterly review engagement",
        "transfer pricing advisory",
        "internal controls assessment",
    ])

    text = f"""ENGAGEMENT LETTER

{issued.isoformat()}

{client}
{_addr(rng)}

Dear Client,

This letter confirms our understanding of the terms of our engagement and
the nature and limitations of the services we will provide.

Scope of Services
We will perform the {service} for the period ending {(issued + timedelta(days=365)).isoformat()}.
Our engagement will be conducted in accordance with applicable professional
standards. Our services will not constitute an examination or review of
matters outside the agreed scope.

Client Responsibilities
Management is responsible for the accuracy and completeness of the records
and information provided, for adopting sound accounting policies, and for
establishing and maintaining internal control.

Fees
Our estimated professional fee for this engagement is ${fee:,.2f}, billed
monthly as work progresses. This estimate assumes records are in good
condition and that assistance is provided as agreed.

Please sign and return the enclosed copy to indicate your agreement.

Very truly yours,
Engagement Partner
"""
    return Document(
        doc_id=f"doc_{idx:05d}",
        doc_type="engagement_letter",
        text=text,
        entities={
            "client_name": client,
            "issue_date": issued.isoformat(),
            "engagement_fee": fee,
            "service_type": service,
        },
    )


_GENERATORS = {
    "invoice": _gen_invoice,
    "receipt": _gen_receipt,
    "bank_statement": _gen_bank_statement,
    "w2": _gen_w2,
    "1099_nec": _gen_1099,
    "purchase_order": _gen_purchase_order,
    "engagement_letter": _gen_engagement_letter,
}


def generate_dataset(n: int = 1400, seed: int = 42) -> list[Document]:
    """Generate a class-balanced dataset of labelled documents."""
    rng = random.Random(seed)
    docs: list[Document] = []
    per_type = n // len(DOC_TYPES)
    idx = 0
    for doc_type in DOC_TYPES:
        for _ in range(per_type):
            docs.append(_GENERATORS[doc_type](rng, idx))
            idx += 1
    rng.shuffle(docs)
    return docs


def write_jsonl(docs: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d)) + "\n")


def read_jsonl(path: Path) -> list[Document]:
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            docs.append(Document(**json.loads(line)))
    return docs


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "data" / "documents.jsonl"
    dataset = generate_dataset()
    write_jsonl(dataset, out)
    print(f"Wrote {len(dataset)} documents to {out}")
