"""
Knowledge-base corpus for the RAG component.

These passages are plain-language summaries of well-established US tax
and accounting concepts, written for this project. They are NOT
authoritative and are not copied from IRS publications or any
commercial source - the RAG layer is being demonstrated, not the
content. A production deployment would ingest the firm's own licensed
research library and internal memoranda.

Each passage carries an `authority` field. The generator is instructed
to surface it, because in tax work "where does this come from" is not
optional metadata.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Passage:
    doc_id: str
    title: str
    authority: str
    text: str


CORPUS: list[Passage] = [
    Passage(
        "kb_001", "Business meal deduction limits", "IRC §274(n) (general rule)",
        "Ordinary and necessary business meals are generally deductible at 50 percent of "
        "cost when the taxpayer or an employee is present and the expense is not lavish. "
        "The temporary 100 percent deduction for restaurant-provided meals applied only to "
        "2021 and 2022 and has expired. Entertainment expenses are generally nondeductible, "
        "so meals must be separately stated on the invoice when bundled with entertainment, "
        "or the whole amount risks disallowance.",
    ),
    Passage(
        "kb_002", "Nonemployee compensation reporting", "Form 1099-NEC instructions (general)",
        "A payer generally files Form 1099-NEC for each non-corporate service provider paid "
        "600 dollars or more during the calendar year in the course of a trade or business. "
        "The form is due to both the recipient and the IRS by January 31. Payments to "
        "corporations are generally exempt, though attorney fees are a notable exception and "
        "are reportable regardless of entity type.",
    ),
    Passage(
        "kb_003", "Worker classification tests", "Common-law control test",
        "Whether a worker is an employee or an independent contractor turns on the degree of "
        "control the business has over the work. The analysis weighs behavioral control, "
        "financial control, and the nature of the relationship. No single factor is decisive. "
        "Misclassification exposes the payer to back employment taxes, penalties, and interest, "
        "so borderline cases warrant documented analysis rather than default contractor treatment.",
    ),
    Passage(
        "kb_004", "Capitalization versus expensing", "Tangible property regulations (general)",
        "Amounts paid to acquire or produce tangible property are generally capitalized, while "
        "amounts for incidental repairs and maintenance may be expensed. A de minimis safe "
        "harbor lets taxpayers expense items below a per-item threshold if an accounting policy "
        "is in place at the start of the year. The routine maintenance safe harbor covers "
        "recurring activities expected more than once over a defined period.",
    ),
    Passage(
        "kb_005", "Software subscription treatment", "General guidance on SaaS costs",
        "Payments for hosted software accessed by subscription are typically treated as ongoing "
        "service expenses deductible as incurred, rather than as acquisitions of an intangible "
        "asset. Where a contract conveys a license to possess and run software, a different "
        "analysis may apply. Implementation and configuration costs are analyzed separately "
        "from the subscription fee itself.",
    ),
    Passage(
        "kb_006", "Travel expense substantiation", "Substantiation requirements (general)",
        "Deductible travel away from home requires the taxpayer to be away substantially longer "
        "than an ordinary workday and to need sleep or rest. Substantiation must establish the "
        "amount, time, place, and business purpose of each expense. Lodging requires receipts; "
        "per diem methods may substitute for actual cost records for meals and incidental "
        "expenses if the method is applied consistently.",
    ),
    Passage(
        "kb_007", "Accrual versus cash method", "Method of accounting (general)",
        "Under the cash method, income is recognized when received and expenses when paid. "
        "Under the accrual method, income is recognized when the right to receive it is fixed "
        "and the amount determinable, and expenses when the all-events test is met and economic "
        "performance occurs. Larger entities and those holding inventory generally must use the "
        "accrual method, subject to a gross receipts exception for smaller taxpayers.",
    ),
    Passage(
        "kb_008", "Audit engagement independence", "Professional independence standards (general)",
        "An audit firm must be independent in both fact and appearance with respect to an "
        "attest client. Providing certain non-attest services - bookkeeping, valuation of "
        "material amounts, internal audit outsourcing, or management functions - can impair "
        "independence. Safeguards include obtaining management's acknowledgment of "
        "responsibility and ensuring a qualified individual at the client oversees the service.",
    ),
    Passage(
        "kb_009", "Revenue recognition five-step model", "ASC 606 framework (general)",
        "Revenue from contracts with customers is recognized by identifying the contract, "
        "identifying performance obligations, determining the transaction price, allocating "
        "that price to the obligations, and recognizing revenue as each obligation is "
        "satisfied. Variable consideration is estimated and constrained so that a significant "
        "reversal of cumulative revenue is not probable.",
    ),
    Passage(
        "kb_010", "Lease classification", "ASC 842 framework (general)",
        "Lessees recognize a right-of-use asset and a lease liability for most leases with "
        "terms longer than twelve months. Classification as finance or operating determines "
        "expense pattern: finance leases produce front-loaded interest plus amortization, while "
        "operating leases produce a straight-line single expense. Short-term leases may be "
        "exempted by policy election.",
    ),
    Passage(
        "kb_011", "Accountable expense reimbursement plans", "Accountable plan rules (general)",
        "Reimbursements under an accountable plan are excluded from wages if the expense has a "
        "business connection, is substantiated within a reasonable period, and amounts in "
        "excess of substantiated expenses are returned. Failing any of these conditions makes "
        "the plan non-accountable, and reimbursements become taxable wages subject to "
        "withholding and payroll tax reporting.",
    ),
    Passage(
        "kb_012", "Record retention for business records", "General retention practice",
        "Supporting records should generally be retained for at least three years from the "
        "later of the filing date or due date of the return, which corresponds to the ordinary "
        "assessment period. A six-year period applies where gross income is substantially "
        "understated, and no limitation applies to a fraudulent or unfiled return. Records "
        "supporting asset basis should be kept until the asset's disposition period closes.",
    ),
    Passage(
        "kb_013", "Bank fee and merchant charge treatment", "Ordinary and necessary expense",
        "Bank service charges, wire fees, and payment-processor discount fees incurred in a "
        "trade or business are ordinary and necessary expenses deductible when incurred under "
        "the taxpayer's method of accounting. They are recorded gross rather than netted "
        "against revenue, so gross receipts are not understated on the return.",
    ),
    Passage(
        "kb_014", "Advertising versus goodwill costs", "Advertising expense (general)",
        "Ordinary advertising costs aimed at generating current business are deductible when "
        "incurred. Costs producing benefits substantially beyond the year, or that are more "
        "properly characterized as building long-term goodwill or acquiring an asset, may "
        "require capitalization. Routine digital ad spend is generally treated as a current "
        "deduction.",
    ),
    Passage(
        "kb_015", "Continuing professional education costs", "Education expense (general)",
        "Education costs that maintain or improve skills required in the taxpayer's current "
        "trade or business are generally deductible. Costs meeting the minimum requirements to "
        "qualify for a new trade or business are not deductible, even when the employer "
        "requires them. Licensing and professional membership dues follow their own analysis.",
    ),
    Passage(
        "kb_016", "Documentation of related-party transactions", "Related-party disclosure (general)",
        "Transactions between related parties must be identified, disclosed, and evaluated for "
        "whether terms approximate arm's length. Financial statement disclosure covers the "
        "nature of the relationship, a description of the transactions, and amounts involved. "
        "Absence of an arm's-length price does not by itself invalidate a transaction but does "
        "raise the substantiation burden.",
    ),
]


def get_corpus() -> list[Passage]:
    return list(CORPUS)
