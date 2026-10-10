"""Helpers for preparing trustworthy financial and transcript evidence exports."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Iterable

# Exact SEC company names prevent substring collisions (e.g. "AMD" in Amdocs).
SEC_COMPANIES = {
    "MICROSOFT CORP": "Microsoft",
    "ALPHABET INC.": "Alphabet",
    "AMAZON COM INC": "Amazon",
    "NVIDIA CORP": "NVIDIA",
    "META PLATFORMS, INC.": "Meta",
    "ADVANCED MICRO DEVICES INC": "AMD",
    "INTEL CORP": "Intel",
    "SALESFORCE, INC.": "Salesforce",
    "INTERNATIONAL BUSINESS MACHINES CORP": "IBM",
    "CISCO SYSTEMS, INC.": "Cisco",
}

# Ordered exact tags: prefer the standard consolidated concept, not substring matches.
METRIC_TAGS = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ),
    "net_income": (
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ),
    "operating_income": ("OperatingIncomeLoss",),
    "gross_profit": ("GrossProfit",),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "eps_diluted": ("EarningsPerShareDiluted",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
}
TAG_TO_METRIC = {
    tag: (metric, priority)
    for metric, tags in METRIC_TAGS.items()
    for priority, tag in enumerate(tags)
}

_TRANSCRIPT_ALIASES = {
    "alphabet": "Alphabet",
    "microsoft": "Microsoft",
    "amzn": "Amazon",
    "amazon": "Amazon",
    "nvda": "NVIDIA",
    "nvidia": "NVIDIA",
    "meta": "Meta",
    "intel": "Intel",
    "cisco": "Cisco",
    "salesforce": "Salesforce",
    "amd": "AMD",
    "ibm": "IBM",
}


def _period_key(value: str) -> tuple[int, int, int] | None:
    """Parse SEC's YYYY-MM-DD period-end field without trusting the filed column."""
    try:
        parsed = date.fromisoformat(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed.year, parsed.month, parsed.day


def latest_filing_by_company(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Return each company's latest filing using statement period-end, not filed timestamp."""
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        company = (row.get("company") or "").strip()
        period = _period_key(row.get("ddate", ""))
        form = (row.get("form") or "").strip().upper()
        adsh = (row.get("adsh") or "").strip()
        if not company or period is None or form not in {"10-Q", "10-K"} or not adsh:
            continue
        # Prefer a 10-K over a 10-Q only when both cover the same period-end.
        rank = (period, 1 if form == "10-K" else 0, adsh)
        current = selected.get(company)
        current_period = _period_key(current.get("ddate", "")) if current else None
        current_rank = (
            current_period,
            1 if current and current.get("form", "").upper() == "10-K" else 0,
            current.get("adsh", "") if current else "",
        )
        if current is None or rank > current_rank:
            selected[company] = row
    return selected


def normalize_transcript_company(label: str | None) -> str | None:
    """Normalize only recognized company aliases; reject noisy extracted labels."""
    if not label:
        return None
    return _TRANSCRIPT_ALIASES.get(label.strip().casefold())


def select_consolidated_facts(
    rows: Iterable[dict[str, str]], company_map: dict[str, str]
) -> list[dict[str, object]]:
    """Select exact, unsegmented consolidated facts and drop ambiguous duplicates."""
    # Normalize legal-name lookup but keep matching exact.
    names = {name.strip().casefold(): company for name, company in company_map.items()}
    candidates: dict[tuple[str, str, str, str, str, str], list[tuple[int, float, dict[str, str]]]] = defaultdict(list)

    for row in rows:
        company = names.get((row.get("name") or "").strip().casefold())
        tag = (row.get("tag") or "").strip()
        if company is None or tag not in TAG_TO_METRIC:
            continue
        if (row.get("form") or "").strip().upper() not in {"10-Q", "10-K"}:
            continue
        if (row.get("segments") or "").strip() or (row.get("coreg") or "").strip():
            continue
        unit = (row.get("uom") or "").strip()
        if not unit:
            continue
        try:
            value = float(row.get("value", ""))
        except (TypeError, ValueError):
            continue
        if not _period_key(row.get("ddate", "")):
            continue
        metric, priority = TAG_TO_METRIC[tag]
        key = (
            company,
            metric,
            row.get("adsh", "").strip(),
            row.get("ddate", "").strip(),
            row.get("qtrs", "").strip(),
            unit,
        )
        candidates[key].append((priority, value, row))

    facts: list[dict[str, object]] = []
    for key, options in candidates.items():
        best_priority = min(option[0] for option in options)
        best = [option for option in options if option[0] == best_priority]
        values = {option[1] for option in best}
        # Conflicting consolidated rows are not silently resolved by row order.
        if len(values) != 1:
            continue
        _, value, source = best[0]
        company, metric, adsh, period_end, qtrs, unit = key
        facts.append(
            {
                "company": company,
                "adsh": adsh,
                "period_end": period_end,
                "form": source.get("form", "").strip().upper(),
                "period_quarters": qtrs,
                "metric": metric,
                "value": value,
                "unit": unit,
                "tag": source.get("tag", "").strip(),
                "cik": source.get("cik", "").strip(),
            }
        )
    facts.sort(key=lambda r: (str(r["company"]), str(r["metric"]), str(r["period_end"]), str(r["period_quarters"])))
    return facts
