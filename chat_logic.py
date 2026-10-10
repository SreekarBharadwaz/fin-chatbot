"""Grounded retrieval and context formatting for the financial-insights chatbot."""

from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterable

FLOW_METRICS = {
    "revenue", "net_income", "operating_income", "gross_profit", "eps_diluted",
    "operating_cash_flow",
}
BALANCE_METRICS = {"assets", "liabilities", "cash"}
METRIC_LABELS = {
    "revenue": "Revenue",
    "net_income": "Net income",
    "operating_income": "Operating income",
    "gross_profit": "Gross profit",
    "assets": "Total assets",
    "liabilities": "Total liabilities",
    "cash": "Cash and cash equivalents",
    "eps_diluted": "Diluted EPS",
    "operating_cash_flow": "Operating cash flow",
}
STOPWORDS = {
    "about", "after", "also", "and", "are", "but", "can", "company", "compare",
    "could", "does", "for", "from", "give", "how", "into", "its", "latest", "more",
    "most", "now", "please", "show", "that", "the", "their", "them", "then", "there",
    "this", "what", "when", "where", "which", "who", "why", "with", "would",
}


def load_csv(path: str | Path) -> list[dict[str, str]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: object) -> float | None:
    try:
        number = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _period(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value).strip())
    except (TypeError, ValueError):
        return None


def _preferred_quarters(metric: str) -> str:
    if metric in BALANCE_METRICS:
        return "0"
    if metric == "operating_cash_flow":
        return "3"
    return "1"


def latest_metric_fact(
    facts: Iterable[dict[str, object]], company: str, metric: str
) -> dict[str, object] | None:
    matches = [
        row for row in facts
        if str(row.get("company", "")).casefold() == company.casefold()
        and str(row.get("metric", "")).casefold() == metric.casefold()
        and _number(row.get("value")) is not None
        and _period(row.get("period_end")) is not None
    ]
    if not matches:
        return None
    preferred = _preferred_quarters(metric)
    preferred_matches = [row for row in matches if str(row.get("period_quarters", "")) == preferred]
    pool = preferred_matches or matches
    latest = max(pool, key=lambda row: (_period(row.get("period_end")), str(row.get("adsh", ""))))
    normalized = dict(latest)
    normalized["value"] = _number(latest.get("value"))
    return normalized


def _recent_fact_pairs(
    facts: list[dict[str, object]], company: str, metric: str
) -> list[dict[str, object]]:
    matches = [
        row for row in facts
        if str(row.get("company", "")).casefold() == company.casefold()
        and str(row.get("metric", "")).casefold() == metric.casefold()
        and _number(row.get("value")) is not None
        and _period(row.get("period_end")) is not None
    ]
    preferred = _preferred_quarters(metric)
    chosen = [row for row in matches if str(row.get("period_quarters", "")) == preferred]
    if not chosen:
        chosen = matches
    # One value per date; conflicting rows should not have survived preparation,
    # but avoid silently duplicating any if data is manually edited later.
    by_date: dict[str, dict[str, object]] = {}
    for row in chosen:
        period_end = str(row.get("period_end", ""))
        existing = by_date.get(period_end)
        if existing is None:
            by_date[period_end] = row
        elif _number(existing.get("value")) != _number(row.get("value")):
            by_date.pop(period_end, None)
    return [by_date[key] for key in sorted(by_date, reverse=True)[:2]]


def format_fact_value(row: dict[str, object]) -> str:
    value = _number(row.get("value"))
    if value is None:
        return "not reported"
    unit = str(row.get("unit", "")).strip()
    if str(row.get("metric", "")).casefold() == "eps_diluted":
        return f"${value:,.2f}/share"
    if unit == "USD":
        if abs(value) >= 1_000_000_000:
            return f"${value / 1_000_000_000:,.2f}B"
        if abs(value) >= 1_000_000:
            return f"${value / 1_000_000:,.2f}M"
        return f"${value:,.2f}"
    if "share" in unit.casefold():
        return f"${value:,.2f}/share"
    return f"{value:,.4g} {unit}".strip()


def _query_terms(question: str) -> set[str]:
    return {
        term.casefold() for term in re.findall(r"[A-Za-z0-9]+", question)
        if len(term) > 2 and term.casefold() not in STOPWORDS
    }


def _relevant_evidence(
    evidence: list[dict[str, str]], company: str, question: str, limit: int = 5
) -> list[dict[str, str]]:
    rows = [row for row in evidence if row.get("company", "").casefold() == company.casefold()]
    query_terms = _query_terms(question)
    scored: list[tuple[int, str, dict[str, str]]] = []
    for row in rows:
        searchable = " ".join(
            row.get(field, "") or ""
            for field in ("sentence", "category", "guidance_type", "risk_type")
        ).casefold()
        words = set(re.findall(r"[a-z0-9]+", searchable))
        score = len(query_terms & words)
        if score:
            scored.append((score, row.get("source_file", ""), row))
    if not scored:
        # Prefer extracted guidance/risk evidence to generic transcript filler.
        fallback = [row for row in rows if row.get("guidance_type") or row.get("risk_type")]
        if not fallback:
            fallback = rows
        return fallback[:limit]
    scored.sort(key=lambda item: (-item[0], item[1], item[2].get("sentence", "")))
    return [row for _, _, row in scored[:limit]]


def _sentiment_summary(evidence: list[dict[str, str]], company: str) -> str:
    counts = Counter(
        row.get("sentiment", "").strip().casefold()
        for row in evidence
        if row.get("company", "").casefold() == company.casefold()
        and row.get("sentiment", "").strip().casefold() in {"positive", "neutral", "negative"}
    )
    total = sum(counts.values())
    if total == 0:
        return "No normalized transcript sentiment labels for this company."
    parts = [f"{label}: {counts[label]} ({counts[label] / total:.1%})" for label in ("positive", "neutral", "negative") if counts[label]]
    return f"Labeled transcript sentences: {total}; " + "; ".join(parts)


def build_company_context(
    company: str,
    facts: Iterable[dict[str, object]],
    evidence: Iterable[dict[str, str]],
    question: str,
    max_evidence: int = 5,
) -> str:
    """Create a compact, source-cited context block; no unsupported values are invented."""
    facts_list = list(facts)
    evidence_list = list(evidence)
    company_facts = [row for row in facts_list if str(row.get("company", "")).casefold() == company.casefold()]
    if not company_facts:
        return f"No curated SEC financial facts are available for {company}. Do not infer or substitute another company."

    latest_period = max(str(row.get("period_end", "")) for row in company_facts)
    latest_filing = max(
        (row for row in company_facts if str(row.get("period_end", "")) == latest_period),
        key=lambda row: (str(row.get("form", "")), str(row.get("adsh", ""))),
    )
    lines = [
        f"Company: {company}.",
        f"Latest included SEC statement period: {latest_period}; filing form: {latest_filing.get('form', 'unknown')}; accession: {latest_filing.get('adsh', 'unknown')}.",
        "Financial values below are reported facts from that filing; USD facts are provided in original reported units and formatted equivalents.",
    ]

    for metric in METRIC_LABELS:
        pair = _recent_fact_pairs(facts_list, company, metric)
        if not pair:
            continue
        label = METRIC_LABELS[metric]
        quarters = str(pair[0].get("period_quarters", ""))
        if quarters == "0":
            scope = "point-in-time"
        elif quarters == "1":
            scope = "one-quarter duration"
        elif quarters == "3":
            scope = "three-quarter year-to-date duration"
        elif quarters == "4":
            scope = "four-quarter fiscal-year duration"
        else:
            scope = f"{quarters or 'unknown'}-quarter duration"
        for index, row in enumerate(pair):
            value = _number(row.get("value"))
            raw = "not reported" if value is None else f"{value:.10g} {row.get('unit', '')}".strip()
            source = f"SEC source: accession {row.get('adsh', 'unknown')}, period {row.get('period_end', 'unknown')}, tag {row.get('tag', 'unknown')}"
            prefix = "Latest" if index == 0 else "Prior available"
            lines.append(f"{label} ({scope}) — {prefix}, {row.get('period_end')}: {format_fact_value(row)}; raw={raw}. {source}.")
        if len(pair) == 2:
            newer, older = pair
            new_date, old_date = _period(newer.get("period_end")), _period(older.get("period_end"))
            new_value, old_value = _number(newer.get("value")), _number(older.get("value"))
            if new_date and old_date and new_date.month == old_date.month and new_date.day == old_date.day and new_date.year == old_date.year + 1 and new_value is not None and old_value is not None and old_value > 0 and new_value >= 0:
                growth = (new_value / old_value - 1) * 100
                lines.append(f"Calculated year-over-year change for {label}, same reported period length: {growth:+.1f}% (computed from the two cited facts).")

    lines.append("Transcript sentiment label mix: " + _sentiment_summary(evidence_list, company))
    snippets = _relevant_evidence(evidence_list, company, question, max_evidence)
    if snippets:
        lines.append("Transcript evidence (quotes are the source text; cite each as [Transcript: filename]):")
        for row in snippets:
            labels = []
            if row.get("guidance_type"):
                labels.append("guidance=" + row["guidance_type"])
            if row.get("risk_type"):
                labels.append("risk=" + row["risk_type"])
            if row.get("category"):
                labels.append("category=" + row["category"])
            annotations = f" ({'; '.join(labels)})" if labels else ""
            lines.append(f"- \"{row.get('sentence', '').strip()}\"{annotations} [Transcript: {row.get('source_file', 'unknown')}]")
    else:
        lines.append("No matching transcript evidence was found for this question; do not invent transcript statements.")
    return "\n".join(lines)


def sentiment_counts(evidence: Iterable[dict[str, str]], company: str) -> dict[str, int]:
    return dict(
        Counter(
            row.get("sentiment", "").strip().casefold()
            for row in evidence
            if row.get("company", "").casefold() == company.casefold()
            and row.get("sentiment", "").strip().casefold() in {"positive", "neutral", "negative"}
        )
    )
