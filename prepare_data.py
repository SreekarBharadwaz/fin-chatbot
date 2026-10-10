"""Build small, deployable chatbot data files from the supplied SEC/NLP exports.

Usage (Windows):
  python prepare_data.py --sec-csv "...\\chatbot_data\\financial_merged.csv" \
    --transcripts-dir "...\\unstructured_final_results"
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from data_prep import METRIC_TAGS, SEC_COMPANIES, TAG_TO_METRIC, normalize_transcript_company, select_consolidated_facts

csv.field_size_limit(20_000_000)

OUTPUT_FACT_FIELDS = [
    "company", "adsh", "period_end", "form", "period_quarters",
    "metric", "value", "unit", "tag", "cik",
]
EVIDENCE_FIELDS = [
    "company", "source_file", "source_type", "sentence", "category",
    "sentiment", "sentiment_confidence", "guidance_type", "guidance_confidence",
    "risk_type", "risk_confidence", "year", "quarter", "speaker",
]


def period_key(value: str) -> tuple[int, int, int] | None:
    try:
        year, month, day = (int(part) for part in value.split("-"))
        return year, month, day
    except (AttributeError, TypeError, ValueError):
        return None


def latest_filing_candidates(sec_csv: Path) -> dict[str, dict[str, str]]:
    """Find each target's latest filing by statement period-end and exact tag."""
    by_legal_name = {name.casefold(): canonical for name, canonical in SEC_COMPANIES.items()}
    candidates: dict[str, dict[str, str]] = {}
    with sec_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            canonical = by_legal_name.get((row.get("name") or "").strip().casefold())
            tag = (row.get("tag") or "").strip()
            form = (row.get("form") or "").strip().upper()
            period = period_key((row.get("ddate") or "").strip())
            adsh = (row.get("adsh") or "").strip()
            if canonical is None or tag not in TAG_TO_METRIC or form not in {"10-Q", "10-K"} or period is None or not adsh:
                continue
            rank = (period, 1 if form == "10-K" else 0, adsh)
            old = candidates.get(canonical)
            old_period = period_key(old.get("ddate", "")) if old else None
            old_rank = (
                old_period,
                1 if old and old.get("form", "").upper() == "10-K" else 0,
                old.get("adsh", "") if old else "",
            )
            if old is None or rank > old_rank:
                candidates[canonical] = {
                    "company": canonical,
                    "ddate": row["ddate"].strip(),
                    "form": form,
                    "adsh": adsh,
                    "cik": (row.get("cik") or "").strip(),
                }
    return candidates


def write_financial_facts(sec_csv: Path, output_dir: Path) -> dict[str, dict[str, str]]:
    candidates = latest_filing_candidates(sec_csv)
    canonical_by_accession = {row["adsh"]: company for company, row in candidates.items()}
    wanted = {tag for tags in METRIC_TAGS.values() for tag in tags}
    selected_rows: list[dict[str, str]] = []
    with sec_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            accession = (row.get("adsh") or "").strip()
            company = canonical_by_accession.get(accession)
            if company is None or (row.get("tag") or "").strip() not in wanted:
                continue
            row["company"] = company
            selected_rows.append(row)

    legal_map = {legal_name: canonical for legal_name, canonical in SEC_COMPANIES.items()}
    facts = select_consolidated_facts(selected_rows, legal_map)
    # Keep only the chosen accession for each company. The raw SEC "filed" column
    # was parsed as a malformed 1970 timestamp in the notebook, so do not use it.
    facts = [fact for fact in facts if fact["adsh"] == candidates.get(str(fact["company"]), {}).get("adsh")]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "financial_facts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FACT_FIELDS)
        writer.writeheader()
        writer.writerows(facts)
    with (output_dir / "filings_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["company", "period_end", "form", "adsh", "cik"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for company in sorted(candidates):
            row = candidates[company]
            writer.writerow({"company": company, "period_end": row["ddate"], "form": row["form"], "adsh": row["adsh"], "cik": row["cik"]})
    return candidates


def write_transcript_evidence(results_dir: Path, output_dir: Path) -> tuple[int, Counter[str]]:
    # Merge the base classified sentences with the corresponding extracted
    # guidance/risk annotations while discarding noisy company labels.
    records: dict[tuple[str, str, str], dict[str, str]] = {}

    def key_for(row: dict[str, str]) -> tuple[str, str, str] | None:
        company = normalize_transcript_company(row.get("main_company"))
        sentence = (row.get("sentence") or "").strip()
        filename = (row.get("file_name") or "").strip()
        if company is None or not sentence or not filename:
            return None
        return company, filename, sentence

    base_path = results_dir / "sentence_df.csv"
    with base_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = key_for(row)
            if key is None:
                continue
            company, filename, sentence = key
            records[key] = {
                "company": company,
                "source_file": filename,
                "source_type": "transcript",
                "sentence": sentence,
                "category": (row.get("category") or "").strip(),
                "sentiment": (row.get("sentiment") or "").strip().casefold(),
                "sentiment_confidence": (row.get("sentiment_score") or "").strip(),
                "guidance_type": "",
                "guidance_confidence": "",
                "risk_type": "",
                "year": (row.get("year") or "").strip(),
                "quarter": (row.get("quarter") or "").strip(),
                "speaker": (row.get("speaker") or "").strip(),
            }

    for filename, category_field, confidence_field in [
        ("guidance_full.csv", "guidance_type", "guidance_confidence"),
        ("risk_sentences.csv", "risk_type", "risk_confidence"),
    ]:
        path = results_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = key_for(row)
                if key is None:
                    continue
                company, source_file, sentence = key
                target = records.setdefault(
                    key,
                    {
                        "company": company,
                        "source_file": source_file,
                        "source_type": "transcript",
                        "sentence": sentence,
                        "category": (row.get("category") or "").strip(),
                        "sentiment": (row.get("sentiment") or "").strip().casefold(),
                        "sentiment_confidence": (row.get("sentiment_score") or "").strip(),
                        "guidance_type": "",
                        "guidance_confidence": "",
                        "risk_type": "",
                        "year": (row.get("year") or "").strip(),
                        "quarter": (row.get("quarter") or "").strip(),
                        "speaker": (row.get("speaker") or "").strip(),
                    },
                )
                target[category_field] = (row.get("guidance_type") or row.get("specific_risk_type") or "").strip()
                target[confidence_field] = (row.get(confidence_field) or "").strip()

    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records.values(), key=lambda row: (row["company"], row["source_file"], row["sentence"]))
    with (output_dir / "transcript_evidence.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVIDENCE_FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    sentiment_counts = Counter(row["sentiment"] for row in ordered if row["sentiment"])
    company_counts = Counter(row["company"] for row in ordered)
    return len(ordered), company_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare compact financial-chatbot data from your existing CSV exports.")
    parser.add_argument("--sec-csv", type=Path, default=Path.home() / "Downloads" / "2025q4" / "chatbot_data" / "financial_merged.csv")
    parser.add_argument("--transcripts-dir", type=Path, default=Path.home() / "Downloads" / "unstructured_final_results")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    if not args.sec_csv.exists():
        raise SystemExit(f"SEC source not found: {args.sec_csv}")
    if not (args.transcripts_dir / "sentence_df.csv").exists():
        raise SystemExit(f"Transcript source not found: {args.transcripts_dir / 'sentence_df.csv'}")

    print(f"Scanning SEC export: {args.sec_csv}")
    candidates = write_financial_facts(args.sec_csv, args.output_dir)
    print("Selected SEC filings:")
    for company, filing in sorted(candidates.items()):
        print(f"  {company}: {filing['form']} period {filing['ddate']} ({filing['adsh']})")
    print(f"Saved {sum(1 for _ in (args.output_dir / 'financial_facts.csv').open(encoding='utf-8')) - 1} consolidated facts.")

    evidence_count, companies = write_transcript_evidence(args.transcripts_dir, args.output_dir)
    print(f"Saved {evidence_count} transcript/guidance/risk records for {len(companies)} recognized companies.")
    print("Transcript coverage:", ", ".join(f"{company}={count}" for company, count in sorted(companies.items())))
    print(f"Output folder: {args.output_dir}")


if __name__ == "__main__":
    main()
