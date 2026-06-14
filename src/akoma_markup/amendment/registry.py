"""Amendment registry parser for CSV/TSV amendment tracking.

Parses amendment registries into structured AmendmentRecord objects
and validates amendment records.
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class AmendmentRecord:
    """Single amendment record from CSV registry.

    Represents one row in the amendment registry CSV - one section
    amended by one amendment act.
    """
    act_name: str
    base_version: str
    amendment_act: str
    amendment_year: str
    gazette_file: str
    sections_amended: list[str]
    effective_date: str | None = None
    notes: str | None = None

    def __post_init__(self):
        """Normalize section list from string if needed."""
        if isinstance(self.sections_amended, str):
            self.sections_amended = [
                s.strip() for s in self.sections_amended.split(",")
                if s.strip()
            ]


@dataclass
class AmendmentDetail:
    """Detailed amendment instruction from TSV.

    Contains the actual text/content for the amendment operation.
    """
    section: str
    operation: Literal["replace", "insert", "delete"]
    new_text: str
    effective_date: str
    amendment_act: str
    gazette_ref: str = ""
    metadata: dict | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def parse_csv_registry(
    csv_path: str | Path,
    delimiter: str = ",",
) -> list[AmendmentRecord]:
    """Parse amendment registry CSV into AmendmentRecord objects.

    Expected CSV format:
        act_name,base_version,amendment_act,amendment_year,
            gazette_file,sections_amended,effective_date,notes

    One row per section amended. The act_name and amendment_act can be repeated
    across multiple rows.

    Args:
        csv_path: Path to the CSV file.
        delimiter: Field delimiter (default: comma).

    Returns:
        List of AmendmentRecord objects.

    Raises:
        FileNotFoundError: If CSV file doesn't exist.
        ValueError: If CSV is malformed or missing required columns.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Registry CSV not found: {csv_path}")

    records: list[AmendmentRecord] = []
    required_fields = {
        "act_name", "base_version", "amendment_act",
        "amendment_year", "gazette_file", "sections_amended"
    }

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        # Validate headers
        if not reader.fieldnames:
            raise ValueError("CSV file is empty or has no headers")

        missing = required_fields - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Missing required columns in CSV: {sorted(missing)}"
            )

        for row_num, row in enumerate(reader, start=2):
            try:
                record = AmendmentRecord(
                    act_name=row["act_name"].strip(),
                    base_version=row["base_version"].strip(),
                    amendment_act=row["amendment_act"].strip(),
                    amendment_year=row["amendment_year"].strip(),
                    gazette_file=row["gazette_file"].strip(),
                    sections_amended=row["sections_amended"],
                    effective_date=row.get("effective_date", "").strip()
                    or None,
                    notes=row.get("notes", "").strip() or None,
                )
                records.append(record)
            except Exception as exc:
                logger.error("Error parsing row %d: %s", row_num, exc)
                raise ValueError(
                    f"Error parsing row {row_num}: {exc}"
                ) from exc

    logger.info(
        "Parsed %d amendment records from %s (%s)",
        len(records), csv_path, len({r.act_name for r in records})
    )
    return records


def parse_tsv_details(
    tsv_path: str | Path,
) -> list[AmendmentDetail]:
    """Parse detailed amendment TSV into AmendmentDetail objects.

    Expected TSV format:
        section	operation	new_text	effective_date	amendment_act
        43	replace	New text...	2009-10-27	IT (Amendment) Act, 2008

    Args:
        tsv_path: Path to the TSV file.

    Returns:
        List of AmendmentDetail objects.

    Raises:
        FileNotFoundError: If TSV file doesn't exist.
        ValueError: If TSV is malformed.
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Details TSV not found: {tsv_path}")

    details: list[AmendmentDetail] = []
    valid_operations = {"replace", "insert", "delete"}

    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        required = {"section", "operation", "new_text",
                    "effective_date", "amendment_act"}
        if not reader.fieldnames:
            raise ValueError("TSV file is empty or has no headers")

        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Missing required columns in TSV: {sorted(missing)}"
            )

        for row_num, row in enumerate(reader, start=2):
            operation = row["operation"].strip().lower()
            if operation not in valid_operations:
                raise ValueError(
                    f"Invalid operation '{operation}' at row {row_num}. "
                    f"Must be one of: {valid_operations}"
                )

            detail = AmendmentDetail(
                section=row["section"].strip(),
                operation=operation,
                new_text=row["new_text"],
                effective_date=row["effective_date"].strip(),
                amendment_act=row["amendment_act"].strip(),
                gazette_ref=row.get("gazette_ref", "").strip(),
                metadata={
                    k: v for k, v in row.items()
                    if k not in required and k != "gazette_ref"
                }
            )
            details.append(detail)

    logger.info(
        "Parsed %d amendment details from %s",
        len(details), tsv_path
    )
    return details


def validate_amendment_records(
    records: list[AmendmentRecord],
    gazette_folder: str | Path | None = None,
) -> dict[str, list[str]]:
    """Validate amendment records and optionally check gazette files.

    Args:
        records: List of AmendmentRecord objects from parse_csv_registry.
        gazette_folder: Optional path to folder containing gazette files
            to verify gazette_file references.

    Returns:
        Dictionary of validation issues by category.
    """
    issues: dict[str, list[str]] = {
        "missing_act_name": [],
        "missing_base_version": [],
        "missing_amendment_act": [],
        "missing_sections": [],
        "duplicate_records": [],
        "missing_gazette_files": [],
    }

    seen: set[str] = set()

    for i, record in enumerate(records):
        if not record.act_name:
            issues["missing_act_name"].append(f"Row {i+1}")
        if not record.base_version:
            issues["missing_base_version"].append(f"Row {i+1}")
        if not record.amendment_act:
            issues["missing_amendment_act"].append(f"Row {i+1}")
        if not record.sections_amended:
            issues["missing_sections"].append(f"Row {i+1}")

        # Check for duplicates
        key = f"{record.act_name}:{record.amendment_act}:"
        f"{','.join(record.sections_amended)}"
        if key in seen:
            issues["duplicate_records"].append(
                f"Row {i+1}: {record.act_name} - "
                f"{record.amendment_act} - sections "
                f"{','.join(record.sections_amended)}"
            )
        seen.add(key)

        # Check gazette files exist if folder provided
        if gazette_folder:
            gazette_path = Path(gazette_folder) / record.gazette_file
            if not gazette_path.exists():
                issues["missing_gazette_files"].append(
                    f"Row {i+1}: {record.gazette_file}"
                )

    return {k: v for k, v in issues.items() if v}


def group_amendments_by_act(
    records: list[AmendmentRecord],
) -> dict[str, list[AmendmentRecord]]:
    """Group amendment records by base act name.

    Args:
        records: List of AmendmentRecord objects.

    Returns:
        Dictionary mapping act_name -> list of records.
    """
    result: dict[str, list[AmendmentRecord]] = {}
    for record in records:
        if record.act_name not in result:
            result[record.act_name] = []
        result[record.act_name].append(record)
    return result


def get_amendment_timeline(
    records: list[AmendmentRecord],
) -> list[AmendmentRecord]:
    """Sort amendment records chronologically by year and effective date.

    Args:
        records: List of AmendmentRecord objects.

    Returns:
        Sorted list of records.
    """
    def sort_key(r: AmendmentRecord) -> tuple:
        return (r.amendment_year, r.effective_date or "")

    return sorted(records, key=sort_key)
