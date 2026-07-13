"""
scripts/01_inspect_data.py

Data reconnaissance script for the HireScore project.

Scans every CSV file inside `datasets/raw` (including subfolders),
collects structural metadata about each dataset (shape, columns,
dtypes, missing values, duplicate rows), prints a human-readable
report to the terminal, and persists the same report to
`docs/dataset_report.txt`.

This script performs NO cleaning, NO merging, and NO modeling.
It is strictly a read-only inspection tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

RAW_DATA_DIR: Path = PROJECT_ROOT / "datasets" / "raw"
REPORT_OUTPUT_PATH: Path = PROJECT_ROOT / "docs" / "dataset_report.txt"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class DatasetReport:
    """Container holding inspection metadata for a single dataset.

    Attributes:
        filename: Name of the CSV file.
        num_rows: Number of rows in the dataset.
        num_columns: Number of columns in the dataset.
        column_names: List of column names.
        dtypes: Mapping of column name to its pandas dtype (as string).
        missing_values: Mapping of column name to count of missing values.
        duplicate_row_count: Number of fully duplicated rows.
        error: Error message if the dataset could not be read, else None.
    """

    filename: str
    num_rows: int = 0
    num_columns: int = 0
    column_names: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    missing_values: dict[str, int] = field(default_factory=dict)
    duplicate_row_count: int = 0
    error: str | None = None


# --------------------------------------------------------------------------- #
# Core functions
# --------------------------------------------------------------------------- #

def discover_csv_files(raw_dir: Path) -> list[Path]:
    """Discover all CSV files inside a given directory, including subfolders.

    Args:
        raw_dir: Directory to search for CSV files.

    Returns:
        A sorted list of Path objects pointing to discovered CSV files.
        Returns an empty list if the directory does not exist or
        contains no CSV files.
    """
    if not raw_dir.exists() or not raw_dir.is_dir():
        return []

    return sorted(raw_dir.rglob("*.csv"))


def _read_csv_with_fallback_encoding(csv_path: Path) -> pd.DataFrame:
    """Read a CSV file, falling back to latin-1 if utf-8 decoding fails.

    Args:
        csv_path: Path to the CSV file to read.

    Returns:
        The loaded DataFrame.

    Raises:
        Exception: Propagates any error that is not a UnicodeDecodeError,
            or if reading still fails after the encoding fallback.
    """
    try:
        return pd.read_csv(csv_path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, low_memory=False, encoding="latin-1")


def inspect_dataset(csv_path: Path) -> DatasetReport:
    """Inspect a single CSV dataset and collect structural metadata.

    Args:
        csv_path: Path to the CSV file to inspect.

    Returns:
        A DatasetReport instance populated with metadata. If the file
        cannot be read, the returned report will have its `error`
        field set and other numeric fields left at their defaults.
    """
    filename = csv_path.name

    try:
        df: pd.DataFrame = _read_csv_with_fallback_encoding(csv_path)
    except Exception as exc:  # noqa: BLE001 - intentional broad catch for robustness
        return DatasetReport(filename=filename, error=f"Failed to read file: {exc}")

    try:
        num_rows, num_columns = df.shape
        column_names = list(df.columns)
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        missing_values = {col: int(df[col].isna().sum()) for col in df.columns}
        duplicate_row_count = int(df.duplicated().sum())

        return DatasetReport(
            filename=filename,
            num_rows=num_rows,
            num_columns=num_columns,
            column_names=column_names,
            dtypes=dtypes,
            missing_values=missing_values,
            duplicate_row_count=duplicate_row_count,
        )
    except Exception as exc:  # noqa: BLE001 - guard against unexpected pandas errors
        return DatasetReport(filename=filename, error=f"Failed to analyze file: {exc}")


def inspect_all_datasets(raw_dir: Path) -> list[DatasetReport]:
    """Discover and inspect every CSV dataset inside a directory.

    Args:
        raw_dir: Directory containing raw CSV datasets.

    Returns:
        A list of DatasetReport objects, one per discovered CSV file.
    """
    csv_files = discover_csv_files(raw_dir)

    if not csv_files:
        print("No CSV files found inside datasets/raw")

    return [inspect_dataset(csv_path) for csv_path in csv_files]


# --------------------------------------------------------------------------- #
# Report formatting
# --------------------------------------------------------------------------- #

def format_dataset_report(report: DatasetReport) -> str:
    """Format a single DatasetReport into a human-readable text block.

    Args:
        report: The DatasetReport to format.

    Returns:
        A formatted multi-line string describing the dataset.
    """
    lines: list[str] = []
    separator = "-" * 70

    lines.append(separator)
    lines.append(f"Dataset: {report.filename}")
    lines.append(separator)

    if report.error:
        lines.append(f"  [ERROR] {report.error}")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"  Rows            : {report.num_rows}")
    lines.append(f"  Columns         : {report.num_columns}")
    lines.append(f"  Duplicate rows  : {report.duplicate_row_count}")
    lines.append("")
    lines.append("  Column details:")

    for col in report.column_names:
        dtype = report.dtypes.get(col, "unknown")
        missing = report.missing_values.get(col, 0)
        lines.append(f"    - {col!r:30} dtype={dtype:10} missing={missing}")

    lines.append("")
    return "\n".join(lines)


def build_full_report(reports: list[DatasetReport]) -> str:
    """Build the complete textual report for all inspected datasets.

    Args:
        reports: List of DatasetReport objects.

    Returns:
        A single string containing the formatted report for all datasets.
    """
    header = [
        "=" * 70,
        "HireScore - Dataset Inspection Report",
        f"Total datasets found: {len(reports)}",
        "=" * 70,
        "",
    ]

    if not reports:
        header.append("No CSV files were found in the raw datasets directory.")
        return "\n".join(header)

    body = [format_dataset_report(report) for report in reports]
    return "\n".join(header) + "\n" + "\n".join(body)


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def save_report(report_text: str, output_path: Path) -> None:
    """Save the report text to a file, creating parent directories as needed.

    Args:
        report_text: The full report content to write.
        output_path: Destination file path for the report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    """Run the dataset inspection pipeline end-to-end."""
    reports = inspect_all_datasets(RAW_DATA_DIR)
    full_report = build_full_report(reports)

    print(full_report)

    try:
        save_report(full_report, REPORT_OUTPUT_PATH)
        print(f"\nReport saved to: {REPORT_OUTPUT_PATH.resolve()}")
    except Exception as exc:  # noqa: BLE001 - intentional broad catch for robustness
        print(f"\n[ERROR] Could not save report to {REPORT_OUTPUT_PATH}: {exc}")


if __name__ == "__main__":
    main()
