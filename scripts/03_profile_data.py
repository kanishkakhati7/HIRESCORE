"""
03_profile_data.py

Purpose
-------
Generate a detailed, human-readable profile report for every cleaned
dataset produced by 02_clean_data.py. For each CSV file found in
datasets/processed/, this script computes dataset-level statistics,
per-column statistics, a numeric summary, and a categorical summary,
then writes the results to an individual plain-text report.

Input directory:  datasets/processed/
Output directory: reports/data_profiles/

This script only profiles data. It intentionally does NOT:
    - clean or modify any dataset
    - engineer features
    - merge datasets
    - train models
    - produce visualizations
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "data_profiles"

SEPARATOR_HEAVY = "=" * 33
SEPARATOR_LIGHT = "-" * 33

TOP_N_CATEGORICAL_VALUES = 10


def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load a single CSV dataset into a pandas DataFrame.

    Parameters
    ----------
    file_path : Path
        Full path to the CSV file to load.

    Returns
    -------
    pd.DataFrame
        The dataset as loaded from disk, unmodified.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    Exception
        Re-raises any error encountered by pandas while reading the file.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # low_memory=False avoids dtype-guessing warnings on large, mixed-type CSVs.
    # Fall back to latin-1 if a file isn't valid UTF-8 (matches prior scripts).
    try:
        return pd.read_csv(file_path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, low_memory=False, encoding="latin-1")


def calculate_memory_usage(df: pd.DataFrame) -> float:
    """
    Calculate the total in-memory footprint of a DataFrame in megabytes.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose memory usage should be measured.

    Returns
    -------
    float
        The total memory usage of the DataFrame, in megabytes, rounded
        to 4 decimal places. Uses deep=True so that object/string columns
        are measured accurately rather than by pointer size alone.
    """
    total_bytes = df.memory_usage(deep=True).sum()
    total_megabytes = total_bytes / (1024 ** 2)
    return round(float(total_megabytes), 4)


def _format_dataset_summary(
    dataset_name: str,
    absolute_path: Path,
    row_count: int,
    column_count: int,
    memory_mb: float,
) -> str:
    """
    Build the "DATASET SUMMARY" section of a profile report.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset (typically the file stem).
    absolute_path : Path
        The absolute path to the source CSV file.
    row_count : int
        The number of rows in the dataset.
    column_count : int
        The number of columns in the dataset.
    memory_mb : float
        The memory footprint of the dataset, in megabytes.

    Returns
    -------
    str
        The formatted dataset summary section, ready to be written to file.
    """
    lines: List[str] = [
        SEPARATOR_HEAVY,
        "DATASET SUMMARY",
        SEPARATOR_HEAVY,
        f"Dataset Name: {dataset_name}",
        f"Absolute Path: {absolute_path}",
        f"Rows: {row_count}",
        f"Columns: {column_count}",
        f"Memory Usage (MB): {memory_mb}",
    ]
    return "\n".join(lines)


def profile_columns(df: pd.DataFrame) -> str:
    """
    Build the "COLUMN INFORMATION" section of a profile report.

    For every column, reports the column name, data type, missing value
    count and percentage, and unique value count. Numeric columns are
    additionally profiled with min, max, mean, median, and standard
    deviation. Non-numeric (categorical) columns are additionally
    profiled with their top 10 most frequent values and counts.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose columns should be profiled.

    Returns
    -------
    str
        The formatted column information section, ready to be written
        to file. If the DataFrame has no columns, a placeholder message
        is returned instead.
    """
    total_rows = len(df)

    if df.shape[1] == 0:
        return "No columns found in this dataset."

    section_lines: List[str] = []

    for column_name in df.columns:
        series = df[column_name]
        missing_count = int(series.isna().sum())
        missing_pct = (
            round((missing_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        )
        unique_count = int(series.nunique(dropna=True))

        section_lines.append(f"Column Name: {column_name}")
        section_lines.append(f"Data Type: {series.dtype}")
        section_lines.append(f"Missing Values (count): {missing_count}")
        section_lines.append(f"Missing Percentage: {missing_pct}%")
        section_lines.append(f"Unique Values: {unique_count}")

        if pd.api.types.is_numeric_dtype(series):
            section_lines.extend(_profile_numeric_column(series))
        else:
            section_lines.extend(_profile_categorical_column(series))

        section_lines.append("")  # blank line between columns

    return "\n".join(section_lines).rstrip("\n")


def _profile_numeric_column(series: pd.Series) -> List[str]:
    """
    Build the numeric-specific detail lines for a single column.

    Parameters
    ----------
    series : pd.Series
        The numeric column to summarize.

    Returns
    -------
    List[str]
        Lines describing min, max, mean, median, and standard deviation.
        If the column contains no non-null values, placeholder values
        of "N/A" are returned instead of raising an error.
    """
    non_null = series.dropna()

    if non_null.empty:
        return [
            "  Min: N/A",
            "  Max: N/A",
            "  Mean: N/A",
            "  Median: N/A",
            "  Std: N/A",
        ]

    return [
        f"  Min: {non_null.min()}",
        f"  Max: {non_null.max()}",
        f"  Mean: {round(float(non_null.mean()), 4)}",
        f"  Median: {round(float(non_null.median()), 4)}",
        f"  Std: {round(float(non_null.std()), 4)}",
    ]


def _profile_categorical_column(series: pd.Series) -> List[str]:
    """
    Build the categorical-specific detail lines for a single column.

    Parameters
    ----------
    series : pd.Series
        The non-numeric column to summarize.

    Returns
    -------
    List[str]
        Lines listing the top 10 most frequent values and their counts.
        If the column contains no non-null values, a placeholder line
        is returned instead of raising an error.
    """
    top_values = series.value_counts(dropna=True).head(TOP_N_CATEGORICAL_VALUES)

    if top_values.empty:
        return ["  Top Values: N/A"]

    lines = ["  Top 10 Most Frequent Values:"]
    for value, count in top_values.items():
        lines.append(f"    {value}: {count}")
    return lines


def profile_numeric(df: pd.DataFrame) -> str:
    """
    Build the "NUMERIC SUMMARY" section of a profile report.

    Equivalent to calling df.describe() on the numeric subset of the
    DataFrame's columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose numeric columns should be summarized.

    Returns
    -------
    str
        The formatted numeric summary table, or a placeholder message
        if the DataFrame has no numeric columns.
    """
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return "No numeric columns found in this dataset."

    return numeric_df.describe().to_string()


def profile_categorical(df: pd.DataFrame) -> str:
    """
    Build the "CATEGORICAL SUMMARY" section of a profile report.

    Equivalent to calling df.describe(include="object") on the
    categorical subset of the DataFrame's columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose categorical columns should be summarized.

    Returns
    -------
    str
        The formatted categorical summary table, or a placeholder
        message if the DataFrame has no categorical (object) columns.
    """
    categorical_df = df.select_dtypes(include="object")

    if categorical_df.empty:
        return "No categorical columns found in this dataset."

    return categorical_df.describe().to_string()


def write_report(report_text: str, output_path: Path) -> None:
    """
    Save a completed profile report to disk as a UTF-8 text file.

    Parameters
    ----------
    report_text : str
        The full, assembled contents of the profile report.
    output_path : Path
        Destination file path (including filename).

    Returns
    -------
    None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode="w", encoding="utf-8") as report_file:
        report_file.write(report_text)


def process_dataset(file_path: Path) -> Dict[str, object]:
    """
    Load, profile, and save a report for a single dataset.

    This function is designed to never raise: any failure is caught and
    returned as part of the result dictionary so that the overall
    pipeline can continue profiling remaining datasets.

    Parameters
    ----------
    file_path : Path
        Full path to the CSV file to profile.

    Returns
    -------
    Dict[str, object]
        A dictionary describing the outcome, with keys:
            - "filename": str
            - "success": bool
            - "rows": int | None
            - "columns": int | None
            - "saved_path": str | None
            - "error": str | None
    """
    dataset_name = file_path.stem

    result: Dict[str, object] = {
        "filename": file_path.name,
        "success": False,
        "rows": None,
        "columns": None,
        "saved_path": None,
        "error": None,
    }

    try:
        print(f"\nProfiling dataset: {file_path.name}")
        df = load_dataset(file_path)
        row_count, column_count = df.shape
        memory_mb = calculate_memory_usage(df)
        print(f"  Rows: {row_count}, Columns: {column_count}")

        summary_section = _format_dataset_summary(
            dataset_name=dataset_name,
            absolute_path=file_path.resolve(),
            row_count=row_count,
            column_count=column_count,
            memory_mb=memory_mb,
        )

        column_section = "\n".join(
            [SEPARATOR_LIGHT, "COLUMN INFORMATION", SEPARATOR_LIGHT, profile_columns(df)]
        )

        numeric_section = "\n".join(
            [SEPARATOR_LIGHT, "NUMERIC SUMMARY", SEPARATOR_LIGHT, profile_numeric(df)]
        )

        categorical_section = "\n".join(
            [
                SEPARATOR_LIGHT,
                "CATEGORICAL SUMMARY",
                SEPARATOR_LIGHT,
                profile_categorical(df),
            ]
        )

        full_report = "\n\n".join(
            [summary_section, column_section, numeric_section, categorical_section]
        )

        output_path = REPORTS_DIR / f"{dataset_name}_profile.txt"
        write_report(full_report, output_path)
        print(f"  Saved to: {output_path}")

        result.update(
            {
                "success": True,
                "rows": row_count,
                "columns": column_count,
                "saved_path": str(output_path),
            }
        )

    except Exception as exc:  # noqa: BLE001 - intentionally broad to keep pipeline alive
        # Continue processing other datasets even if this one fails.
        print(f"  FAILED to profile {file_path.name}: {exc}")
        result["error"] = str(exc)

    return result


def print_summary(results: List[Dict[str, object]]) -> None:
    """
    Print a final summary of the entire profiling run.

    Parameters
    ----------
    results : List[Dict[str, object]]
        The list of per-dataset result dictionaries produced by
        process_dataset.

    Returns
    -------
    None
    """
    processed = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print("\n" + SEPARATOR_HEAVY)
    print("PROFILE SUMMARY")
    print(SEPARATOR_HEAVY)

    print(f"Datasets processed successfully: {len(processed)}")
    for r in processed:
        print(f"  - {r['filename']}")

    print(f"Datasets failed: {len(failed)}")
    for r in failed:
        print(f"  - {r['filename']}: {r['error']}")

    print(f"Location of reports: {REPORTS_DIR}")
    print(SEPARATOR_HEAVY)


def main() -> None:
    """
    Run the full data profiling pipeline over every dataset found in
    datasets/processed/.

    Returns
    -------
    None
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(PROCESSED_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {PROCESSED_DIR}")
        return

    results: List[Dict[str, object]] = []

    for file_path in csv_files:
        result = process_dataset(file_path)
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()