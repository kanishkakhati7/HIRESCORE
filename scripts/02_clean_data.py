"""
02_clean_data.py

Purpose
-------
Perform basic, non-destructive cleaning of the raw RealityCheck datasets:
    - Remove duplicate rows
    - Normalize column names (strip whitespace, lowercase)
    - Strip leading/trailing whitespace from string values
    - Leave missing values untouched
    - Save each cleaned dataset individually (no merging, no feature
      engineering, no ML, no visualization)

Input directory:  datasets/raw/
Output directory: datasets/processed/

This script only cleans data. It intentionally does NOT:
    - engineer features
    - merge datasets
    - train models
    - produce visualizations
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"

# Only these datasets are in scope for this cleaning step.
DATASET_FILENAMES: List[str] = [
    "Resume.csv",
    "postings.csv",
    "salaries.csv",
    "skills.csv",
    "job_skills.csv",
    "companies.csv",
    "company_industries.csv",
    "industries.csv",
    "global_ai_jobs.csv",
]


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
        The raw, unmodified dataset as loaded from disk.

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
    # Fall back to latin-1 if a file isn't valid UTF-8 (matches inspection script).
    try:
        return pd.read_csv(file_path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, low_memory=False, encoding="latin-1")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from column names and convert them to lowercase.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose column names should be normalized.

    Returns
    -------
    pd.DataFrame
        A DataFrame with cleaned column names (whitespace stripped and
        lowercased). The underlying data is not modified.
    """
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from all string (object) values.

    Missing values (NaN) are left completely unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose string values should be stripped.

    Returns
    -------
    pd.DataFrame
        A DataFrame with whitespace-stripped string values. Non-string
        columns and missing values are untouched.
    """
    df = df.copy()

    # Identify columns holding object or pandas "string" dtype (typically strings)
    object_columns = df.select_dtypes(include=["object", "string"]).columns

    for col in object_columns:
        # Only strip actual string entries; leave NaN/None and other
        # non-string objects (e.g. mixed types) untouched.
        df[col] = df[col].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    return df


def remove_duplicate_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove exact duplicate rows from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to de-duplicate.

    Returns
    -------
    Tuple[pd.DataFrame, int]
        A tuple of (de-duplicated DataFrame, number of duplicate rows removed).
    """
    original_row_count = len(df)
    df_deduped = df.drop_duplicates()
    duplicates_removed = original_row_count - len(df_deduped)
    return df_deduped, duplicates_removed


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Apply the full, non-destructive cleaning pipeline to a single dataset.

    Steps applied (in order):
        1. Normalize column names (strip + lowercase).
        2. Strip whitespace from string values.
        3. Remove duplicate rows.

    Missing values are left unchanged throughout. No feature engineering
    or merging is performed.

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame to clean.

    Returns
    -------
    Tuple[pd.DataFrame, int]
        A tuple of (cleaned DataFrame, number of duplicate rows removed).
    """
    df = normalize_column_names(df)
    df = strip_string_values(df)
    df, duplicates_removed = remove_duplicate_rows(df)
    return df, duplicates_removed


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save a cleaned DataFrame to disk as a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned DataFrame to save.
    output_path : Path
        Destination file path (including filename).

    Returns
    -------
    None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def process_single_dataset(filename: str) -> Dict[str, object]:
    """
    Load, clean, and save a single dataset by filename, reporting results.

    This function is designed to never raise: any failure is caught and
    returned as part of the result dictionary so that the overall pipeline
    can continue processing remaining datasets.

    Parameters
    ----------
    filename : str
        The CSV filename (must exist under RAW_DIR) to process.

    Returns
    -------
    Dict[str, object]
        A dictionary describing the outcome, with keys:
            - "filename": str
            - "success": bool
            - "original_rows": int | None
            - "cleaned_rows": int | None
            - "duplicates_removed": int | None
            - "saved_path": str | None
            - "error": str | None
    """
    matches = list(RAW_DIR.rglob(filename))

    result: Dict[str, object] = {
        "filename": filename,
        "success": False,
        "original_rows": None,
        "cleaned_rows": None,
        "duplicates_removed": None,
        "saved_path": None,
        "error": None,
    }

    try:
        if not matches:
            raise FileNotFoundError(f"File not found: {filename}")

        input_path = matches[0]
        output_path = PROCESSED_DIR / filename

        print(f"\nLoading dataset: {filename}")
        raw_df = load_dataset(input_path)
        original_rows = len(raw_df)
        print(f"  Original rows: {original_rows}")

        cleaned_df, duplicates_removed = clean_dataset(raw_df)
        cleaned_rows = len(cleaned_df)
        print(f"  Cleaned rows: {cleaned_rows}")
        print(f"  Duplicates removed: {duplicates_removed}")

        save_dataset(cleaned_df, output_path)
        print(f"  Saved to: {output_path}")

        result.update(
            {
                "success": True,
                "original_rows": original_rows,
                "cleaned_rows": cleaned_rows,
                "duplicates_removed": duplicates_removed,
                "saved_path": str(output_path),
            }
        )

    except Exception as exc:  # noqa: BLE001 - intentionally broad to keep pipeline alive
        # Continue processing other datasets even if this one fails.
        print(f"  FAILED to process {filename}: {exc}")
        result["error"] = str(exc)

    return result


def print_summary(results: List[Dict[str, object]]) -> None:
    """
    Print a final summary of the entire cleaning run.

    Parameters
    ----------
    results : List[Dict[str, object]]
        The list of per-dataset result dictionaries produced by
        process_single_dataset.

    Returns
    -------
    None
    """
    processed = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    total_duplicates_removed = sum(
        r["duplicates_removed"] for r in processed if r["duplicates_removed"] is not None
    )

    print("\n" + "=" * 50)
    print("CLEANING SUMMARY")
    print("=" * 50)
    print(f"Datasets processed successfully: {len(processed)}")
    for r in processed:
        print(f"  - {r['filename']}")

    print(f"Datasets failed: {len(failed)}")
    for r in failed:
        print(f"  - {r['filename']}: {r['error']}")

    print(f"Total duplicates removed across all datasets: {total_duplicates_removed}")
    print("=" * 50)


def main() -> None:
    """
    Run the full data cleaning pipeline over all configured datasets.

    Returns
    -------
    None
    """
    results: List[Dict[str, object]] = []

    for filename in DATASET_FILENAMES:
        result = process_single_dataset(filename)
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()
