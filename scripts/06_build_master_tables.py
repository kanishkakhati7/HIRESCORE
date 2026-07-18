"""
06_build_master_datasets.py

Purpose
-------
First data-integration stage for HIRESCORE. Builds two master datasets
from the cleaned CSVs in datasets/processed/:

    1. jobs_master.csv
       Base table: postings.csv
       Left-joined, in order, with:
           - companies.csv           on company_id
           - company_industries.csv  on company_id (pre-aggregated, since
             a single company may legitimately map to more than one
             industry; aggregating first prevents postings rows from
             being duplicated by the join)
           - salaries.csv            on job_id
       Every posting row is preserved even when related company or
       salary information could not be found. Any incoming column name
       that would collide with an existing postings column is suffixed
       rather than overwritten.

    2. resumes_master.csv
       Base table: Resume.csv, with exact duplicate rows removed and
       rows with empty/missing resume text removed. No NLP, no
       feature engineering, no column changes.

Explicitly ignored (per project scope): global_ai_jobs.csv,
industries.csv, job_skills.csv, skills.csv. These are never read by
this script.

Input directory:  datasets/processed/
Output directory: datasets/final/
Report:           reports/master_dataset_report.txt

This script only reads from datasets/processed/ and never modifies or
overwrites anything in that directory. It intentionally does NOT:
    - engineer features
    - perform NLP / text extraction
    - train models
    - produce visualizations

Usage
-----
    python scripts/06_build_master_datasets.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
FINAL_DIR = PROJECT_ROOT / "datasets" / "final"
REPORTS_DIR = PROJECT_ROOT / "reports"

JOBS_MASTER_PATH = FINAL_DIR / "jobs_master.csv"
RESUMES_MASTER_PATH = FINAL_DIR / "resumes_master.csv"
REPORT_PATH = REPORTS_DIR / "master_dataset_report.txt"

# Datasets read and joined into jobs_master.csv, in join order.
POSTINGS_FILENAME = "postings.csv"
COMPANIES_FILENAME = "companies.csv"
COMPANY_INDUSTRIES_FILENAME = "company_industries.csv"
SALARIES_FILENAME = "salaries.csv"

# Dataset used to build resumes_master.csv.
RESUME_FILENAME = "Resume.csv"
RESUME_TEXT_COLUMN = "resume_str"

# Datasets that are explicitly in scope for the project but out of scope
# for this integration stage. Never read by this script; listed here only
# so the report can document that the exclusion was deliberate.
IGNORED_DATASETS: List[str] = [
    "global_ai_jobs.csv",
    "industries.csv",
    "job_skills.csv",
    "skills.csv",
]

# Join keys.
COMPANY_KEY = "company_id"
JOB_KEY = "job_id"

# Suffixes applied to incoming columns that collide with an existing
# jobs_master column. The base (postings) column always keeps its
# original, unsuffixed name.
COMPANIES_SUFFIX = "_company"
COMPANY_INDUSTRIES_SUFFIX = "_company_industry"
SALARIES_SUFFIX = "_salaries"

INDUSTRY_SEPARATOR = "; "

logger = logging.getLogger("build_master_datasets")


# ---------------------------------------------------------------------------
# Diagnostics containers
# ---------------------------------------------------------------------------

@dataclass
class MergeStepDiagnostics:
    """Diagnostics captured for a single left-join step.

    Attributes
    ----------
    step_name : str
        Human-readable name of the merge step (e.g. "postings + companies").
    join_key : str
        Column name used as the join key.
    rows_before : int
        Row count of the left-hand table before this merge.
    rows_after : int
        Row count of the result after this merge.
    right_table_rows : Optional[int]
        Row count of the right-hand (joined-in) table, or None if that
        table could not be loaded.
    duplicate_keys_in_right_table : Optional[int]
        Number of duplicated join-key values found in the right-hand
        table before merging (a non-zero value here is a fan-out risk
        and is flagged).
    unmatched_key_count : Optional[int]
        Number of left-hand rows whose (non-null) join key had no match
        in the right-hand table.
    null_key_count : int
        Number of left-hand rows whose join key was itself missing.
    added_columns : List[str]
        Column names added to the result by this merge.
    skipped : bool
        True if this step was skipped (e.g. source file missing or
        failed to load).
    skip_reason : Optional[str]
        Explanation for why the step was skipped, if applicable.
    """

    step_name: str
    join_key: str
    rows_before: int = 0
    rows_after: int = 0
    right_table_rows: Optional[int] = None
    duplicate_keys_in_right_table: Optional[int] = None
    unmatched_key_count: Optional[int] = None
    null_key_count: int = 0
    added_columns: List[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class ResumeCleaningDiagnostics:
    """Diagnostics captured while building resumes_master.csv.

    Attributes
    ----------
    rows_before : int
        Row count of Resume.csv as loaded.
    duplicate_rows_removed : int
        Number of exact duplicate rows removed.
    empty_text_rows_removed : int
        Number of rows removed for having missing/empty resume text.
    rows_after : int
        Final row count of resumes_master.csv.
    """

    rows_before: int = 0
    duplicate_rows_removed: int = 0
    empty_text_rows_removed: int = 0
    rows_after: int = 0


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """
    Configure console logging for this script.

    Returns
    -------
    None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

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

    try:
        return pd.read_csv(file_path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, low_memory=False, encoding="latin-1")


def try_load_dataset(filename: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Attempt to load a dataset from PROCESSED_DIR, never raising.

    Parameters
    ----------
    filename : str
        Name of the CSV file inside PROCESSED_DIR.

    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[str]]
        (dataframe, None) on success, or (None, error_message) on failure.
    """
    file_path = PROCESSED_DIR / filename
    try:
        df = load_dataset(file_path)
        logger.info("Loaded %s (%d rows, %d columns)", filename, len(df), df.shape[1])
        return df, None
    except Exception as exc:  # noqa: BLE001 - report and continue
        message = str(exc)
        logger.error("Failed to load %s: %s", filename, message)
        return None, message


# ---------------------------------------------------------------------------
# jobs_master.csv construction
# ---------------------------------------------------------------------------

def aggregate_company_industries(company_industries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse company_industries.csv to one row per company_id.

    A single company can legitimately be tagged with more than one
    industry. Aggregating before the join (rather than joining the raw
    table) guarantees the jobs_master merge cannot multiply posting rows.

    Parameters
    ----------
    company_industries_df : pd.DataFrame
        Raw company_industries dataset with columns "company_id", "industry".

    Returns
    -------
    pd.DataFrame
        One row per company_id, with columns:
            - "company_id"
            - "industries": all distinct industries for that company,
              joined with "; "
            - "industry_count": number of distinct industries
    """
    grouped = (
        company_industries_df.dropna(subset=[COMPANY_KEY, "industry"])
        .groupby(COMPANY_KEY)["industry"]
        .agg(lambda values: sorted(set(values)))
    )

    aggregated = pd.DataFrame(
        {
            COMPANY_KEY: grouped.index,
            "industries": grouped.apply(lambda values: INDUSTRY_SEPARATOR.join(values)),
            "industry_count": grouped.apply(len),
        }
    ).reset_index(drop=True)

    return aggregated


def count_duplicate_keys(df: pd.DataFrame, key_col: str) -> int:
    """
    Count duplicated (non-null) values in a candidate join key column.

    Parameters
    ----------
    df : pd.DataFrame
        The table to inspect.
    key_col : str
        Name of the key column.

    Returns
    -------
    int
        Number of duplicated key occurrences, or 0 if the column is
        absent from the DataFrame.
    """
    if key_col not in df.columns:
        return 0
    return int(df[key_col].dropna().duplicated().sum())


def count_unmatched_keys(left_df: pd.DataFrame, left_key: str, right_key_values: pd.Series) -> Tuple[int, int]:
    """
    Count how many left-hand rows have a join key with no match on the right.

    Parameters
    ----------
    left_df : pd.DataFrame
        The left-hand (base) table.
    left_key : str
        Name of the join key column in the left-hand table.
    right_key_values : pd.Series
        The set of key values present in the right-hand table.

    Returns
    -------
    Tuple[int, int]
        (unmatched_non_null_key_count, null_key_count)
    """
    key_series = left_df[left_key]
    null_key_count = int(key_series.isna().sum())

    non_null_keys = key_series.dropna()
    unmatched_count = int((~non_null_keys.isin(set(right_key_values.dropna()))).sum())

    return unmatched_count, null_key_count


def left_join_step(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    join_key: str,
    suffix: str,
    step_name: str,
) -> Tuple[pd.DataFrame, MergeStepDiagnostics]:
    """
    Perform one non-destructive left join and collect diagnostics for it.

    Existing left-hand columns are never overwritten: any right-hand
    column name that collides with a left-hand column (other than the
    join key itself) is suffixed.

    Parameters
    ----------
    left_df : pd.DataFrame
        The base (left-hand) table. All of its rows are preserved.
    right_df : pd.DataFrame
        The table being joined in.
    join_key : str
        Column name to join on (must exist in both tables).
    suffix : str
        Suffix applied to right-hand columns that collide with left-hand
        column names.
    step_name : str
        Human-readable label for this step, used in diagnostics/logging.

    Returns
    -------
    Tuple[pd.DataFrame, MergeStepDiagnostics]
        The merged DataFrame and diagnostics describing the join.
    """
    rows_before = len(left_df)
    columns_before = set(left_df.columns)

    duplicate_keys_in_right = count_duplicate_keys(right_df, join_key)
    if duplicate_keys_in_right > 0:
        logger.warning(
            "%s: right-hand table has %d duplicated '%s' values; "
            "this can duplicate rows in the join result.",
            step_name,
            duplicate_keys_in_right,
            join_key,
        )

    unmatched_count, null_key_count = count_unmatched_keys(
        left_df, join_key, right_df[join_key]
    )

    merged = left_df.merge(
        right_df,
        on=join_key,
        how="left",
        suffixes=("", suffix),
    )

    added_columns = [col for col in merged.columns if col not in columns_before]

    diagnostics = MergeStepDiagnostics(
        step_name=step_name,
        join_key=join_key,
        rows_before=rows_before,
        rows_after=len(merged),
        right_table_rows=len(right_df),
        duplicate_keys_in_right_table=duplicate_keys_in_right,
        unmatched_key_count=unmatched_count,
        null_key_count=null_key_count,
        added_columns=added_columns,
    )

    if diagnostics.rows_after != rows_before:
        logger.warning(
            "%s: row count changed from %d to %d after the join "
            "(expected an unchanged count for a clean left join).",
            step_name,
            rows_before,
            diagnostics.rows_after,
        )
    else:
        logger.info(
            "%s: %d rows preserved, %d column(s) added, %d unmatched key(s).",
            step_name,
            rows_before,
            len(added_columns),
            unmatched_count,
        )

    return merged, diagnostics


def build_jobs_master() -> Tuple[Optional[pd.DataFrame], List[MergeStepDiagnostics]]:
    """
    Build jobs_master.csv by left-joining postings with companies,
    company_industries, and salaries.

    Every step is attempted independently: if a source file is missing
    or fails to load, that step is skipped (logged and recorded in
    diagnostics) rather than aborting the whole pipeline, so the
    resulting jobs_master still contains every postings row with
    whatever enrichment was available.

    Returns
    -------
    Tuple[Optional[pd.DataFrame], List[MergeStepDiagnostics]]
        The built DataFrame (or None if postings.csv itself could not be
        loaded, since it is the mandatory base table) and the list of
        per-step diagnostics.
    """
    diagnostics: List[MergeStepDiagnostics] = []

    postings_df, error = try_load_dataset(POSTINGS_FILENAME)
    if postings_df is None:
        logger.error(
            "Cannot build jobs_master.csv without postings.csv: %s", error
        )
        return None, diagnostics

    jobs_master = postings_df

    # Step 1: postings + companies, on company_id.
    companies_df, error = try_load_dataset(COMPANIES_FILENAME)
    if companies_df is None:
        diagnostics.append(
            MergeStepDiagnostics(
                step_name="postings + companies",
                join_key=COMPANY_KEY,
                rows_before=len(jobs_master),
                rows_after=len(jobs_master),
                skipped=True,
                skip_reason=error,
            )
        )
    else:
        jobs_master, step_diag = left_join_step(
            jobs_master,
            companies_df,
            join_key=COMPANY_KEY,
            suffix=COMPANIES_SUFFIX,
            step_name="postings + companies",
        )
        diagnostics.append(step_diag)

    # Step 2: (postings + companies) + company_industries, on company_id.
    company_industries_df, error = try_load_dataset(COMPANY_INDUSTRIES_FILENAME)
    if company_industries_df is None:
        diagnostics.append(
            MergeStepDiagnostics(
                step_name="+ company_industries (aggregated)",
                join_key=COMPANY_KEY,
                rows_before=len(jobs_master),
                rows_after=len(jobs_master),
                skipped=True,
                skip_reason=error,
            )
        )
    else:
        aggregated_industries = aggregate_company_industries(company_industries_df)
        jobs_master, step_diag = left_join_step(
            jobs_master,
            aggregated_industries,
            join_key=COMPANY_KEY,
            suffix=COMPANY_INDUSTRIES_SUFFIX,
            step_name="+ company_industries (aggregated)",
        )
        diagnostics.append(step_diag)

    # Step 3: (postings + companies + company_industries) + salaries, on job_id.
    salaries_df, error = try_load_dataset(SALARIES_FILENAME)
    if salaries_df is None:
        diagnostics.append(
            MergeStepDiagnostics(
                step_name="+ salaries",
                join_key=JOB_KEY,
                rows_before=len(jobs_master),
                rows_after=len(jobs_master),
                skipped=True,
                skip_reason=error,
            )
        )
    else:
        jobs_master, step_diag = left_join_step(
            jobs_master,
            salaries_df,
            join_key=JOB_KEY,
            suffix=SALARIES_SUFFIX,
            step_name="+ salaries",
        )
        diagnostics.append(step_diag)

    return jobs_master, diagnostics


# ---------------------------------------------------------------------------
# resumes_master.csv construction
# ---------------------------------------------------------------------------

def clean_resume_rows(resume_df: pd.DataFrame) -> Tuple[pd.DataFrame, ResumeCleaningDiagnostics]:
    """
    Remove exact duplicate rows and rows with empty resume text.

    No column values are modified, no features are engineered, and no
    NLP is performed. This mirrors the non-destructive philosophy of
    02_clean_data.py, applied a second time at the master-dataset stage.

    Parameters
    ----------
    resume_df : pd.DataFrame
        The raw Resume.csv contents.

    Returns
    -------
    Tuple[pd.DataFrame, ResumeCleaningDiagnostics]
        The cleaned DataFrame and diagnostics describing what was removed.
    """
    rows_before = len(resume_df)

    deduplicated = resume_df.drop_duplicates()
    duplicate_rows_removed = rows_before - len(deduplicated)

    if RESUME_TEXT_COLUMN in deduplicated.columns:
        text = deduplicated[RESUME_TEXT_COLUMN]
        is_empty = text.isna() | (text.astype(str).str.strip() == "")
        cleaned = deduplicated.loc[~is_empty].reset_index(drop=True)
    else:
        logger.warning(
            "Column '%s' not found in %s; skipping the empty-text filter.",
            RESUME_TEXT_COLUMN,
            RESUME_FILENAME,
        )
        cleaned = deduplicated.reset_index(drop=True)

    empty_text_rows_removed = len(deduplicated) - len(cleaned)

    diagnostics = ResumeCleaningDiagnostics(
        rows_before=rows_before,
        duplicate_rows_removed=duplicate_rows_removed,
        empty_text_rows_removed=empty_text_rows_removed,
        rows_after=len(cleaned),
    )

    return cleaned, diagnostics


def build_resumes_master() -> Tuple[Optional[pd.DataFrame], Optional[ResumeCleaningDiagnostics]]:
    """
    Build resumes_master.csv from Resume.csv.

    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[ResumeCleaningDiagnostics]]
        The cleaned DataFrame and its diagnostics, or (None, None) if
        Resume.csv could not be loaded.
    """
    resume_df, error = try_load_dataset(RESUME_FILENAME)
    if resume_df is None:
        logger.error("Cannot build resumes_master.csv: %s", error)
        return None, None

    cleaned_df, diagnostics = clean_resume_rows(resume_df)
    logger.info(
        "resumes_master: %d -> %d rows (%d duplicate row(s), %d empty-text row(s) removed).",
        diagnostics.rows_before,
        diagnostics.rows_after,
        diagnostics.duplicate_rows_removed,
        diagnostics.empty_text_rows_removed,
    )

    return cleaned_df, diagnostics


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save a DataFrame to disk as a CSV file, creating parent directories
    as needed.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    output_path : Path
        Destination file path (including filename).

    Returns
    -------
    None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def compute_memory_mb(df: pd.DataFrame) -> float:
    """
    Compute the deep in-memory footprint of a DataFrame in megabytes.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to measure.

    Returns
    -------
    float
        Memory usage in megabytes, rounded to 4 decimal places.
    """
    total_bytes = df.memory_usage(deep=True).sum()
    return round(float(total_bytes / (1024 ** 2)), 4)


def format_schema(df: pd.DataFrame) -> str:
    """
    Format a DataFrame's column names and dtypes as aligned text lines.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose schema should be formatted.

    Returns
    -------
    str
        One "column_name: dtype" line per column.
    """
    lines = [f"  {col:<35} {str(df[col].dtype)}" for col in df.columns]
    return "\n".join(lines)


def format_jobs_master_section(
    jobs_master: Optional[pd.DataFrame], diagnostics: List[MergeStepDiagnostics]
) -> str:
    """
    Format the jobs_master.csv section of the report.

    Parameters
    ----------
    jobs_master : Optional[pd.DataFrame]
        The built jobs_master DataFrame, or None if it could not be built.
    diagnostics : List[MergeStepDiagnostics]
        Per-step diagnostics collected during the join.

    Returns
    -------
    str
        Formatted report section.
    """
    lines = ["=" * 80, "JOBS_MASTER.CSV", "=" * 80, ""]

    if jobs_master is None:
        lines.append("FAILED TO BUILD: postings.csv could not be loaded.")
        lines.append("")
        return "\n".join(lines)

    for step in diagnostics:
        lines.append(f"Step: {step.step_name}")
        lines.append(f"  Join key: {step.join_key}")

        if step.skipped:
            lines.append(f"  SKIPPED: {step.skip_reason}")
            lines.append("")
            continue

        lines.append(f"  Rows before join: {step.rows_before:,}")
        lines.append(f"  Rows after join: {step.rows_after:,}")
        lines.append(f"  Right-hand table rows: {step.right_table_rows:,}")
        lines.append(
            f"  Duplicate join keys in right-hand table: "
            f"{step.duplicate_keys_in_right_table:,}"
        )
        lines.append(f"  Null join key in left-hand table: {step.null_key_count:,}")
        lines.append(f"  Unmatched (no match found): {step.unmatched_key_count:,}")
        lines.append(
            f"  Columns added ({len(step.added_columns)}): "
            f"{', '.join(step.added_columns) if step.added_columns else 'None'}"
        )
        lines.append("")

    lines.append("-" * 80)
    lines.append("FINAL JOBS_MASTER SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Final row count: {len(jobs_master):,}")
    lines.append(f"Final column count: {jobs_master.shape[1]:,}")
    lines.append(f"Memory usage: {compute_memory_mb(jobs_master):.2f} MB")
    lines.append("")
    lines.append("Final schema:")
    lines.append(format_schema(jobs_master))
    lines.append("")

    return "\n".join(lines)


def format_resumes_master_section(
    resumes_master: Optional[pd.DataFrame], diagnostics: Optional[ResumeCleaningDiagnostics]
) -> str:
    """
    Format the resumes_master.csv section of the report.

    Parameters
    ----------
    resumes_master : Optional[pd.DataFrame]
        The built resumes_master DataFrame, or None if it could not be built.
    diagnostics : Optional[ResumeCleaningDiagnostics]
        Diagnostics collected during cleaning.

    Returns
    -------
    str
        Formatted report section.
    """
    lines = ["=" * 80, "RESUMES_MASTER.CSV", "=" * 80, ""]

    if resumes_master is None or diagnostics is None:
        lines.append("FAILED TO BUILD: Resume.csv could not be loaded.")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"Rows before cleaning: {diagnostics.rows_before:,}")
    lines.append(f"Duplicate rows removed: {diagnostics.duplicate_rows_removed:,}")
    lines.append(f"Empty resume-text rows removed: {diagnostics.empty_text_rows_removed:,}")
    lines.append(f"Rows after cleaning: {diagnostics.rows_after:,}")
    lines.append("")
    lines.append(f"Memory usage: {compute_memory_mb(resumes_master):.2f} MB")
    lines.append("")
    lines.append("Final schema:")
    lines.append(format_schema(resumes_master))
    lines.append("")

    return "\n".join(lines)


def format_ignored_datasets_section() -> str:
    """
    Format the section documenting datasets deliberately excluded from
    this integration stage.

    Returns
    -------
    str
        Formatted report section.
    """
    lines = ["=" * 80, "IGNORED DATASETS (OUT OF SCOPE FOR THIS STAGE)", "=" * 80, ""]
    for filename in IGNORED_DATASETS:
        lines.append(f"  - {filename}")
    lines.append("")
    return "\n".join(lines)


def build_report(
    jobs_master: Optional[pd.DataFrame],
    jobs_diagnostics: List[MergeStepDiagnostics],
    resumes_master: Optional[pd.DataFrame],
    resumes_diagnostics: Optional[ResumeCleaningDiagnostics],
) -> str:
    """
    Assemble the full master-dataset report.

    Parameters
    ----------
    jobs_master : Optional[pd.DataFrame]
        The built jobs_master DataFrame, or None on failure.
    jobs_diagnostics : List[MergeStepDiagnostics]
        Per-step join diagnostics for jobs_master.
    resumes_master : Optional[pd.DataFrame]
        The built resumes_master DataFrame, or None on failure.
    resumes_diagnostics : Optional[ResumeCleaningDiagnostics]
        Cleaning diagnostics for resumes_master.

    Returns
    -------
    str
        The complete, formatted report text.
    """
    header = [
        "HIRESCORE - MASTER DATASET BUILD REPORT",
        "=" * 80,
        "",
    ]

    sections = [
        format_jobs_master_section(jobs_master, jobs_diagnostics),
        format_resumes_master_section(resumes_master, resumes_diagnostics),
        format_ignored_datasets_section(),
    ]

    return "\n".join(header) + "\n".join(sections)


def save_report(report_text: str, output_path: Path) -> None:
    """
    Save the report text to disk, creating parent directories as needed.

    Parameters
    ----------
    report_text : str
        The full report content to write.
    output_path : Path
        Destination file path.

    Returns
    -------
    None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the full master-dataset build pipeline.

    Returns
    -------
    None
    """
    configure_logging()

    logger.info("Starting HIRESCORE master dataset build.")
    logger.info(
        "Ignored datasets (out of scope): %s", ", ".join(IGNORED_DATASETS)
    )

    jobs_master, jobs_diagnostics = build_jobs_master()
    if jobs_master is not None:
        try:
            save_dataframe(jobs_master, JOBS_MASTER_PATH)
            logger.info("Saved jobs_master.csv to %s", JOBS_MASTER_PATH)
        except Exception as exc:  # noqa: BLE001 - report and continue
            logger.error("Failed to save jobs_master.csv: %s", exc)

    resumes_master, resumes_diagnostics = build_resumes_master()
    if resumes_master is not None:
        try:
            save_dataframe(resumes_master, RESUMES_MASTER_PATH)
            logger.info("Saved resumes_master.csv to %s", RESUMES_MASTER_PATH)
        except Exception as exc:  # noqa: BLE001 - report and continue
            logger.error("Failed to save resumes_master.csv: %s", exc)

    try:
        report_text = build_report(
            jobs_master, jobs_diagnostics, resumes_master, resumes_diagnostics
        )
        save_report(report_text, REPORT_PATH)
        logger.info("Saved master dataset report to %s", REPORT_PATH)
    except Exception as exc:  # noqa: BLE001 - report and continue
        logger.error("Failed to build/save master dataset report: %s", exc)

    logger.info("Master dataset build complete.")


if __name__ == "__main__":
    main()