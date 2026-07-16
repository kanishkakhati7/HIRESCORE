"""
05_dataset_relationships.py

Cross-dataset relationship analysis for RealityCheck.

Reads every CSV file in `datasets/processed/` and analyzes how the datasets
relate to one another, without modifying, merging, or engineering anything.
Specifically, it:

    - Profiles each dataset individually (shape, dtypes).
    - Detects possible primary key column(s) per dataset.
    - Finds columns with matching (or near-matching) names across datasets.
    - Compares datatypes for those matching columns.
    - Estimates foreign-key relationships by measuring how well the values
      of a column in one dataset are contained within a candidate primary
      key column of another dataset.
    - Estimates overall join compatibility between dataset pairs.
    - Produces a human-readable recommendation of which datasets should be
      merged, which should stay separate, and what each dataset contributes
      to the final ML pipeline.

The report is written to `reports/dataset_relationships.txt`.

This script is strictly analytical: it never writes to, merges, or alters
any input dataset, and it performs no feature engineering or model
training.

Usage:
    python scripts/05_dataset_relationships.py
"""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

INPUT_DIR = Path("datasets/processed")
OUTPUT_PATH = Path("reports/dataset_relationships.txt")

LINE_WIDTH = 80

# A column is a primary-key candidate if it has no missing values and its
# number of unique values equals the row count.
PK_MIN_ROWS = 1

# Composite primary key search is limited to this many high-cardinality
# candidate columns, to keep the pairwise search bounded.
MAX_COMPOSITE_CANDIDATES = 6

# A child column is considered "contained" in a parent (PK-like) column if
# at least this fraction of its non-null values appear in the parent's
# value set.
STRONG_CONTAINMENT_THRESHOLD = 0.90
MODERATE_CONTAINMENT_THRESHOLD = 0.60

# Column name matching: names are normalized before comparison, and a
# trailing "_id" / "id" is stripped so e.g. "job_id" (in a fact table) can
# be matched against "id" (in a dimension table).
ID_SUFFIXES = ("_id", "id")


# --------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------

def _section_header(title: str) -> str:
    """Build a visually distinct section header for the text report.

    Args:
        title: Section title text.

    Returns:
        A formatted multi-line header string.
    """
    bar = "=" * LINE_WIDTH
    return f"\n{bar}\n{title}\n{bar}\n"


def _sub_header(title: str) -> str:
    """Build a lightweight sub-section header.

    Args:
        title: Sub-section title text.

    Returns:
        A formatted header string.
    """
    bar = "-" * LINE_WIDTH
    return f"\n{title}\n{bar}\n"


# --------------------------------------------------------------------------
# Discovery / IO
# --------------------------------------------------------------------------

def discover_csv_files(input_dir: Path) -> List[Path]:
    """Find all CSV files inside the input directory.

    Args:
        input_dir: Directory to scan for CSV files.

    Returns:
        Sorted list of CSV file paths.
    """
    if not input_dir.exists():
        return []
    return sorted(p for p in input_dir.glob("*.csv") if p.is_file())


def load_datasets(csv_paths: List[Path]) -> Tuple[Dict[str, pd.DataFrame], List[Tuple[str, str]]]:
    """Load every CSV file into a DataFrame, tolerating individual failures.

    Args:
        csv_paths: List of CSV file paths to load.

    Returns:
        Tuple of:
            - dict mapping dataset name (file stem) to its DataFrame
            - list of (filename, error_message) for files that failed to load
    """
    datasets: Dict[str, pd.DataFrame] = {}
    failures: List[Tuple[str, str]] = []

    for csv_path in csv_paths:
        print(f"Loading '{csv_path.name}'...", end=" ")
        try:
            try:
                df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, low_memory=False, encoding="latin-1")
            datasets[csv_path.stem] = df
            print(f"ok ({df.shape[0]:,} rows x {df.shape[1]:,} cols)")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"FAILED ({exc})")
            failures.append((csv_path.name, str(exc)))
            traceback.print_exc(file=sys.stderr)

    return datasets, failures


# --------------------------------------------------------------------------
# Primary key detection
# --------------------------------------------------------------------------

def detect_single_column_primary_keys(df: pd.DataFrame) -> List[str]:
    """Detect single columns that could serve as a primary key.

    A column qualifies if it has zero missing values and every value is
    unique (nunique equals row count).

    Args:
        df: Dataset to analyze.

    Returns:
        List of column names that are valid single-column primary key
        candidates.
    """
    if len(df) < PK_MIN_ROWS:
        return []

    candidates: List[str] = []
    row_count = len(df)
    for col in df.columns:
        try:
            if df[col].isna().sum() == 0 and df[col].nunique(dropna=True) == row_count:
                candidates.append(col)
        except TypeError:
            # Unhashable column types (e.g. lists) cannot be checked for
            # uniqueness; skip safely.
            continue
    return candidates


def detect_composite_primary_keys(
    df: pd.DataFrame, single_column_pks: List[str]
) -> List[Tuple[str, str]]:
    """Detect 2-column combinations that together form a unique key.

    Only run when no single-column primary key was found, and restricted to
    a bounded set of high-cardinality "id-like" candidate columns to avoid
    an expensive exhaustive search.

    Args:
        df: Dataset to analyze.
        single_column_pks: Already-detected single-column primary keys
            (if any are present, composite search is skipped).

    Returns:
        List of (col_a, col_b) tuples whose combined values are unique
        across all rows.
    """
    if single_column_pks or len(df) < PK_MIN_ROWS:
        return []

    row_count = len(df)
    candidate_cols = _rank_composite_key_candidates(df, row_count)

    composites: List[Tuple[str, str]] = []
    for i, col_a in enumerate(candidate_cols):
        for col_b in candidate_cols[i + 1:]:
            try:
                combined = df[[col_a, col_b]].dropna()
                if len(combined) == row_count and not combined.duplicated().any():
                    composites.append((col_a, col_b))
            except TypeError:
                continue
    return composites


def _rank_composite_key_candidates(df: pd.DataFrame, row_count: int) -> List[str]:
    """Select a bounded list of promising columns for composite key search.

    Prefers columns with high cardinality and/or "id"-like names.

    Args:
        df: Dataset to analyze.
        row_count: Number of rows in the dataset.

    Returns:
        Up to MAX_COMPOSITE_CANDIDATES column names, ranked by cardinality.
    """
    if row_count == 0:
        return []

    scored: List[Tuple[float, str]] = []
    for col in df.columns:
        try:
            uniqueness = df[col].nunique(dropna=True) / row_count
        except TypeError:
            continue
        name_bonus = 0.05 if "id" in col.lower() else 0.0
        if uniqueness >= 0.5:
            scored.append((uniqueness + name_bonus, col))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [col for _, col in scored[:MAX_COMPOSITE_CANDIDATES]]


# --------------------------------------------------------------------------
# Column name matching and dtype comparison
# --------------------------------------------------------------------------

def normalize_column_name(col: str) -> str:
    """Normalize a column name for cross-dataset matching.

    Lowercases the name, replaces non-alphanumeric characters with
    underscores, and strips a trailing "id"/"_id" token so that e.g.
    "job_id" and "id" can be recognized as related.

    Args:
        col: Raw column name.

    Returns:
        Normalized column name.
    """
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", col.strip().lower()).strip("_")
    for suffix in ID_SUFFIXES:
        if normalized.endswith(suffix) and normalized != suffix:
            normalized = normalized[: -len(suffix)].strip("_")
            break
    return normalized or col.strip().lower()


def dtype_family(series: pd.Series) -> str:
    """Classify a column's dtype into a broad comparability family.

    Args:
        series: Column to classify.

    Returns:
        One of "numeric", "datetime", "boolean", "text", or "other".
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(series)
    ):
        return "text"
    return "other"


def find_matching_columns(
    datasets: Dict[str, pd.DataFrame]
) -> Dict[str, List[Tuple[str, str, str]]]:
    """Find columns that share the same normalized name across datasets.

    Args:
        datasets: Mapping of dataset name to DataFrame.

    Returns:
        Dict mapping normalized column name to a list of
        (dataset_name, actual_column_name, dtype_family) tuples, restricted
        to normalized names that appear in more than one dataset.
    """
    groups: Dict[str, List[Tuple[str, str, str]]] = {}
    for dataset_name, df in datasets.items():
        for col in df.columns:
            normalized = normalize_column_name(col)
            groups.setdefault(normalized, []).append(
                (dataset_name, col, dtype_family(df[col]))
            )

    return {
        name: entries
        for name, entries in groups.items()
        if len({dataset for dataset, _, _ in entries}) > 1
    }


# --------------------------------------------------------------------------
# Foreign key / join compatibility estimation
# --------------------------------------------------------------------------

def compute_containment_ratio(child_series: pd.Series, parent_series: pd.Series) -> float:
    """Estimate what fraction of a child column's values exist in a parent column.

    Args:
        child_series: Candidate foreign-key column (the "many" side).
        parent_series: Candidate primary-key column (the "one" side).

    Returns:
        Fraction (0.0-1.0) of the child's distinct non-null values that are
        present in the parent's distinct non-null value set. Returns 0.0 if
        the child has no non-null values.
    """
    child_values = set(child_series.dropna().unique())
    if not child_values:
        return 0.0
    parent_values = set(parent_series.dropna().unique())
    if not parent_values:
        return 0.0
    matched = child_values & parent_values
    return len(matched) / len(child_values)


def classify_join_compatibility(containment_ratio: float, dtype_compatible: bool) -> str:
    """Translate a containment ratio and dtype match into a compatibility label.

    Args:
        containment_ratio: Fraction of child values found in the parent set.
        dtype_compatible: Whether the two columns share a comparable dtype
            family.

    Returns:
        One of "Strong", "Moderate", "Weak", or "Incompatible".
    """
    if not dtype_compatible:
        return "Incompatible"
    if containment_ratio >= STRONG_CONTAINMENT_THRESHOLD:
        return "Strong"
    if containment_ratio >= MODERATE_CONTAINMENT_THRESHOLD:
        return "Moderate"
    return "Weak"


def detect_foreign_key_candidates(
    datasets: Dict[str, pd.DataFrame],
    primary_keys: Dict[str, List[str]],
    matching_columns: Dict[str, List[Tuple[str, str, str]]],
) -> List[Dict[str, object]]:
    """Estimate foreign-key relationships between datasets.

    For every normalized column name shared by two or more datasets, and
    for every ordered pair of datasets where one side's column is a
    primary-key candidate on the "parent" side, estimate how well the
    "child" side's values are contained in the parent.

    Args:
        datasets: Mapping of dataset name to DataFrame.
        primary_keys: Mapping of dataset name to its single-column primary
            key candidates.
        matching_columns: Output of find_matching_columns.

    Returns:
        List of relationship dicts, each containing: normalized_name,
        child_dataset, child_column, parent_dataset, parent_column,
        containment_ratio, dtype_compatible, compatibility label.
    """
    relationships: List[Dict[str, object]] = []

    for normalized_name, entries in matching_columns.items():
        for parent_dataset, parent_col, parent_family in entries:
            if parent_col not in primary_keys.get(parent_dataset, []):
                continue  # parent side must look like a primary key

            for child_dataset, child_col, child_family in entries:
                if child_dataset == parent_dataset:
                    continue

                try:
                    child_series = datasets[child_dataset][child_col]
                    parent_series = datasets[parent_dataset][parent_col]
                    containment = compute_containment_ratio(child_series, parent_series)
                except Exception:  # noqa: BLE001 - skip unanalyzable pair
                    continue

                dtype_compatible = child_family == parent_family
                compatibility = classify_join_compatibility(containment, dtype_compatible)

                relationships.append(
                    {
                        "normalized_name": normalized_name,
                        "child_dataset": child_dataset,
                        "child_column": child_col,
                        "parent_dataset": parent_dataset,
                        "parent_column": parent_col,
                        "containment_ratio": containment,
                        "dtype_compatible": dtype_compatible,
                        "compatibility": compatibility,
                    }
                )

    relationships.sort(key=lambda r: r["containment_ratio"], reverse=True)
    return relationships


# --------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------

def build_dataset_overview_section(
    datasets: Dict[str, pd.DataFrame],
    primary_keys: Dict[str, List[str]],
    composite_keys: Dict[str, List[Tuple[str, str]]],
) -> str:
    """Build the DATASET OVERVIEW section covering every loaded dataset.

    Args:
        datasets: Mapping of dataset name to DataFrame.
        primary_keys: Mapping of dataset name to single-column PK candidates.
        composite_keys: Mapping of dataset name to composite PK candidates.

    Returns:
        Formatted section text.
    """
    lines = [_section_header("DATASET OVERVIEW")]

    for name, df in sorted(datasets.items()):
        lines.append(_sub_header(f"Dataset: {name}"))
        lines.append(f"Rows               : {len(df):,}")
        lines.append(f"Columns            : {df.shape[1]:,}")

        family_counts: Dict[str, int] = {}
        for col in df.columns:
            family = dtype_family(df[col])
            family_counts[family] = family_counts.get(family, 0) + 1
        family_summary = ", ".join(f"{fam}={cnt}" for fam, cnt in sorted(family_counts.items()))
        lines.append(f"Column type mix    : {family_summary}")

        pk_cols = primary_keys.get(name, [])
        if pk_cols:
            lines.append(f"Primary key(s)     : {', '.join(pk_cols)}")
        else:
            comp = composite_keys.get(name, [])
            if comp:
                comp_strs = [f"({a} + {b})" for a, b in comp[:3]]
                lines.append(f"Composite key(s)   : {', '.join(comp_strs)}")
            else:
                lines.append("Primary key(s)     : none detected")

    return "\n".join(lines)


def build_matching_columns_section(
    matching_columns: Dict[str, List[Tuple[str, str, str]]]
) -> str:
    """Build the MATCHING COLUMNS ACROSS DATASETS section.

    Args:
        matching_columns: Output of find_matching_columns.

    Returns:
        Formatted section text.
    """
    lines = [_section_header("MATCHING COLUMNS ACROSS DATASETS")]

    if not matching_columns:
        lines.append("No columns with matching (or near-matching) names were found.")
        return "\n".join(lines)

    for normalized_name, entries in sorted(matching_columns.items()):
        lines.append(_sub_header(f"Match group: '{normalized_name}'"))
        families = {family for _, _, family in entries}
        for dataset_name, col, family in sorted(entries):
            lines.append(f"    {dataset_name:<25}{col:<25}dtype family: {family}")
        if len(families) > 1:
            lines.append(
                "    NOTE: dtype families differ across this match group; "
                "joining on these columns may require type conversion."
            )

    return "\n".join(lines)


def build_foreign_key_section(relationships: List[Dict[str, object]]) -> str:
    """Build the FOREIGN KEY CANDIDATES & JOIN COMPATIBILITY section.

    Args:
        relationships: Output of detect_foreign_key_candidates.

    Returns:
        Formatted section text.
    """
    lines = [_section_header("FOREIGN KEY CANDIDATES & JOIN COMPATIBILITY")]

    if not relationships:
        lines.append(
            "No foreign-key-like relationships were detected. This can happen "
            "when no dataset has a clear primary key, or when no shared "
            "column names point to one."
        )
        return "\n".join(lines)

    header = (
        f"{'Child dataset.column':<38}{'-> Parent dataset.column':<38}"
        f"{'Containment':>12}{'Dtype OK':>10}{'Rating':>12}"
    )
    lines.append(header)
    lines.append("-" * LINE_WIDTH)

    for rel in relationships:
        child_ref = f"{rel['child_dataset']}.{rel['child_column']}"
        parent_ref = f"{rel['parent_dataset']}.{rel['parent_column']}"
        lines.append(
            f"{child_ref:<38}{'-> ' + parent_ref:<38}"
            f"{rel['containment_ratio'] * 100:>11.1f}%"
            f"{'yes' if rel['dtype_compatible'] else 'no':>10}"
            f"{rel['compatibility']:>12}"
        )

    return "\n".join(lines)


def describe_ml_contribution(df: pd.DataFrame) -> str:
    """Summarize what a dataset likely contributes to an ML pipeline.

    Purely descriptive: counts numeric/categorical/datetime columns and
    flags likely target-like columns by name. Performs no feature
    engineering.

    Args:
        df: Dataset to describe.

    Returns:
        A one-sentence description of the dataset's likely role.
    """
    numeric_cols = [c for c in df.columns if dtype_family(df[c]) == "numeric"]
    text_cols = [c for c in df.columns if dtype_family(df[c]) == "text"]
    datetime_cols = [c for c in df.columns if dtype_family(df[c]) == "datetime"]

    target_hints = ("target", "label", "salary", "outcome", "y")
    likely_targets = [
        c for c in df.columns
        if any(hint == c.lower() or c.lower().endswith(f"_{hint}") for hint in target_hints)
    ]

    parts = [
        f"{len(df):,} rows",
        f"{len(numeric_cols)} numeric column(s)",
        f"{len(text_cols)} categorical/text column(s)",
    ]
    if datetime_cols:
        parts.append(f"{len(datetime_cols)} datetime column(s)")
    description = ", ".join(parts) + "."

    if likely_targets:
        description += f" Possible target column(s): {', '.join(likely_targets)}."
    else:
        description += (
            " No obvious target column detected by name; likely a "
            "feature/reference table."
        )

    return description


def build_recommendations_section(
    datasets: Dict[str, pd.DataFrame],
    relationships: List[Dict[str, object]],
) -> str:
    """Build the RECOMMENDATIONS section: merge guidance and ML contribution.

    Args:
        datasets: Mapping of dataset name to DataFrame.
        relationships: Output of detect_foreign_key_candidates.

    Returns:
        Formatted section text.
    """
    lines = [_section_header("RECOMMENDATIONS")]

    strong_or_moderate = [
        r for r in relationships if r["compatibility"] in ("Strong", "Moderate")
    ]

    lines.append(_sub_header("Merge Guidance"))
    if not strong_or_moderate:
        lines.append(
            "No dataset pair showed strong or moderate join compatibility. "
            "Recommendation: keep all datasets separate for now, and revisit "
            "once shared identifier columns are added or cleaned up."
        )
    else:
        seen_pairs = set()
        for rel in strong_or_moderate:
            pair_key = tuple(sorted((rel["child_dataset"], rel["parent_dataset"])))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            lines.append(
                f"MERGE CANDIDATE: '{rel['child_dataset']}' -> '{rel['parent_dataset']}' "
                f"via {rel['child_column']} -> {rel['parent_column']} "
                f"({rel['compatibility']} join, "
                f"{rel['containment_ratio'] * 100:.1f}% containment). "
                f"A left join from '{rel['child_dataset']}' onto "
                f"'{rel['parent_dataset']}' looks safe to perform in a later "
                "pipeline stage."
            )

        all_dataset_names = set(datasets.keys())
        linked_names = {name for pair in seen_pairs for name in pair}
        unlinked = sorted(all_dataset_names - linked_names)
        if unlinked:
            lines.append(
                "KEEP SEPARATE: "
                f"{', '.join(unlinked)} showed no strong/moderate relationship "
                "to any other dataset and should remain standalone unless a "
                "shared key is introduced."
            )

    lines.append(_sub_header("Per-Dataset Contribution to the ML Pipeline"))
    for name, df in sorted(datasets.items()):
        lines.append(f"    {name}:")
        lines.append(f"        - {describe_ml_contribution(df)}")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def analyze_datasets(
    datasets: Dict[str, pd.DataFrame]
) -> Tuple[
    Dict[str, List[str]],
    Dict[str, List[Tuple[str, str]]],
    Dict[str, List[Tuple[str, str, str]]],
    List[Dict[str, object]],
]:
    """Run the full relationship analysis over a collection of datasets.

    Args:
        datasets: Mapping of dataset name to DataFrame.

    Returns:
        Tuple of (primary_keys, composite_keys, matching_columns,
        foreign_key_relationships).
    """
    primary_keys: Dict[str, List[str]] = {}
    composite_keys: Dict[str, List[Tuple[str, str]]] = {}

    for name, df in datasets.items():
        print(f"Analyzing keys for '{name}'...", end=" ")
        try:
            single_pks = detect_single_column_primary_keys(df)
            primary_keys[name] = single_pks
            if not single_pks:
                composite_keys[name] = detect_composite_primary_keys(df, single_pks)
            print("done")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"FAILED ({exc})")
            traceback.print_exc(file=sys.stderr)
            primary_keys[name] = []

    print("Matching column names across datasets...", end=" ")
    matching_columns = find_matching_columns(datasets)
    print(f"done ({len(matching_columns)} match group(s) found)")

    print("Estimating foreign-key relationships...", end=" ")
    relationships = detect_foreign_key_candidates(datasets, primary_keys, matching_columns)
    print(f"done ({len(relationships)} candidate relationship(s) found)")

    return primary_keys, composite_keys, matching_columns, relationships


def build_summary_section(
    datasets: Dict[str, pd.DataFrame],
    relationships: List[Dict[str, object]],
    load_failures: List[Tuple[str, str]],
) -> str:
    """Build the final SUMMARY section of the report.

    Args:
        datasets: Mapping of dataset name to DataFrame.
        relationships: Output of detect_foreign_key_candidates.
        load_failures: List of (filename, error_message) for datasets that
            failed to load.

    Returns:
        Formatted section text.
    """
    lines = [_section_header("SUMMARY")]

    lines.append(f"Datasets analyzed          : {len(datasets)}")
    for name in sorted(datasets.keys()):
        lines.append(f"    - {name}")

    lines.append(f"\nDatasets failed to load    : {len(load_failures)}")
    for filename, error in load_failures:
        lines.append(f"    - {filename}: {error}")

    strong = sum(1 for r in relationships if r["compatibility"] == "Strong")
    moderate = sum(1 for r in relationships if r["compatibility"] == "Moderate")
    weak = sum(1 for r in relationships if r["compatibility"] == "Weak")

    lines.append(f"\nForeign-key candidates     : {len(relationships)}")
    lines.append(f"    - Strong join compatibility   : {strong}")
    lines.append(f"    - Moderate join compatibility : {moderate}")
    lines.append(f"    - Weak join compatibility     : {weak}")

    lines.append(f"\nReport location            : {OUTPUT_PATH.resolve()}")

    return "\n".join(lines)


def build_full_report(
    datasets: Dict[str, pd.DataFrame],
    primary_keys: Dict[str, List[str]],
    composite_keys: Dict[str, List[Tuple[str, str]]],
    matching_columns: Dict[str, List[Tuple[str, str, str]]],
    relationships: List[Dict[str, object]],
    load_failures: List[Tuple[str, str]],
) -> str:
    """Assemble the complete human-readable report text.

    Args:
        datasets: Mapping of dataset name to DataFrame.
        primary_keys: Mapping of dataset name to single-column PK candidates.
        composite_keys: Mapping of dataset name to composite PK candidates.
        matching_columns: Output of find_matching_columns.
        relationships: Output of detect_foreign_key_candidates.
        load_failures: List of (filename, error_message) for datasets that
            failed to load.

    Returns:
        Full report text, ready to write to disk.
    """
    sections = [
        "DATASET RELATIONSHIP REPORT",
        build_dataset_overview_section(datasets, primary_keys, composite_keys),
        build_matching_columns_section(matching_columns),
        build_foreign_key_section(relationships),
        build_recommendations_section(datasets, relationships),
        build_summary_section(datasets, relationships, load_failures),
        "",
    ]
    return "\n".join(sections)


def run_relationship_analysis(
    input_dir: Path = INPUT_DIR, output_path: Path = OUTPUT_PATH
) -> None:
    """Run the end-to-end dataset relationship analysis.

    Args:
        input_dir: Directory containing processed CSV files.
        output_path: Path to write the final text report to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_files = discover_csv_files(input_dir)
    if not csv_files:
        print(f"No CSV files found in '{input_dir}'. Nothing to do.")
        return

    print(f"Found {len(csv_files)} CSV file(s) in '{input_dir}'.\n")
    datasets, load_failures = load_datasets(csv_files)

    if not datasets:
        print("\nNo datasets could be loaded successfully. Aborting analysis.")
        return

    print()
    primary_keys, composite_keys, matching_columns, relationships = analyze_datasets(datasets)

    print("\nWriting report...", end=" ")
    try:
        report_text = build_full_report(
            datasets, primary_keys, composite_keys, matching_columns, relationships, load_failures
        )
        output_path.write_text(report_text, encoding="utf-8")
        print(f"done -> {output_path}")
    except Exception as exc:  # noqa: BLE001 - surface the failure clearly
        print(f"FAILED ({exc})")
        traceback.print_exc(file=sys.stderr)
        return

    print(build_summary_section(datasets, relationships, load_failures))


def main() -> None:
    """Entry point for the dataset relationship analysis script."""
    run_relationship_analysis()


if __name__ == "__main__":
    main()