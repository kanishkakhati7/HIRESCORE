"""
04_eda.py

Exploratory Data Analysis (EDA) for the HIRESCORE project.

Reads every CSV file inside `datasets/processed/` and, for each dataset,
writes a plain-text EDA report to `reports/eda/<dataset_name>_eda.txt`
covering: dataset overview, per-column information, numeric analysis with
IQR outlier detection, Pearson correlation analysis, categorical analysis,
data quality checks, ML readiness classification, rule-based observations,
and cleaning/encoding/scaling recommendations.

This script is strictly read-only with respect to input data: it never
modifies, merges, or engineers features from the source CSVs, and it
produces no plots (text reports only).

Usage:
    python scripts/04_eda.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

INPUT_DIR = Path("datasets/processed")
OUTPUT_DIR = Path("reports/eda")

LINE_WIDTH = 80
TOP_N_CATEGORICAL_VALUES = 20

HIGH_MISSING_THRESHOLD_PCT = 70.0
HIGH_CORRELATION_THRESHOLD = 0.80
ID_UNIQUE_RATIO_THRESHOLD = 0.95
HIGH_CARDINALITY_UNIQUE_RATIO = 0.50
HIGH_CARDINALITY_MIN_UNIQUE = 50

TARGET_NAME_HINTS = (
    "target",
    "label",
    "y",
    "outcome",
    "hired",
    "hire",
    "selected",
    "selection",
    "status",
    "result",
    "score",
    "offer",
)

logger = logging.getLogger("hirescore.eda")


# --------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------

def configure_logging() -> None:
    """Configure clean, human-readable console logging for this script.

    Sets the root logger for this module to INFO level with a concise
    timestamped format. Safe to call multiple times.
    """
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------

def section_header(title: str) -> str:
    """Build a visually distinct section header for the text report.

    Args:
        title: Section title text.

    Returns:
        A formatted multi-line header string.
    """
    bar = "=" * LINE_WIDTH
    return f"\n{bar}\n{title}\n{bar}\n"


def sub_header(title: str) -> str:
    """Build a lightweight sub-section header.

    Args:
        title: Sub-section title text.

    Returns:
        A formatted header string.
    """
    bar = "-" * LINE_WIDTH
    return f"\n{title}\n{bar}\n"


def format_bytes(num_bytes: float) -> str:
    """Convert a byte count into a human-readable string.

    Args:
        num_bytes: Number of bytes.

    Returns:
        Human-readable size string (e.g. "3.42 MB").
    """
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


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
    return sorted(path for path in input_dir.glob("*.csv") if path.is_file())


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame, tolerating non-UTF-8 encodings.

    Attempts UTF-8 first and falls back to Latin-1 if the file cannot be
    decoded as UTF-8.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        Exception: Propagates any error encountered while reading the CSV.
    """
    try:
        return pd.read_csv(csv_path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, low_memory=False, encoding="latin-1")


# --------------------------------------------------------------------------
# Shared dtype helpers
# --------------------------------------------------------------------------

def is_numeric_column(series: pd.Series) -> bool:
    """Determine whether a column should be treated as numeric.

    Args:
        series: Column to inspect.

    Returns:
        True if the column has a numeric dtype and is not boolean.
    """
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def is_categorical_column(series: pd.Series) -> bool:
    """Determine whether a column should be treated as categorical/text.

    Args:
        series: Column to inspect.

    Returns:
        True if the column dtype is object, category, string, or boolean.
    """
    return (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


# --------------------------------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------------------------------

def build_overview_section(dataframe: pd.DataFrame, csv_path: Path) -> str:
    """Build the DATASET OVERVIEW section.

    Args:
        dataframe: Dataset to analyze.
        csv_path: Path to the source CSV file.

    Returns:
        Formatted section text.
    """
    memory_bytes = dataframe.memory_usage(deep=True).sum()
    duplicate_rows = int(dataframe.duplicated().sum())

    lines = [section_header("DATASET OVERVIEW")]
    lines.append(f"Filename           : {csv_path.name}")
    lines.append(f"Rows               : {len(dataframe):,}")
    lines.append(f"Columns            : {dataframe.shape[1]:,}")
    lines.append(f"Memory usage       : {format_bytes(memory_bytes)}")
    lines.append(f"Duplicate rows     : {duplicate_rows:,}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# COLUMN INFORMATION
# --------------------------------------------------------------------------

def build_column_information_section(dataframe: pd.DataFrame) -> Tuple[str, pd.Series]:
    """Build the COLUMN INFORMATION section.

    Args:
        dataframe: Dataset to analyze.

    Returns:
        Tuple of (formatted section text, missing percentage series indexed
        by column name) so downstream sections can reuse the computation.
    """
    row_count = len(dataframe)
    missing_count = dataframe.isna().sum()
    missing_pct = (missing_count / row_count * 100) if row_count > 0 else missing_count * 0.0

    lines = [section_header("COLUMN INFORMATION")]
    header = f"{'Column':<30}{'Dtype':<15}{'Unique':>10}{'Missing':>12}{'Missing %':>12}"
    lines.append(header)
    lines.append("-" * LINE_WIDTH)

    for column in dataframe.columns:
        try:
            unique_count = dataframe[column].nunique(dropna=True)
        except TypeError:
            unique_count = -1
        lines.append(
            f"{str(column):<30}{str(dataframe[column].dtype):<15}"
            f"{unique_count:>10,}{int(missing_count[column]):>12,}"
            f"{missing_pct[column]:>11.2f}%"
        )

    return "\n".join(lines), missing_pct


# --------------------------------------------------------------------------
# NUMERIC ANALYSIS
# --------------------------------------------------------------------------

def compute_iqr_outlier_count(series: pd.Series) -> Tuple[int, float]:
    """Count outliers in a numeric series using the IQR method.

    A value is considered an outlier if it falls outside
    [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR].

    Args:
        series: Numeric series (missing values are dropped internally).

    Returns:
        Tuple of (outlier_count, outlier_percentage). Returns (0, 0.0) if
        the IQR cannot be computed (e.g. empty series or zero IQR).
    """
    clean_series = series.dropna()
    if clean_series.empty:
        return 0, 0.0

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or pd.isna(iqr):
        return 0, 0.0

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (clean_series < lower_bound) | (clean_series > upper_bound)
    outlier_count = int(outlier_mask.sum())
    outlier_pct = (outlier_count / len(clean_series)) * 100
    return outlier_count, outlier_pct


def build_numeric_section(
    dataframe: pd.DataFrame,
) -> Tuple[str, List[str], Dict[str, int]]:
    """Build the NUMERIC ANALYSIS section.

    Includes descriptive statistics (count, mean, std, min, Q1, median, Q3,
    max) and IQR-based outlier counts for every numeric column.

    Args:
        dataframe: Dataset to analyze.

    Returns:
        Tuple of (formatted section text, list of numeric column names,
        dict mapping numeric column name to outlier count).
    """
    numeric_columns = [col for col in dataframe.columns if is_numeric_column(dataframe[col])]
    lines = [section_header("NUMERIC ANALYSIS")]
    outlier_counts: Dict[str, int] = {}

    if not numeric_columns:
        lines.append("No numeric columns found.")
        return "\n".join(lines), numeric_columns, outlier_counts

    for column in numeric_columns:
        series = dataframe[column].dropna()
        lines.append(sub_header(f"Column: {column}"))

        if series.empty:
            lines.append("No non-missing numeric values available.")
            outlier_counts[column] = 0
            continue

        description = series.describe(percentiles=[0.25, 0.5, 0.75])
        lines.append(f"Count              : {int(description['count']):,}")
        lines.append(f"Mean               : {description['mean']:.4f}")
        lines.append(f"Std                : {description['std']:.4f}")
        lines.append(f"Min                : {description['min']:.4f}")
        lines.append(f"Q1                 : {description['25%']:.4f}")
        lines.append(f"Median             : {description['50%']:.4f}")
        lines.append(f"Q3                 : {description['75%']:.4f}")
        lines.append(f"Max                : {description['max']:.4f}")

        outlier_count, outlier_pct = compute_iqr_outlier_count(series)
        outlier_counts[column] = outlier_count
        lines.append(f"IQR outlier count  : {outlier_count:,}")
        lines.append(f"IQR outlier %      : {outlier_pct:.2f}%")

    return "\n".join(lines), numeric_columns, outlier_counts


# --------------------------------------------------------------------------
# CORRELATION ANALYSIS
# --------------------------------------------------------------------------

def build_correlation_section(
    dataframe: pd.DataFrame, numeric_columns: List[str]
) -> Tuple[str, Optional[pd.DataFrame], List[Tuple[str, str, float]]]:
    """Build the CORRELATION ANALYSIS section.

    Args:
        dataframe: Dataset to analyze.
        numeric_columns: List of numeric column names.

    Returns:
        Tuple of (formatted section text, correlation matrix or None if
        fewer than 2 numeric columns exist, list of highly correlated
        (col_a, col_b, correlation) triples with |r| >= threshold).
    """
    lines = [section_header("CORRELATION ANALYSIS")]

    if len(numeric_columns) < 2:
        lines.append("Fewer than 2 numeric columns; correlation matrix skipped.")
        return "\n".join(lines), None, []

    correlation_matrix = dataframe[numeric_columns].corr(method="pearson", numeric_only=True)
    lines.append(sub_header("Pearson Correlation Matrix"))
    lines.append(correlation_matrix.round(3).to_string())

    high_correlation_pairs = find_high_correlation_pairs(
        correlation_matrix, HIGH_CORRELATION_THRESHOLD
    )
    lines.append(sub_header(f"Highlighted Correlations (|r| >= {HIGH_CORRELATION_THRESHOLD})"))
    if high_correlation_pairs:
        for col_a, col_b, value in high_correlation_pairs:
            lines.append(f"    {col_a} <-> {col_b} : r = {value:.3f}")
    else:
        lines.append("None found.")

    return "\n".join(lines), correlation_matrix, high_correlation_pairs


def find_high_correlation_pairs(
    correlation_matrix: pd.DataFrame, threshold: float
) -> List[Tuple[str, str, float]]:
    """Find column pairs whose absolute Pearson correlation exceeds a threshold.

    Args:
        correlation_matrix: Pearson correlation matrix.
        threshold: Minimum absolute correlation to flag.

    Returns:
        List of (col_a, col_b, correlation) tuples sorted by descending
        absolute correlation.
    """
    pairs: List[Tuple[str, str, float]] = []
    columns = list(correlation_matrix.columns)
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1:]:
            value = correlation_matrix.loc[col_a, col_b]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append((col_a, col_b, float(value)))
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    return pairs


# --------------------------------------------------------------------------
# CATEGORICAL ANALYSIS
# --------------------------------------------------------------------------

def build_categorical_section(dataframe: pd.DataFrame) -> List[str]:
    """Identify categorical columns for downstream analysis.

    Args:
        dataframe: Dataset to analyze.

    Returns:
        List of categorical column names.
    """
    categorical_columns = [
        col for col in dataframe.columns if is_categorical_column(dataframe[col])
    ]
    return categorical_columns


def render_categorical_section_text(
    dataframe: pd.DataFrame, categorical_columns: List[str]
) -> str:
    """Render the CATEGORICAL ANALYSIS section text.

    Args:
        dataframe: Dataset to analyze.
        categorical_columns: List of categorical column names.

    Returns:
        Formatted section text.
    """
    lines = [section_header("CATEGORICAL ANALYSIS")]

    if not categorical_columns:
        lines.append("No categorical columns found.")
        return "\n".join(lines)

    for column in categorical_columns:
        series = dataframe[column]
        cardinality = series.nunique(dropna=True)
        lines.append(sub_header(f"Column: {column}"))
        lines.append(f"Cardinality        : {cardinality:,}")

        top_values = series.value_counts(dropna=True).head(TOP_N_CATEGORICAL_VALUES)
        if top_values.empty:
            lines.append("No non-missing values to summarize.")
        else:
            lines.append(f"Top {min(TOP_N_CATEGORICAL_VALUES, len(top_values))} values:")
            for value, frequency in top_values.items():
                lines.append(f"    {str(value):<40}{frequency:>10,}")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# DATA QUALITY CHECKS
# --------------------------------------------------------------------------

def find_duplicate_columns(dataframe: pd.DataFrame) -> List[Tuple[str, str]]:
    """Identify columns that are exact duplicates of an earlier column.

    Args:
        dataframe: Dataset to analyze.

    Returns:
        List of (original_column, duplicate_column) pairs.
    """
    duplicates: List[Tuple[str, str]] = []
    columns = list(dataframe.columns)
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1:]:
            try:
                if dataframe[col_a].equals(dataframe[col_b]):
                    duplicates.append((col_a, col_b))
            except Exception:
                continue
    return duplicates


def find_constant_columns(dataframe: pd.DataFrame) -> List[str]:
    """Identify columns with zero or one distinct non-missing value.

    Args:
        dataframe: Dataset to analyze.

    Returns:
        List of constant (zero-variance) column names.
    """
    constant_columns: List[str] = []
    for column in dataframe.columns:
        try:
            if dataframe[column].nunique(dropna=True) <= 1:
                constant_columns.append(column)
        except TypeError:
            continue
    return constant_columns


def find_fully_empty_columns(dataframe: pd.DataFrame) -> List[str]:
    """Identify columns that are entirely missing.

    Args:
        dataframe: Dataset to analyze.

    Returns:
        List of fully-empty column names.
    """
    return [column for column in dataframe.columns if dataframe[column].isna().all()]


def find_high_missing_columns(missing_pct: pd.Series, threshold: float) -> List[str]:
    """Identify columns whose missing percentage exceeds a threshold.

    Args:
        missing_pct: Per-column missing percentage.
        threshold: Percentage threshold (0-100).

    Returns:
        List of column names exceeding the threshold.
    """
    return missing_pct[missing_pct > threshold].index.tolist()


def find_potential_id_columns(dataframe: pd.DataFrame) -> List[str]:
    """Identify columns that look like identifier columns.

    Heuristic: near-unique values (ratio above threshold) relative to row
    count AND an integer/object/category/string dtype (continuous floats
    are excluded, since high cardinality there usually reflects a
    measurement rather than an identifier), or a column name containing
    "id" with a moderate-to-high uniqueness ratio.

    Args:
        dataframe: Dataset to analyze.

    Returns:
        List of column names flagged as potential IDs.
    """
    id_columns: List[str] = []
    row_count = len(dataframe)
    if row_count == 0:
        return id_columns

    for column in dataframe.columns:
        series = dataframe[column]
        try:
            unique_ratio = series.nunique(dropna=True) / row_count
        except TypeError:
            continue

        name_hints_id = "id" in column.lower()
        is_id_like_dtype = (
            pd.api.types.is_integer_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(series)
        )
        looks_unique = unique_ratio >= ID_UNIQUE_RATIO_THRESHOLD and is_id_like_dtype
        if looks_unique or (name_hints_id and unique_ratio > 0.5):
            id_columns.append(column)

    return id_columns


def find_high_cardinality_columns(
    dataframe: pd.DataFrame, categorical_columns: List[str]
) -> List[str]:
    """Identify categorical columns with unusually high cardinality.

    Args:
        dataframe: Dataset to analyze.
        categorical_columns: List of categorical column names.

    Returns:
        List of high-cardinality categorical column names.
    """
    row_count = len(dataframe)
    if row_count == 0:
        return []

    high_cardinality: List[str] = []
    for column in categorical_columns:
        unique_count = dataframe[column].nunique(dropna=True)
        unique_ratio = unique_count / row_count
        if unique_count >= HIGH_CARDINALITY_MIN_UNIQUE and unique_ratio >= HIGH_CARDINALITY_UNIQUE_RATIO:
            high_cardinality.append(column)
    return high_cardinality


def build_data_quality_section(
    dataframe: pd.DataFrame, missing_pct: pd.Series, categorical_columns: List[str]
) -> Tuple[str, Dict[str, Any]]:
    """Build the DATA QUALITY CHECKS section.

    Args:
        dataframe: Dataset to analyze.
        missing_pct: Per-column missing percentage.
        categorical_columns: List of categorical column names.

    Returns:
        Tuple of (formatted section text, dict of quality findings for
        reuse in later sections). Findings keys: duplicate_columns,
        constant_columns, fully_empty_columns, high_missing_columns,
        id_columns, high_cardinality_columns.
    """
    findings: Dict[str, Any] = {
        "duplicate_columns": find_duplicate_columns(dataframe),
        "constant_columns": find_constant_columns(dataframe),
        "fully_empty_columns": find_fully_empty_columns(dataframe),
        "high_missing_columns": find_high_missing_columns(
            missing_pct, HIGH_MISSING_THRESHOLD_PCT
        ),
        "id_columns": find_potential_id_columns(dataframe),
        "high_cardinality_columns": find_high_cardinality_columns(
            dataframe, categorical_columns
        ),
    }

    lines = [section_header("DATA QUALITY CHECKS")]

    lines.append(sub_header("Duplicate Columns (identical values)"))
    if findings["duplicate_columns"]:
        for original, duplicate in findings["duplicate_columns"]:
            lines.append(f"    {duplicate}  (duplicate of {original})")
    else:
        lines.append("None found.")

    lines.append(sub_header("Constant Columns"))
    lines.append(
        ", ".join(findings["constant_columns"]) if findings["constant_columns"] else "None found."
    )

    lines.append(sub_header("Fully Empty Columns"))
    lines.append(
        ", ".join(findings["fully_empty_columns"])
        if findings["fully_empty_columns"]
        else "None found."
    )

    lines.append(sub_header(f"Columns With >{HIGH_MISSING_THRESHOLD_PCT:.0f}% Missing"))
    lines.append(
        ", ".join(findings["high_missing_columns"])
        if findings["high_missing_columns"]
        else "None found."
    )

    lines.append(sub_header("Possible ID Columns"))
    lines.append(", ".join(findings["id_columns"]) if findings["id_columns"] else "None found.")

    lines.append(sub_header("High-Cardinality Columns"))
    lines.append(
        ", ".join(findings["high_cardinality_columns"])
        if findings["high_cardinality_columns"]
        else "None found."
    )

    return "\n".join(lines), findings


# --------------------------------------------------------------------------
# ML READINESS
# --------------------------------------------------------------------------

def classify_columns_for_ml(
    dataframe: pd.DataFrame, quality_findings: Dict[str, Any]
) -> Dict[str, List[str]]:
    """Classify every column into ML-readiness buckets.

    Buckets are: likely feature columns, possible target columns, likely
    identifier columns, and columns recommended to drop. Classification is
    purely descriptive (name/statistics based); no columns are altered.

    Args:
        dataframe: Dataset to analyze.
        quality_findings: Findings dict produced by build_data_quality_section.

    Returns:
        Dict with keys "features", "targets", "identifiers", "drop", each
        mapping to a list of column names.
    """
    drop_candidates = set(quality_findings["constant_columns"])
    drop_candidates |= set(quality_findings["fully_empty_columns"])
    drop_candidates |= set(quality_findings["high_missing_columns"])
    drop_candidates |= {duplicate for _, duplicate in quality_findings["duplicate_columns"]}

    identifier_columns = set(quality_findings["id_columns"])

    target_columns = [
        column
        for column in dataframe.columns
        if column not in identifier_columns
        and any(hint == column.lower() or hint in column.lower().split("_") for hint in TARGET_NAME_HINTS)
    ]

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in identifier_columns
        and column not in drop_candidates
        and column not in target_columns
    ]

    return {
        "features": feature_columns,
        "targets": target_columns,
        "identifiers": sorted(identifier_columns),
        "drop": sorted(drop_candidates),
    }


def build_ml_readiness_section(ml_classification: Dict[str, List[str]]) -> str:
    """Build the ML READINESS section.

    Args:
        ml_classification: Output of classify_columns_for_ml.

    Returns:
        Formatted section text.
    """
    lines = [section_header("ML READINESS")]

    lines.append(sub_header("Likely Feature Columns"))
    lines.append(
        ", ".join(ml_classification["features"]) if ml_classification["features"] else "None."
    )

    lines.append(sub_header("Possible Target Columns"))
    lines.append(
        ", ".join(ml_classification["targets"]) if ml_classification["targets"] else "None detected."
    )

    lines.append(sub_header("Likely Identifier Columns"))
    lines.append(
        ", ".join(ml_classification["identifiers"])
        if ml_classification["identifiers"]
        else "None detected."
    )

    lines.append(sub_header("Columns Recommended To Drop"))
    lines.append(", ".join(ml_classification["drop"]) if ml_classification["drop"] else "None.")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# OBSERVATIONS
# --------------------------------------------------------------------------

def build_observations_section(
    dataframe: pd.DataFrame,
    missing_pct: pd.Series,
    high_correlation_pairs: List[Tuple[str, str, float]],
    outlier_counts: Dict[str, int],
    quality_findings: Dict[str, Any],
    ml_classification: Dict[str, List[str]],
) -> List[str]:
    """Generate rule-based observations about the dataset.

    Args:
        dataframe: Dataset to analyze.
        missing_pct: Per-column missing percentage.
        high_correlation_pairs: Output of find_high_correlation_pairs.
        outlier_counts: Per-column IQR outlier counts.
        quality_findings: Findings dict produced by build_data_quality_section.
        ml_classification: Output of classify_columns_for_ml.

    Returns:
        List of observation sentences.
    """
    observations: List[str] = []

    notable_missing = missing_pct[missing_pct > 0].sort_values(ascending=False)
    if notable_missing.empty:
        observations.append("No missing data detected across any column.")
    else:
        worst_column = notable_missing.index[0]
        observations.append(
            f"Column '{worst_column}' has the highest missing rate at "
            f"{notable_missing.iloc[0]:.1f}%; review whether imputation or "
            "removal is appropriate."
        )
        if quality_findings["high_missing_columns"]:
            observations.append(
                f"{len(quality_findings['high_missing_columns'])} column(s) exceed "
                f"{HIGH_MISSING_THRESHOLD_PCT:.0f}% missing "
                f"({', '.join(quality_findings['high_missing_columns'])}); these are "
                "weak candidates for modeling without heavy imputation."
            )

    if high_correlation_pairs:
        pair_strings = [
            f"{col_a} & {col_b} (r={value:.2f})" for col_a, col_b, value in high_correlation_pairs[:5]
        ]
        observations.append(
            f"Highly correlated numeric pairs detected (|r| >= {HIGH_CORRELATION_THRESHOLD}): "
            + "; ".join(pair_strings)
            + ". Consider dropping redundant features, and check whether either "
            "column could leak information about a target."
        )

    noisy_columns = {column: count for column, count in outlier_counts.items() if count > 0}
    if noisy_columns:
        worst_outlier_column = max(noisy_columns, key=noisy_columns.get)
        observations.append(
            f"Column '{worst_outlier_column}' has the most IQR-flagged outliers "
            f"({noisy_columns[worst_outlier_column]:,}); inspect for data entry "
            "errors versus genuine extreme values."
        )

    if quality_findings["constant_columns"]:
        observations.append(
            "Constant column(s) found "
            f"({', '.join(quality_findings['constant_columns'])}); these carry no "
            "predictive signal and should be dropped."
        )

    if quality_findings["fully_empty_columns"]:
        observations.append(
            "Fully empty column(s) found "
            f"({', '.join(quality_findings['fully_empty_columns'])}); these should "
            "be dropped before modeling."
        )

    if quality_findings["duplicate_columns"]:
        duplicate_strings = [
            f"{duplicate} == {original}" for original, duplicate in quality_findings["duplicate_columns"]
        ]
        observations.append(
            "Duplicate column(s) detected (" + "; ".join(duplicate_strings) + "); keep "
            "only one copy of each to avoid redundant features."
        )

    if ml_classification["targets"]:
        observations.append(
            "Possible target column(s) detected by name "
            f"({', '.join(ml_classification['targets'])}); confirm these are not "
            "accidentally included as predictive features, which would cause "
            "data leakage."
        )

    if ml_classification["identifiers"]:
        observations.append(
            "Potential identifier column(s) detected "
            f"({', '.join(ml_classification['identifiers'])}); these should be "
            "excluded from the feature set since they carry no generalizable "
            "signal and can cause the model to memorize row identity."
        )

    if quality_findings["high_cardinality_columns"]:
        observations.append(
            "High-cardinality categorical column(s) detected "
            f"({', '.join(quality_findings['high_cardinality_columns'])}); one-hot "
            "encoding these directly would create excessive dimensionality."
        )

    if ml_classification["features"]:
        observations.append(
            f"{len(ml_classification['features'])} column(s) currently look like "
            "usable ML features based on completeness, variance, and naming: "
            f"{', '.join(ml_classification['features'])}."
        )
    else:
        observations.append(
            "No columns clearly stood out as strong ML feature candidates; "
            "manual review is recommended."
        )

    return observations


def render_observations_section(observations: List[str]) -> str:
    """Render the OBSERVATIONS section text.

    Args:
        observations: List of observation sentences.

    Returns:
        Formatted section text.
    """
    lines = [section_header("OBSERVATIONS")]
    for index, observation in enumerate(observations, start=1):
        lines.append(f"{index}. {observation}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# RECOMMENDATIONS
# --------------------------------------------------------------------------

def build_recommendations_section(
    dataframe: pd.DataFrame,
    quality_findings: Dict[str, Any],
    ml_classification: Dict[str, List[str]],
    categorical_columns: List[str],
    numeric_columns: List[str],
) -> str:
    """Build the RECOMMENDATIONS section: cleaning, encoding, and scaling guidance.

    Args:
        dataframe: Dataset to analyze.
        quality_findings: Findings dict produced by build_data_quality_section.
        ml_classification: Output of classify_columns_for_ml.
        categorical_columns: List of categorical column names.
        numeric_columns: List of numeric column names.

    Returns:
        Formatted section text.
    """
    lines = [section_header("RECOMMENDATIONS")]

    lines.append(sub_header("Cleaning Actions"))
    cleaning_actions: List[str] = []
    if quality_findings["fully_empty_columns"]:
        cleaning_actions.append(
            f"Drop fully empty column(s): {', '.join(quality_findings['fully_empty_columns'])}."
        )
    if quality_findings["constant_columns"]:
        cleaning_actions.append(
            f"Drop constant column(s): {', '.join(quality_findings['constant_columns'])}."
        )
    if quality_findings["duplicate_columns"]:
        duplicate_names = [duplicate for _, duplicate in quality_findings["duplicate_columns"]]
        cleaning_actions.append(f"Drop duplicate column(s): {', '.join(duplicate_names)}.")
    if quality_findings["high_missing_columns"]:
        cleaning_actions.append(
            "Drop or heavily investigate high-missing column(s): "
            f"{', '.join(quality_findings['high_missing_columns'])}."
        )
    moderate_missing_numeric = [
        column
        for column in numeric_columns
        if column not in quality_findings["high_missing_columns"]
        and dataframe[column].isna().any()
    ]
    moderate_missing_categorical = [
        column
        for column in categorical_columns
        if column not in quality_findings["high_missing_columns"]
        and dataframe[column].isna().any()
    ]
    if moderate_missing_numeric:
        cleaning_actions.append(
            "Impute remaining missing numeric column(s) using the median: "
            f"{', '.join(moderate_missing_numeric)}."
        )
    if moderate_missing_categorical:
        cleaning_actions.append(
            "Impute remaining missing categorical column(s) using the mode "
            f"(most frequent value): {', '.join(moderate_missing_categorical)}."
        )
    if not cleaning_actions:
        cleaning_actions.append("No major cleaning actions required.")
    lines.extend(cleaning_actions)

    lines.append(sub_header("Encoding Strategy"))
    ignore_columns_preview = set(ml_classification["drop"]) | set(ml_classification["identifiers"])
    low_cardinality_categorical = [
        column
        for column in categorical_columns
        if column not in quality_findings["high_cardinality_columns"]
        and column not in ignore_columns_preview
    ]
    high_cardinality_non_ignored = [
        column
        for column in quality_findings["high_cardinality_columns"]
        if column not in ignore_columns_preview
    ]
    if low_cardinality_categorical:
        lines.append(
            "One-hot or ordinal encode low-cardinality categorical column(s): "
            f"{', '.join(low_cardinality_categorical)}."
        )
    if high_cardinality_non_ignored:
        lines.append(
            "Use target/frequency encoding or hashing for high-cardinality "
            f"column(s): {', '.join(high_cardinality_non_ignored)} to avoid "
            "excessive dimensionality from one-hot encoding."
        )
    if not low_cardinality_categorical and not high_cardinality_non_ignored:
        lines.append("No categorical columns require encoding beyond those already dropped/ignored.")

    lines.append(sub_header("Scaling Strategy"))
    scalable_numeric_columns = [
        column for column in numeric_columns if column not in ignore_columns_preview
    ]
    if scalable_numeric_columns:
        lines.append(
            "Standardize or min-max scale numeric column(s): "
            f"{', '.join(scalable_numeric_columns)}, particularly for "
            "distance-based or gradient-based models; tree-based models do "
            "not require scaling."
        )
    else:
        lines.append("No numeric columns require scaling beyond those already dropped/ignored.")

    lines.append(sub_header("Columns To Ignore"))
    ignore_columns = sorted(set(ml_classification["drop"]) | set(ml_classification["identifiers"]))
    lines.append(", ".join(ignore_columns) if ignore_columns else "None.")

    lines.append(sub_header("Columns Worth Keeping"))
    lines.append(
        ", ".join(ml_classification["features"]) if ml_classification["features"] else "None."
    )

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def generate_eda_report(csv_path: Path, output_dir: Path) -> Path:
    """Generate a full EDA text report for a single dataset.

    Args:
        csv_path: Path to the source CSV file.
        output_dir: Directory in which to write the report.

    Returns:
        Path to the written report file.

    Raises:
        Exception: Propagates any error encountered while reading or
        analyzing the dataset.
    """
    dataframe = load_dataset(csv_path)

    overview_text = build_overview_section(dataframe, csv_path)
    column_info_text, missing_pct = build_column_information_section(dataframe)
    numeric_text, numeric_columns, outlier_counts = build_numeric_section(dataframe)
    correlation_text, _correlation_matrix, high_correlation_pairs = build_correlation_section(
        dataframe, numeric_columns
    )
    categorical_columns = build_categorical_section(dataframe)
    categorical_text = render_categorical_section_text(dataframe, categorical_columns)
    quality_text, quality_findings = build_data_quality_section(
        dataframe, missing_pct, categorical_columns
    )
    ml_classification = classify_columns_for_ml(dataframe, quality_findings)
    ml_readiness_text = build_ml_readiness_section(ml_classification)
    observations = build_observations_section(
        dataframe,
        missing_pct,
        high_correlation_pairs,
        outlier_counts,
        quality_findings,
        ml_classification,
    )
    observations_text = render_observations_section(observations)
    recommendations_text = build_recommendations_section(
        dataframe, quality_findings, ml_classification, categorical_columns, numeric_columns
    )

    report_text = "\n".join(
        [
            f"EDA REPORT: {csv_path.name}",
            overview_text,
            column_info_text,
            numeric_text,
            correlation_text,
            categorical_text,
            quality_text,
            ml_readiness_text,
            observations_text,
            recommendations_text,
            "",
        ]
    )

    dataset_name = csv_path.stem
    report_path = output_dir / f"{dataset_name}_eda.txt"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def run_eda(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> None:
    """Run EDA over every CSV file in the input directory.

    Args:
        input_dir: Directory containing processed CSV files.
        output_dir: Directory in which to write EDA reports.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = discover_csv_files(input_dir)
    if not csv_files:
        logger.warning("No CSV files found in '%s'. Nothing to do.", input_dir)
        return

    logger.info("Found %d CSV file(s) in '%s'.", len(csv_files), input_dir)

    processed: List[str] = []
    failed: List[Tuple[str, str]] = []

    for csv_path in csv_files:
        logger.info("Processing '%s'...", csv_path.name)
        try:
            report_path = generate_eda_report(csv_path, output_dir)
            logger.info("Completed '%s' -> %s", csv_path.name, report_path.name)
            processed.append(csv_path.name)
        except Exception as exc:  # noqa: BLE001 - report and continue
            logger.error("Failed to process '%s': %s", csv_path.name, exc, exc_info=True)
            failed.append((csv_path.name, str(exc)))

    print_final_summary(processed, failed, output_dir)


def print_final_summary(
    processed: List[str], failed: List[Tuple[str, str]], output_dir: Path
) -> None:
    """Print the final run summary.

    Args:
        processed: Names of successfully processed dataset files.
        failed: List of (filename, error_message) for failed datasets.
        output_dir: Directory where reports were written.
    """
    print(section_header("EDA SUMMARY"))
    print(f"Processed datasets ({len(processed)}):")
    for name in processed:
        print(f"    - {name}")

    print(f"\nFailed datasets ({len(failed)}):")
    for name, error in failed:
        print(f"    - {name}: {error}")

    print(f"\nReports written to: {output_dir.resolve()}")


def main() -> None:
    """Entry point for the HIRESCORE EDA script."""
    configure_logging()
    run_eda()


if __name__ == "__main__":
    main()