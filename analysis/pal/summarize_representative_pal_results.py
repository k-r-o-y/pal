"""
Summarise representative PAL benchmark results.

This module reads the row-level CSV produced by:

    analysis/pal/run_representative_pal_benchmark.py

It validates the result schema, computes aggregate statistics, identifies
best- and worst-case configurations, compares polynomial bases, and writes:

1. A machine-readable JSON summary.
2. A CSV table containing aggregate metrics by basis and dtype.
3. A CSV table containing aggregate metrics by scenario, basis, and dtype.
4. A CSV table containing aggregate metrics by dimension, degree, basis,
   and dtype.
5. A CSV table containing basis rankings.
6. Optional LaTeX tables suitable for inclusion in the dissertation.

The summariser is deliberately defensive. It supports small differences in
column naming between benchmark-runner versions and skips unavailable metrics
rather than silently inventing values.

Example
-------

python -m analysis.pal.summarize_representative_pal_results

python -m analysis.pal.summarize_representative_pal_results \
    --results-path results/pal/representative_pal_results.csv \
    --summary-path results/pal/representative_pal_summary.json \
    --tables-dir results/pal/tables \
    --latex-dir results/pal/latex
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_RESULTS_PATH = Path(
    "results/pal/representative_pal_results.csv"
)
DEFAULT_SUMMARY_PATH = Path(
    "results/pal/representative_pal_summary.json"
)
DEFAULT_TABLES_DIR = Path("results/pal/tables")
DEFAULT_LATEX_DIR = Path("results/pal/latex")

DEFAULT_BASE_ORDER = ("monomial", "legendre", "chebyshev")
DEFAULT_DTYPE_ORDER = ("float32", "float64")

IDENTIFIER_COLUMNS = (
    "scenario",
    "constraint_name",
    "constraint_type",
    "query_name",
    "schedule",
    "trajectory_step",
    "dimension",
    "source_polynomial_degree",
    "density_polynomial_degree",
    "trial",
    "basis",
    "dtype",
    "seed",
)

LOWER_IS_BETTER_METRICS = (
    "partition_function_relative_error",
    "partition_relative_error",
    "normalising_constant_relative_error",
    "normalization_constant_relative_error",
    "query_probability_absolute_error",
    "query_absolute_error",
    "query_probability_relative_error",
    "query_relative_error",
    "relative_integration_error",
    "integration_relative_error",
    "coefficient_perturbation_sensitivity",
    "perturbation_sensitivity",
    "query_probability_perturbation_sensitivity",
    "query_perturbation_sensitivity",
    "coefficient_recovery_relative_error",
    "recovered_integral_relative_error",
    "basis_conversion_residual",
    "conversion_residual",
    "total_evaluation_ms",
    "runtime_ms",
    "elapsed_ms",
)

HIGHER_IS_BETTER_METRICS = (
    "numerical_rank_fraction",
    "rank_fraction",
)

DIAGNOSTIC_METRICS = (
    "basis_condition_number",
    "condition_number",
    "coefficient_noise_amplification",
    "function_value_noise_amplification",
)

REFERENCE_COLUMNS = (
    "reference_partition_function",
    "reference_normalising_constant",
    "reference_normalization_constant",
    "reference_integral",
    "reference_query_probability",
)

COMPUTED_COLUMNS = (
    "computed_partition_function",
    "computed_normalising_constant",
    "computed_normalization_constant",
    "computed_integral",
    "computed_query_probability",
)

BOOLEAN_COLUMNS = (
    "is_finite",
    "sampled_negative_density",
    "has_negative_density",
    "perturbation_rounded_away",
    "conversion_warning",
    "pal_success",
    "success",
)


@dataclass(frozen=True)
class MetricDescriptor:
    """Description of one numeric benchmark metric."""

    name: str
    direction: str
    category: str

    @property
    def lower_is_better(self) -> bool:
        return self.direction == "lower"

    @property
    def higher_is_better(self) -> bool:
        return self.direction == "higher"


@dataclass(frozen=True)
class SummaryConfig:
    """Configuration recorded in the generated JSON summary."""

    results_path: str
    summary_path: str
    tables_dir: str
    latex_dir: str | None
    confidence_level: float
    bootstrap_samples: int
    random_seed: int
    write_latex: bool


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise the row-level output from the representative PAL "
            "benchmark."
        )
    )

    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=(
            "Path to representative_pal_results.csv. "
            f"Default: {DEFAULT_RESULTS_PATH}"
        ),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=(
            "Destination for the JSON summary. "
            f"Default: {DEFAULT_SUMMARY_PATH}"
        ),
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
        help=(
            "Directory for generated CSV summary tables. "
            f"Default: {DEFAULT_TABLES_DIR}"
        ),
    )
    parser.add_argument(
        "--latex-dir",
        type=Path,
        default=DEFAULT_LATEX_DIR,
        help=(
            "Directory for generated LaTeX tables. "
            f"Default: {DEFAULT_LATEX_DIR}"
        ),
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Do not generate LaTeX table files.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Bootstrap confidence level. Default: 0.95.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help=(
            "Number of bootstrap samples used for median confidence "
            "intervals. Use 0 to disable. Default: 2000."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260806,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail when core identifier columns are missing instead of "
            "continuing with the columns that are available."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console summary output.",
    )

    args = parser.parse_args(argv)

    if not 0.0 < args.confidence_level < 1.0:
        parser.error("--confidence-level must lie strictly between 0 and 1.")

    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be non-negative.")

    return args


def normalise_column_name(name: str) -> str:
    """Convert a source column name into a stable snake-case form."""
    text = str(name).strip().lower()

    replacements = {
        " ": "_",
        "-": "_",
        "/": "_",
        "%": "percent",
        "(": "",
        ")": "",
        "[": "",
        "]": "",
        ".": "_",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    while "__" in text:
        text = text.replace("__", "_")

    return text.strip("_")


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalised and unique column names."""
    result = frame.copy()

    normalised: list[str] = []
    counts: dict[str, int] = {}

    for original in result.columns:
        base = normalise_column_name(str(original))
        if not base:
            base = "unnamed"

        count = counts.get(base, 0)
        counts[base] = count + 1

        if count == 0:
            normalised.append(base)
        else:
            normalised.append(f"{base}_{count + 1}")

    result.columns = normalised
    return result


def canonicalise_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add canonical aliases for commonly renamed benchmark columns.

    Existing columns are never overwritten.
    """
    result = frame.copy()

    aliases: Mapping[str, Sequence[str]] = {
        "scenario": (
            "pal_scenario",
            "benchmark_scenario",
            "constraint_scenario",
        ),
        "basis": (
            "basis_name",
            "polynomial_basis",
            "representation",
        ),
        "dtype": (
            "precision",
            "arithmetic",
            "floating_point_dtype",
        ),
        "dimension": (
            "n_dimension",
            "num_dimensions",
            "n",
        ),
        "source_polynomial_degree": (
            "source_degree",
            "q_degree",
        ),
        "density_polynomial_degree": (
            "density_degree",
            "polynomial_degree",
            "degree",
        ),
        "trial": (
            "trial_index",
            "repeat",
            "repetition",
        ),
        "total_evaluation_ms": (
            "runtime_ms",
            "elapsed_ms",
            "total_runtime_ms",
            "evaluation_time_ms",
        ),
        "basis_condition_number": (
            "condition_number",
            "basis_matrix_condition_number",
            "matrix_condition_number",
        ),
        "numerical_rank_fraction": (
            "rank_fraction",
            "retained_rank_fraction",
        ),
        "partition_function_relative_error": (
            "partition_relative_error",
            "normalising_constant_relative_error",
            "normalization_constant_relative_error",
            "partition_error",
        ),
        "query_probability_absolute_error": (
            "query_absolute_error",
            "query_abs_error",
            "absolute_query_error",
        ),
        "query_probability_relative_error": (
            "query_relative_error",
            "relative_query_error",
        ),
        "coefficient_perturbation_sensitivity": (
            "perturbation_sensitivity",
            "coefficient_sensitivity",
        ),
        "query_probability_perturbation_sensitivity": (
            "query_perturbation_sensitivity",
            "query_sensitivity",
        ),
        "basis_conversion_residual": (
            "conversion_residual",
            "basis_residual",
        ),
        "coefficient_noise_amplification": (
            "function_value_noise_amplification",
            "noise_amplification",
        ),
        "relative_integration_error": (
            "integration_relative_error",
            "integral_relative_error",
        ),
        "pal_success": (
            "success",
            "completed_successfully",
        ),
    }

    for canonical, candidates in aliases.items():
        if canonical in result.columns:
            continue

        for candidate in candidates:
            if candidate in result.columns:
                result[canonical] = result[candidate]
                break

    return result


def coerce_boolean_series(series: pd.Series) -> pd.Series:
    """Convert common textual and numeric Boolean encodings."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    lowered = series.astype(str).str.strip().str.lower()

    mapping = {
        "true": True,
        "t": True,
        "yes": True,
        "y": True,
        "1": True,
        "1.0": True,
        "false": False,
        "f": False,
        "no": False,
        "n": False,
        "0": False,
        "0.0": False,
        "nan": pd.NA,
        "none": pd.NA,
        "": pd.NA,
    }

    mapped = lowered.map(mapping)
    return mapped.astype("boolean")


def infer_numeric_columns(frame: pd.DataFrame) -> set[str]:
    """Infer columns that should be converted to numeric values."""
    numeric_keywords = (
        "error",
        "sensitivity",
        "condition",
        "rank",
        "runtime",
        "elapsed",
        "_ms",
        "amplification",
        "residual",
        "integral",
        "probability",
        "partition",
        "normalising",
        "normalization",
        "coefficient",
        "density_min",
        "density_max",
        "degree",
        "dimension",
        "trial",
        "step",
        "seed",
        "count",
    )

    inferred: set[str] = set()

    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            inferred.add(column)
            continue

        if any(keyword in column for keyword in numeric_keywords):
            inferred.add(column)

    return inferred


def coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce likely numeric and Boolean fields while preserving identifiers."""
    result = frame.copy()

    for column in BOOLEAN_COLUMNS:
        if column in result.columns:
            result[column] = coerce_boolean_series(result[column])

    numeric_columns = infer_numeric_columns(result)

    protected_text_columns = {
        "basis",
        "dtype",
        "scenario",
        "constraint_name",
        "constraint_type",
        "query_name",
        "schedule",
        "reference_method",
        "status",
        "error_message",
    }

    for column in sorted(numeric_columns):
        if column in protected_text_columns:
            continue

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    for column in (
        "basis",
        "dtype",
        "scenario",
        "schedule",
        "constraint_name",
        "constraint_type",
        "query_name",
    ):
        if column in result.columns:
            result[column] = (
                result[column]
                .astype("string")
                .str.strip()
                .str.lower()
            )

    return result


def load_results(path: Path) -> pd.DataFrame:
    """Load and normalise the benchmark result CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Representative PAL result file was not found: {path}"
        )

    if not path.is_file():
        raise ValueError(f"Results path is not a regular file: {path}")

    frame = pd.read_csv(path)

    if frame.empty:
        raise ValueError(f"Result file contains no rows: {path}")

    frame = normalise_columns(frame)
    frame = canonicalise_aliases(frame)
    frame = coerce_types(frame)

    return frame


def validate_schema(
    frame: pd.DataFrame,
    *,
    strict: bool,
) -> list[str]:
    """Validate important identifiers and return warning messages."""
    warnings: list[str] = []

    required = ("basis", "dtype")
    missing_required = [
        column for column in required if column not in frame.columns
    ]

    if missing_required:
        message = (
            "Missing required identifier column(s): "
            + ", ".join(missing_required)
        )
        if strict:
            raise ValueError(message)
        warnings.append(message)

    recommended = (
        "scenario",
        "dimension",
        "density_polynomial_degree",
        "trial",
    )

    missing_recommended = [
        column
        for column in recommended
        if column not in frame.columns
    ]

    if missing_recommended:
        warnings.append(
            "Missing recommended column(s): "
            + ", ".join(missing_recommended)
        )

    metric_columns = discover_metric_columns(frame)

    if not metric_columns:
        raise ValueError(
            "No recognised numeric benchmark metrics were found."
        )

    duplicated_rows = int(frame.duplicated().sum())
    if duplicated_rows:
        warnings.append(
            f"The input contains {duplicated_rows} exactly duplicated row(s)."
        )

    return warnings


def metric_descriptor(column: str) -> MetricDescriptor:
    """Describe the optimisation direction and category of a metric."""
    if column in HIGHER_IS_BETTER_METRICS:
        return MetricDescriptor(
            name=column,
            direction="higher",
            category="rank",
        )

    if column in LOWER_IS_BETTER_METRICS:
        return MetricDescriptor(
            name=column,
            direction="lower",
            category=classify_metric(column),
        )

    if column in DIAGNOSTIC_METRICS:
        return MetricDescriptor(
            name=column,
            direction="lower",
            category="diagnostic",
        )

    if "rank_fraction" in column:
        return MetricDescriptor(
            name=column,
            direction="higher",
            category="rank",
        )

    return MetricDescriptor(
        name=column,
        direction="lower",
        category=classify_metric(column),
    )


def classify_metric(column: str) -> str:
    """Assign a broad category to a metric name."""
    if "condition" in column:
        return "conditioning"
    if "rank" in column:
        return "rank"
    if "runtime" in column or column.endswith("_ms"):
        return "runtime"
    if "query" in column:
        return "query"
    if "partition" in column or "normalis" in column:
        return "partition"
    if "perturb" in column or "sensitivity" in column:
        return "perturbation"
    if "amplification" in column:
        return "amplification"
    if "conversion" in column or "residual" in column:
        return "conversion"
    if "recover" in column:
        return "recovery"
    if "integration" in column or "integral" in column:
        return "integration"
    return "other"


def discover_metric_columns(frame: pd.DataFrame) -> list[str]:
    """Return recognised numeric benchmark metrics present in the data."""
    excluded = set(IDENTIFIER_COLUMNS)
    excluded.update(REFERENCE_COLUMNS)
    excluded.update(COMPUTED_COLUMNS)

    candidates: list[str] = []

    metric_keywords = (
        "error",
        "sensitivity",
        "condition",
        "rank_fraction",
        "runtime",
        "_ms",
        "amplification",
        "residual",
    )

    for column in frame.columns:
        if column in excluded:
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        if any(keyword in column for keyword in metric_keywords):
            candidates.append(column)

    priority = [
        "partition_function_relative_error",
        "query_probability_absolute_error",
        "query_probability_relative_error",
        "relative_integration_error",
        "basis_condition_number",
        "numerical_rank_fraction",
        "coefficient_perturbation_sensitivity",
        "query_probability_perturbation_sensitivity",
        "coefficient_noise_amplification",
        "coefficient_recovery_relative_error",
        "recovered_integral_relative_error",
        "basis_conversion_residual",
        "total_evaluation_ms",
    ]

    order = {name: index for index, name in enumerate(priority)}

    return sorted(
        set(candidates),
        key=lambda name: (order.get(name, len(order)), name),
    )


def finite_numeric_values(series: pd.Series) -> np.ndarray:
    """Return finite floating-point values from a pandas Series."""
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    return numeric[np.isfinite(numeric)]


def safe_float(value: Any) -> float | None:
    """Convert a scalar to a JSON-safe finite float."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric):
        return None

    return numeric


def bootstrap_median_interval(
    values: np.ndarray,
    *,
    confidence_level: float,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    """Bootstrap a percentile confidence interval for the median."""
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]

    if clean.size == 0 or samples <= 0:
        return None, None

    if clean.size == 1:
        value = float(clean[0])
        return value, value

    indices = rng.integers(
        low=0,
        high=clean.size,
        size=(samples, clean.size),
        endpoint=False,
    )
    resampled = clean[indices]
    medians = np.median(resampled, axis=1)

    alpha = 1.0 - confidence_level
    lower = float(np.quantile(medians, alpha / 2.0))
    upper = float(np.quantile(medians, 1.0 - alpha / 2.0))

    return lower, upper


def descriptive_statistics(
    series: pd.Series,
    *,
    confidence_level: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Compute robust descriptive statistics for one metric."""
    values = finite_numeric_values(series)

    count_total = int(len(series))
    count_finite = int(values.size)
    count_missing_or_nonfinite = count_total - count_finite

    if count_finite == 0:
        return {
            "count": count_total,
            "finite_count": 0,
            "missing_or_nonfinite_count": count_missing_or_nonfinite,
            "mean": None,
            "std": None,
            "minimum": None,
            "q1": None,
            "median": None,
            "q3": None,
            "maximum": None,
            "iqr": None,
            "median_ci_lower": None,
            "median_ci_upper": None,
        }

    q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
    ci_lower, ci_upper = bootstrap_median_interval(
        values,
        confidence_level=confidence_level,
        samples=bootstrap_samples,
        rng=rng,
    )

    return {
        "count": count_total,
        "finite_count": count_finite,
        "missing_or_nonfinite_count": count_missing_or_nonfinite,
        "mean": safe_float(np.mean(values)),
        "std": safe_float(
            np.std(values, ddof=1) if count_finite > 1 else 0.0
        ),
        "minimum": safe_float(np.min(values)),
        "q1": safe_float(q1),
        "median": safe_float(median),
        "q3": safe_float(q3),
        "maximum": safe_float(np.max(values)),
        "iqr": safe_float(q3 - q1),
        "median_ci_lower": ci_lower,
        "median_ci_upper": ci_upper,
    }


def sorted_unique_values(
    frame: pd.DataFrame,
    column: str,
) -> list[Any]:
    """Extract stable JSON-safe unique values from a column."""
    if column not in frame.columns:
        return []

    series = frame[column].dropna()

    if series.empty:
        return []

    values = series.unique().tolist()

    def convert(value: Any) -> Any:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            numeric = float(value)
            return numeric if math.isfinite(numeric) else None
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    converted = [convert(value) for value in values]

    try:
        return sorted(converted)
    except TypeError:
        return sorted(converted, key=lambda item: str(item))


def build_dataset_overview(
    frame: pd.DataFrame,
    metric_columns: Sequence[str],
) -> dict[str, Any]:
    """Build a compact description of the complete result set."""
    overview: dict[str, Any] = {
        "records": int(len(frame)),
        "columns": list(frame.columns),
        "metric_columns": list(metric_columns),
        "duplicate_records": int(frame.duplicated().sum()),
    }

    for column in (
        "scenario",
        "constraint_type",
        "schedule",
        "query_name",
        "dimension",
        "source_polynomial_degree",
        "density_polynomial_degree",
        "basis",
        "dtype",
        "trial",
    ):
        if column in frame.columns:
            overview[column] = sorted_unique_values(frame, column)

    overview["finite_metric_counts"] = {
        metric: int(
            np.isfinite(
                pd.to_numeric(
                    frame[metric],
                    errors="coerce",
                ).to_numpy(dtype=np.float64, na_value=np.nan)
            ).sum()
        )
        for metric in metric_columns
    }

    return overview


def build_quality_summary(
    frame: pd.DataFrame,
    metric_columns: Sequence[str],
) -> dict[str, Any]:
    """Summarise finite-value and benchmark-health diagnostics."""
    quality: dict[str, Any] = {}

    all_metric_values: list[np.ndarray] = []

    for metric in metric_columns:
        numeric = pd.to_numeric(frame[metric], errors="coerce").to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )
        all_metric_values.append(numeric)

        quality[f"{metric}_finite_count"] = int(
            np.isfinite(numeric).sum()
        )
        quality[f"{metric}_nonfinite_count"] = int(
            (~np.isfinite(numeric)).sum()
        )

    if all_metric_values:
        stacked = np.column_stack(all_metric_values)
        quality["records_with_all_metrics_finite"] = int(
            np.all(np.isfinite(stacked), axis=1).sum()
        )
        quality["records_with_any_nonfinite_metric"] = int(
            np.any(~np.isfinite(stacked), axis=1).sum()
        )

    for column in BOOLEAN_COLUMNS:
        if column not in frame.columns:
            continue

        series = coerce_boolean_series(frame[column])
        quality[f"{column}_true_count"] = int(
            series.fillna(False).sum()
        )
        quality[f"{column}_false_count"] = int(
            (~series.fillna(True)).sum()
        )
        quality[f"{column}_missing_count"] = int(series.isna().sum())

    if "basis_condition_number" in frame.columns:
        values = pd.to_numeric(
            frame["basis_condition_number"],
            errors="coerce",
        ).to_numpy(dtype=np.float64, na_value=np.nan)

        quality["infinite_condition_numbers"] = int(
            np.isinf(values).sum()
        )

    return quality


def available_group_columns(
    frame: pd.DataFrame,
    requested: Sequence[str],
) -> list[str]:
    return [column for column in requested if column in frame.columns]


def grouped_metric_table(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
    confidence_level: float,
    bootstrap_samples: int,
    random_seed: int,
) -> pd.DataFrame:
    """Create a wide aggregate table for the requested grouping."""
    groups = available_group_columns(frame, group_columns)

    if not groups:
        grouped_items: Iterable[tuple[Any, pd.DataFrame]] = [
            ((), frame)
        ]
    else:
        grouped_items = frame.groupby(
            groups,
            dropna=False,
            sort=True,
            observed=False,
        )

    rows: list[dict[str, Any]] = []

    seed_sequence = np.random.SeedSequence(random_seed)
    child_seeds = seed_sequence.spawn(max(1, len(frame)))

    for group_index, (group_key, group_frame) in enumerate(grouped_items):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row: dict[str, Any] = {
            column: value
            for column, value in zip(groups, group_key)
        }
        row["record_count"] = int(len(group_frame))

        child_seed = child_seeds[
            min(group_index, len(child_seeds) - 1)
        ]
        rng = np.random.default_rng(child_seed)

        for metric in metric_columns:
            stats = descriptive_statistics(
                group_frame[metric],
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            )

            for statistic, value in stats.items():
                row[f"{metric}_{statistic}"] = value

        rows.append(row)

    table = pd.DataFrame(rows)

    if groups and not table.empty:
        table = sort_summary_table(table, groups)

    return table


def sort_summary_table(
    table: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Sort summary tables with stable basis and dtype ordering."""
    result = table.copy()

    temporary_columns: list[str] = []
    sort_columns: list[str] = []

    for column in group_columns:
        if column not in result.columns:
            continue

        if column == "basis":
            temp = "_basis_order"
            mapping = {
                value: index
                for index, value in enumerate(DEFAULT_BASE_ORDER)
            }
            result[temp] = (
                result[column]
                .map(mapping)
                .fillna(len(mapping))
            )
            temporary_columns.append(temp)
            sort_columns.append(temp)
        elif column == "dtype":
            temp = "_dtype_order"
            mapping = {
                value: index
                for index, value in enumerate(DEFAULT_DTYPE_ORDER)
            }
            result[temp] = (
                result[column]
                .map(mapping)
                .fillna(len(mapping))
            )
            temporary_columns.append(temp)
            sort_columns.append(temp)
        else:
            sort_columns.append(column)

    if sort_columns:
        result = result.sort_values(
            sort_columns,
            kind="stable",
            na_position="last",
        )

    if temporary_columns:
        result = result.drop(columns=temporary_columns)

    return result.reset_index(drop=True)


def compact_metric_summary(
    table: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    """Select the most useful statistics from a wide grouped table."""
    columns: list[str] = [
        column
        for column in group_columns
        if column in table.columns
    ]

    if "record_count" in table.columns:
        columns.append("record_count")

    suffixes = (
        "median",
        "q1",
        "q3",
        "maximum",
        "finite_count",
    )

    for metric in metric_columns:
        for suffix in suffixes:
            column = f"{metric}_{suffix}"
            if column in table.columns:
                columns.append(column)

    return table.loc[:, columns].copy()


def row_to_json_safe_dict(row: pd.Series) -> dict[str, Any]:
    """Convert a pandas row into a JSON-safe dictionary."""
    result: dict[str, Any] = {}

    for key, value in row.items():
        if pd.isna(value):
            result[key] = None
        elif isinstance(value, np.integer):
            result[key] = int(value)
        elif isinstance(value, np.floating):
            number = float(value)
            result[key] = number if math.isfinite(number) else None
        elif isinstance(value, np.bool_):
            result[key] = bool(value)
        elif isinstance(value, pd.Timestamp):
            result[key] = value.isoformat()
        else:
            result[key] = value

    return result


def extreme_record(
    frame: pd.DataFrame,
    metric: str,
    *,
    mode: str,
) -> dict[str, Any] | None:
    """Return the row containing the finite minimum or maximum metric."""
    values = pd.to_numeric(
        frame[metric],
        errors="coerce",
    )

    finite_mask = np.isfinite(
        values.to_numpy(dtype=np.float64, na_value=np.nan)
    )

    if not finite_mask.any():
        return None

    finite_frame = frame.loc[finite_mask].copy()
    finite_values = pd.to_numeric(
        finite_frame[metric],
        errors="coerce",
    )

    if mode == "minimum":
        index = finite_values.idxmin()
    elif mode == "maximum":
        index = finite_values.idxmax()
    else:
        raise ValueError(f"Unsupported extreme mode: {mode}")

    preferred_columns = [
        column
        for column in IDENTIFIER_COLUMNS
        if column in frame.columns
    ]

    preferred_columns.extend(
        column
        for column in (
            metric,
            *REFERENCE_COLUMNS,
            *COMPUTED_COLUMNS,
        )
        if column in frame.columns
        and column not in preferred_columns
    )

    return row_to_json_safe_dict(frame.loc[index, preferred_columns])


def build_extreme_records(
    frame: pd.DataFrame,
    metric_columns: Sequence[str],
) -> dict[str, Any]:
    """Find minimum and maximum finite record for every metric."""
    result: dict[str, Any] = {}

    for metric in metric_columns:
        result[metric] = {
            "minimum": extreme_record(
                frame,
                metric,
                mode="minimum",
            ),
            "maximum": extreme_record(
                frame,
                metric,
                mode="maximum",
            ),
        }

    return result


def rank_bases(
    basis_table: pd.DataFrame,
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Rank bases for every dtype and metric using aggregate medians.

    A lower rank is better. Ties receive the average rank.
    """
    if "basis" not in basis_table.columns:
        return pd.DataFrame()

    dtype_groups: list[str | None]

    if "dtype" in basis_table.columns:
        dtype_groups = [
            str(value)
            for value in basis_table["dtype"].dropna().unique()
        ]
    else:
        dtype_groups = [None]

    rows: list[dict[str, Any]] = []

    for dtype in dtype_groups:
        if dtype is None:
            subset = basis_table.copy()
        else:
            subset = basis_table[
                basis_table["dtype"].astype(str) == dtype
            ].copy()

        for metric in metric_columns:
            median_column = f"{metric}_median"

            if median_column not in subset.columns:
                continue

            ranking_frame = subset[
                ["basis", median_column]
            ].dropna()

            if ranking_frame.empty:
                continue

            descriptor = metric_descriptor(metric)
            ascending = descriptor.lower_is_better

            ranking_frame = ranking_frame.copy()
            ranking_frame["rank"] = ranking_frame[
                median_column
            ].rank(
                method="average",
                ascending=ascending,
            )

            ranking_frame["dtype"] = dtype
            ranking_frame["metric"] = metric
            ranking_frame["direction"] = descriptor.direction
            ranking_frame["category"] = descriptor.category
            ranking_frame = ranking_frame.rename(
                columns={median_column: "median_value"}
            )

            rows.extend(
                ranking_frame[
                    [
                        "dtype",
                        "metric",
                        "category",
                        "direction",
                        "basis",
                        "median_value",
                        "rank",
                    ]
                ].to_dict(orient="records")
            )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result["_dtype_order"] = result["dtype"].map(
        {
            value: index
            for index, value in enumerate(DEFAULT_DTYPE_ORDER)
        }
    ).fillna(len(DEFAULT_DTYPE_ORDER))

    result["_basis_order"] = result["basis"].map(
        {
            value: index
            for index, value in enumerate(DEFAULT_BASE_ORDER)
        }
    ).fillna(len(DEFAULT_BASE_ORDER))

    result = result.sort_values(
        [
            "_dtype_order",
            "metric",
            "rank",
            "_basis_order",
        ],
        kind="stable",
    ).drop(
        columns=["_dtype_order", "_basis_order"]
    )

    return result.reset_index(drop=True)


def build_overall_scores(
    ranking_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute mean rank by basis and dtype.

    Conditioning-only diagnostics and runtime remain visible separately, so
    both an all-metric score and an accuracy/stability-only score are emitted.
    """
    if ranking_table.empty:
        return pd.DataFrame()

    excluded_from_accuracy_score = {
        "runtime",
        "conditioning",
        "diagnostic",
    }

    group_columns = ["basis"]

    if "dtype" in ranking_table.columns:
        group_columns.insert(0, "dtype")

    rows: list[dict[str, Any]] = []

    for group_key, group_frame in ranking_table.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = {
            column: value
            for column, value in zip(group_columns, group_key)
        }

        row["all_metric_mean_rank"] = safe_float(
            group_frame["rank"].mean()
        )
        row["all_metric_count"] = int(len(group_frame))

        accuracy_frame = group_frame[
            ~group_frame["category"].isin(
                excluded_from_accuracy_score
            )
        ]

        row["accuracy_stability_mean_rank"] = safe_float(
            accuracy_frame["rank"].mean()
            if not accuracy_frame.empty
            else np.nan
        )
        row["accuracy_stability_metric_count"] = int(
            len(accuracy_frame)
        )

        runtime_frame = group_frame[
            group_frame["category"] == "runtime"
        ]
        row["runtime_mean_rank"] = safe_float(
            runtime_frame["rank"].mean()
            if not runtime_frame.empty
            else np.nan
        )

        conditioning_frame = group_frame[
            group_frame["category"].isin(
                {"conditioning", "diagnostic", "rank"}
            )
        ]
        row["conditioning_mean_rank"] = safe_float(
            conditioning_frame["rank"].mean()
            if not conditioning_frame.empty
            else np.nan
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    sort_columns = [
        column
        for column in ("dtype", "accuracy_stability_mean_rank")
        if column in result.columns
    ]

    if sort_columns:
        result = result.sort_values(
            sort_columns,
            kind="stable",
            na_position="last",
        )

    return result.reset_index(drop=True)


def dataframe_to_records(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records."""
    return [
        row_to_json_safe_dict(row)
        for _, row in frame.iterrows()
    ]


def write_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Write a CSV file, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def latex_escape(text: str) -> str:
    """Escape a string for safe use in a basic LaTeX table."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    result = str(text)

    for source, replacement in replacements.items():
        result = result.replace(source, replacement)

    return result


def scientific_notation_latex(
    value: Any,
    *,
    precision: int = 3,
) -> str:
    """Format a scalar using compact LaTeX scientific notation."""
    if value is None or pd.isna(value):
        return "--"

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return latex_escape(str(value))

    if not math.isfinite(numeric):
        return "--"

    if numeric == 0.0:
        return "$0$"

    absolute = abs(numeric)

    if 1.0e-3 <= absolute < 1.0e4:
        return f"${numeric:.{precision}g}$"

    exponent = int(math.floor(math.log10(absolute)))
    mantissa = numeric / (10.0**exponent)

    return (
        f"${mantissa:.{precision}f}"
        rf"\times10^{{{exponent}}}$"
    )


def write_latex_metric_table(
    table: pd.DataFrame,
    *,
    path: Path,
    metric_columns: Sequence[str],
    caption: str,
    label: str,
) -> None:
    """Write a compact LaTeX table of median values."""
    identifier_columns = [
        column
        for column in ("scenario", "basis", "dtype")
        if column in table.columns
    ]

    median_columns = [
        f"{metric}_median"
        for metric in metric_columns
        if f"{metric}_median" in table.columns
    ]

    selected_columns = identifier_columns + median_columns

    if not selected_columns:
        return

    compact = table[selected_columns].copy()

    display_headers = {
        "scenario": "Scenario",
        "basis": "Basis",
        "dtype": "Precision",
    }

    for metric in metric_columns:
        display_headers[f"{metric}_median"] = (
            metric.replace("_", " ").title()
        )

    alignment = (
        "l" * len(identifier_columns)
        + "r" * len(median_columns)
    )

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(
            latex_escape(
                display_headers.get(column, column)
            )
            for column in selected_columns
        )
        + r" \\",
        r"\midrule",
    ]

    for _, row in compact.iterrows():
        cells: list[str] = []

        for column in selected_columns:
            if column in identifier_columns:
                value = row[column]
                cells.append(
                    "--"
                    if pd.isna(value)
                    else latex_escape(str(value))
                )
            else:
                cells.append(
                    scientific_notation_latex(row[column])
                )

        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_rank_table(
    scores: pd.DataFrame,
    path: Path,
) -> None:
    """Write a LaTeX table containing overall basis ranks."""
    if scores.empty:
        return

    columns = [
        column
        for column in (
            "dtype",
            "basis",
            "accuracy_stability_mean_rank",
            "conditioning_mean_rank",
            "runtime_mean_rank",
            "all_metric_mean_rank",
        )
        if column in scores.columns
    ]

    alignment = "ll" + "r" * max(0, len(columns) - 2)

    headers = {
        "dtype": "Precision",
        "basis": "Basis",
        "accuracy_stability_mean_rank": "Accuracy/stability rank",
        "conditioning_mean_rank": "Conditioning rank",
        "runtime_mean_rank": "Runtime rank",
        "all_metric_mean_rank": "All-metric rank",
    }

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        (
            r"\caption{Mean representative PAL benchmark rank "
            r"by basis and arithmetic precision.}"
        ),
        r"\label{tab:representative-pal-basis-ranks}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(headers[column] for column in columns) + r" \\",
        r"\midrule",
    ]

    for _, row in scores.iterrows():
        cells: list[str] = []

        for column in columns:
            value = row[column]

            if column in {"dtype", "basis"}:
                cells.append(
                    "--"
                    if pd.isna(value)
                    else latex_escape(str(value))
                )
            else:
                cells.append(
                    "--"
                    if pd.isna(value)
                    else f"${float(value):.3f}$"
                )

        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def json_default(value: Any) -> Any:
    """JSON serializer for NumPy and pandas values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serialisable"
    )


def write_json(
    payload: Mapping[str, Any],
    path: Path,
) -> None:
    """Write formatted JSON, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=False,
            allow_nan=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def print_heading(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def print_compact_table(
    table: pd.DataFrame,
    *,
    max_rows: int = 60,
) -> None:
    if table.empty:
        print("(no rows)")
        return

    with pd.option_context(
        "display.max_rows",
        max_rows,
        "display.max_columns",
        None,
        "display.width",
        220,
        "display.float_format",
        lambda value: f"{value:.4e}",
    ):
        print(table.to_string(index=False))


def choose_console_metrics(
    metric_columns: Sequence[str],
) -> list[str]:
    """Select the most informative metrics for console output."""
    preferred = (
        "partition_function_relative_error",
        "query_probability_absolute_error",
        "query_probability_relative_error",
        "relative_integration_error",
        "basis_condition_number",
        "numerical_rank_fraction",
        "coefficient_perturbation_sensitivity",
        "query_probability_perturbation_sensitivity",
        "basis_conversion_residual",
        "total_evaluation_ms",
    )

    selected = [
        metric
        for metric in preferred
        if metric in metric_columns
    ]

    if selected:
        return selected

    return list(metric_columns[:8])


def print_console_summary(
    *,
    frame: pd.DataFrame,
    warnings: Sequence[str],
    metric_columns: Sequence[str],
    basis_table: pd.DataFrame,
    ranking_table: pd.DataFrame,
    overall_scores: pd.DataFrame,
    output_paths: Sequence[Path],
) -> None:
    print_heading("Representative PAL benchmark summary")
    print(f"records: {len(frame)}")
    print(f"columns: {len(frame.columns)}")
    print(f"metrics: {len(metric_columns)}")

    if "basis" in frame.columns:
        print(
            "bases: "
            + str(sorted_unique_values(frame, "basis"))
        )

    if "dtype" in frame.columns:
        print(
            "dtypes: "
            + str(sorted_unique_values(frame, "dtype"))
        )

    if "scenario" in frame.columns:
        print(
            "scenarios: "
            + str(sorted_unique_values(frame, "scenario"))
        )

    if warnings:
        print_heading("Warnings")
        for warning in warnings:
            print(f"- {warning}")

    console_metrics = choose_console_metrics(metric_columns)

    if not basis_table.empty:
        console_columns = [
            column
            for column in ("basis", "dtype", "record_count")
            if column in basis_table.columns
        ]

        for metric in console_metrics:
            median_column = f"{metric}_median"
            if median_column in basis_table.columns:
                console_columns.append(median_column)

        print_heading("Median metrics by basis and dtype")
        print_compact_table(
            basis_table[console_columns]
        )

    if not ranking_table.empty:
        print_heading("Best basis by metric and dtype")

        best_rows = (
            ranking_table[
                ranking_table["rank"]
                == ranking_table.groupby(
                    ["dtype", "metric"],
                    dropna=False,
                )["rank"].transform("min")
            ]
            .sort_values(
                ["dtype", "metric", "basis"],
                kind="stable",
            )
        )

        print_compact_table(
            best_rows[
                [
                    "dtype",
                    "metric",
                    "direction",
                    "basis",
                    "median_value",
                    "rank",
                ]
            ]
        )

    if not overall_scores.empty:
        print_heading("Aggregate basis ranking")
        print_compact_table(overall_scores)

    print_heading("Written files")
    for path in output_paths:
        print(path)


def create_summary(
    *,
    frame: pd.DataFrame,
    config: SummaryConfig,
    warnings: Sequence[str],
    metric_columns: Sequence[str],
    basis_table: pd.DataFrame,
    scenario_table: pd.DataFrame,
    configuration_table: pd.DataFrame,
    ranking_table: pd.DataFrame,
    overall_scores: pd.DataFrame,
) -> dict[str, Any]:
    """Assemble the complete JSON summary."""
    return {
        "summary_version": 1,
        "benchmark": "representative_pal",
        "configuration": asdict(config),
        "warnings": list(warnings),
        "dataset": build_dataset_overview(
            frame,
            metric_columns,
        ),
        "quality": build_quality_summary(
            frame,
            metric_columns,
        ),
        "metric_descriptors": [
            asdict(metric_descriptor(metric))
            for metric in metric_columns
        ],
        "aggregate_by_basis_and_dtype": dataframe_to_records(
            compact_metric_summary(
                basis_table,
                group_columns=("basis", "dtype"),
                metric_columns=metric_columns,
            )
        ),
        "aggregate_by_scenario_basis_and_dtype": dataframe_to_records(
            compact_metric_summary(
                scenario_table,
                group_columns=(
                    "scenario",
                    "basis",
                    "dtype",
                ),
                metric_columns=metric_columns,
            )
        ),
        "aggregate_by_configuration": dataframe_to_records(
            compact_metric_summary(
                configuration_table,
                group_columns=(
                    "scenario",
                    "dimension",
                    "density_polynomial_degree",
                    "basis",
                    "dtype",
                ),
                metric_columns=metric_columns,
            )
        ),
        "basis_rankings": dataframe_to_records(ranking_table),
        "overall_basis_scores": dataframe_to_records(
            overall_scores
        ),
        "extreme_records": build_extreme_records(
            frame,
            metric_columns,
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)

    config = SummaryConfig(
        results_path=str(args.results_path),
        summary_path=str(args.summary_path),
        tables_dir=str(args.tables_dir),
        latex_dir=(
            None if args.no_latex else str(args.latex_dir)
        ),
        confidence_level=float(args.confidence_level),
        bootstrap_samples=int(args.bootstrap_samples),
        random_seed=int(args.random_seed),
        write_latex=not bool(args.no_latex),
    )

    try:
        frame = load_results(args.results_path)
        warnings = validate_schema(
            frame,
            strict=bool(args.strict),
        )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(
            f"Error: {error}",
            file=sys.stderr,
        )
        return 1

    metric_columns = discover_metric_columns(frame)

    basis_group_columns = available_group_columns(
        frame,
        ("basis", "dtype"),
    )
    scenario_group_columns = available_group_columns(
        frame,
        ("scenario", "basis", "dtype"),
    )
    configuration_group_columns = available_group_columns(
        frame,
        (
            "scenario",
            "constraint_type",
            "query_name",
            "schedule",
            "dimension",
            "source_polynomial_degree",
            "density_polynomial_degree",
            "basis",
            "dtype",
        ),
    )

    basis_table = grouped_metric_table(
        frame,
        group_columns=basis_group_columns,
        metric_columns=metric_columns,
        confidence_level=args.confidence_level,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )

    scenario_table = grouped_metric_table(
        frame,
        group_columns=scenario_group_columns,
        metric_columns=metric_columns,
        confidence_level=args.confidence_level,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed + 1,
    )

    configuration_table = grouped_metric_table(
        frame,
        group_columns=configuration_group_columns,
        metric_columns=metric_columns,
        confidence_level=args.confidence_level,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed + 2,
    )

    ranking_table = rank_bases(
        basis_table,
        metric_columns,
    )
    overall_scores = build_overall_scores(ranking_table)

    summary = create_summary(
        frame=frame,
        config=config,
        warnings=warnings,
        metric_columns=metric_columns,
        basis_table=basis_table,
        scenario_table=scenario_table,
        configuration_table=configuration_table,
        ranking_table=ranking_table,
        overall_scores=overall_scores,
    )

    output_paths: list[Path] = []

    write_json(summary, args.summary_path)
    output_paths.append(args.summary_path)

    basis_path = (
        args.tables_dir
        / "representative_pal_by_basis_and_dtype.csv"
    )
    scenario_path = (
        args.tables_dir
        / "representative_pal_by_scenario_basis_dtype.csv"
    )
    configuration_path = (
        args.tables_dir
        / "representative_pal_by_configuration.csv"
    )
    ranking_path = (
        args.tables_dir
        / "representative_pal_basis_rankings.csv"
    )
    score_path = (
        args.tables_dir
        / "representative_pal_overall_basis_scores.csv"
    )

    write_csv(basis_table, basis_path)
    write_csv(scenario_table, scenario_path)
    write_csv(configuration_table, configuration_path)
    write_csv(ranking_table, ranking_path)
    write_csv(overall_scores, score_path)

    output_paths.extend(
        [
            basis_path,
            scenario_path,
            configuration_path,
            ranking_path,
            score_path,
        ]
    )

    if not args.no_latex:
        basis_latex_path = (
            args.latex_dir
            / "representative_pal_basis_summary.tex"
        )
        scenario_latex_path = (
            args.latex_dir
            / "representative_pal_scenario_summary.tex"
        )
        rank_latex_path = (
            args.latex_dir
            / "representative_pal_basis_ranks.tex"
        )

        preferred_latex_metrics = [
            metric
            for metric in (
                "partition_function_relative_error",
                "query_probability_absolute_error",
                "relative_integration_error",
                "basis_condition_number",
                "numerical_rank_fraction",
                "coefficient_perturbation_sensitivity",
                "total_evaluation_ms",
            )
            if metric in metric_columns
        ]

        if not preferred_latex_metrics:
            preferred_latex_metrics = list(metric_columns[:6])

        write_latex_metric_table(
            basis_table,
            path=basis_latex_path,
            metric_columns=preferred_latex_metrics,
            caption=(
                "Median representative PAL benchmark metrics by "
                "polynomial basis and arithmetic precision."
            ),
            label="tab:representative-pal-basis-summary",
        )

        write_latex_metric_table(
            scenario_table,
            path=scenario_latex_path,
            metric_columns=preferred_latex_metrics,
            caption=(
                "Median representative PAL benchmark metrics by "
                "scenario, polynomial basis, and arithmetic precision."
            ),
            label="tab:representative-pal-scenario-summary",
        )

        write_latex_rank_table(
            overall_scores,
            rank_latex_path,
        )

        for path in (
            basis_latex_path,
            scenario_latex_path,
            rank_latex_path,
        ):
            if path.exists():
                output_paths.append(path)

    if not args.quiet:
        print_console_summary(
            frame=frame,
            warnings=warnings,
            metric_columns=metric_columns,
            basis_table=basis_table,
            ranking_table=ranking_table,
            overall_scores=overall_scores,
            output_paths=output_paths,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())