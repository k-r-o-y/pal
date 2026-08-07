#!/usr/bin/env python3
"""
Generate publication-quality figures for the representative PAL benchmark.

Expected input
--------------
A CSV produced by ``analysis.pal.run_representative_pal_benchmark``. The script is
deliberately tolerant of minor schema differences and resolves several common aliases
for each metric.

Typical usage
-------------
python -m analysis.pal.plot_representative_pal_results \
    --results-path results/pal/representative_pal_results.csv \
    --summary-path results/pal/representative_pal_summary.json \
    --output-dir figures \
    --formats pdf png svg

The script generates:

* pal_partition_function_error_vs_degree
* pal_query_probability_error_vs_degree
* pal_condition_number_vs_degree
* pal_rank_fraction_vs_degree
* pal_perturbation_sensitivity_vs_degree
* pal_coefficient_noise_amplification_vs_degree
* pal_basis_conversion_residual_vs_degree
* pal_runtime_vs_degree
* pal_partition_function_error_vs_dimension
* pal_query_probability_error_vs_dimension
* pal_runtime_vs_dimension
* pal_error_by_scenario
* pal_runtime_by_scenario
* pal_error_summary

All figures are written in every requested format. A JSON plotting summary is also
written to ``<output-dir>/../results/pal`` unless ``--plot-summary-path`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants and schema aliases
# ---------------------------------------------------------------------------

DEFAULT_BASES = ("monomial", "legendre", "chebyshev")
DEFAULT_DTYPES = ("float32", "float64")
DEFAULT_FORMATS = ("pdf", "png", "svg")

BASIS_LABELS = {
    "monomial": "Monomial",
    "legendre": "Legendre",
    "chebyshev": "Chebyshev",
}

DTYPE_LABELS = {
    "float32": "float32",
    "float64": "float64",
}

LINE_STYLES = {
    "float32": "--",
    "float64": "-",
}

MARKERS = {
    "monomial": "o",
    "legendre": "s",
    "chebyshev": "^",
}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "basis": (
        "basis",
        "basis_name",
        "polynomial_basis",
        "representation",
    ),
    "dtype": (
        "dtype",
        "precision",
        "arithmetic",
        "floating_point_dtype",
    ),
    "dimension": (
        "dimension",
        "n_dimensions",
        "num_dimensions",
        "input_dimension",
    ),
    "source_degree": (
        "source_polynomial_degree",
        "source_degree",
        "q_degree",
        "base_degree",
    ),
    "density_degree": (
        "density_polynomial_degree",
        "density_degree",
        "polynomial_degree",
        "degree",
    ),
    "scenario": (
        "scenario",
        "constraint_scenario",
        "pal_scenario",
        "case",
    ),
    "trial": (
        "trial",
        "trial_index",
        "repeat",
        "replicate",
    ),
    "query_scale": (
        "query_scale",
        "query_region_scale",
        "query_fraction",
    ),
    "partition_error": (
        "partition_function_relative_error",
        "partition_relative_error",
        "partition_error",
        "relative_partition_error",
        "z_relative_error",
        "z_error",
    ),
    "query_error": (
        "query_probability_absolute_error",
        "query_absolute_error",
        "query_error",
        "absolute_query_error",
        "normalised_query_absolute_error",
        "normalized_query_absolute_error",
    ),
    "query_relative_error": (
        "query_probability_relative_error",
        "query_relative_error",
        "relative_query_error",
    ),
    "condition_number": (
        "basis_condition_number",
        "condition_number",
        "matrix_condition_number",
        "vandermonde_condition_number",
    ),
    "rank_fraction": (
        "numerical_rank_fraction",
        "rank_fraction",
        "retained_rank_fraction",
    ),
    "perturbation_sensitivity": (
        "perturbation_sensitivity",
        "partition_perturbation_sensitivity",
        "query_perturbation_sensitivity",
        "coefficient_perturbation_sensitivity",
    ),
    "noise_amplification": (
        "coefficient_noise_amplification",
        "noise_amplification",
        "coefficient_recovery_amplification",
    ),
    "conversion_residual": (
        "basis_conversion_residual",
        "conversion_residual",
        "relative_conversion_residual",
    ),
    "runtime_ms": (
        "total_evaluation_ms",
        "runtime_ms",
        "total_runtime_ms",
        "evaluation_runtime_ms",
        "elapsed_ms",
    ),
    "partition_reference": (
        "reference_partition_function",
        "partition_reference",
        "reference_z",
    ),
    "partition_computed": (
        "computed_partition_function",
        "partition_computed",
        "computed_z",
    ),
    "query_reference": (
        "reference_query_probability",
        "query_reference",
        "reference_query",
    ),
    "query_computed": (
        "computed_query_probability",
        "query_computed",
        "computed_query",
    ),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlotConfig:
    results_path: Path
    summary_path: Path | None
    output_dir: Path
    plot_summary_path: Path
    formats: tuple[str, ...]
    dpi: int
    detail_dimension: int | None
    detail_scenario: str | None
    detail_query_scale: float | None
    bases: tuple[str, ...]
    dtypes: tuple[str, ...]


@dataclass(frozen=True)
class MetricSpec:
    canonical_name: str
    title: str
    y_label: str
    log_y: bool = True
    positive_floor: float | None = None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot representative PAL benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results/pal/representative_pal_results.csv"),
        help="Row-level representative PAL benchmark CSV.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("results/pal/representative_pal_summary.json"),
        help="Optional benchmark summary JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--plot-summary-path",
        type=Path,
        default=Path("results/pal/representative_pal_plot_summary.json"),
        help="JSON path for aggregate statistics used by the plotting script.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(DEFAULT_FORMATS),
        choices=("pdf", "png", "svg"),
        help="Figure formats to write.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output resolution.",
    )
    parser.add_argument(
        "--detail-dimension",
        type=int,
        default=None,
        help="Dimension used for degree-based detail figures. Defaults to the largest available dimension.",
    )
    parser.add_argument(
        "--detail-scenario",
        type=str,
        default=None,
        help="Scenario used for detail figures. Defaults to a dynamic scenario when available.",
    )
    parser.add_argument(
        "--detail-query-scale",
        type=float,
        default=None,
        help="Query scale used for query-error detail figures. Defaults to 0.5 or the nearest available value.",
    )
    parser.add_argument(
        "--bases",
        nargs="+",
        default=list(DEFAULT_BASES),
        help="Basis values to include.",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=list(DEFAULT_DTYPES),
        help="Arithmetic dtypes to include.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Validation and schema normalisation
# ---------------------------------------------------------------------------

def ensure_file(path: Path, *, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{description} is not a regular file: {path}")


def first_existing_column(
    columns: Iterable[str],
    aliases: Sequence[str],
) -> str | None:
    column_set = set(columns)
    for alias in aliases:
        if alias in column_set:
            return alias
    return None


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for canonical_name, aliases in COLUMN_ALIASES.items():
        actual = first_existing_column(df.columns, aliases)
        if actual is not None:
            resolved[canonical_name] = actual
    return resolved


def canonicalise_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    resolved = resolve_columns(df)
    renamed = df.copy()

    rename_map = {
        actual: canonical
        for canonical, actual in resolved.items()
        if actual != canonical
    }
    renamed = renamed.rename(columns=rename_map)

    for name in (
        "dimension",
        "source_degree",
        "density_degree",
        "trial",
        "query_scale",
        "partition_error",
        "query_error",
        "query_relative_error",
        "condition_number",
        "rank_fraction",
        "perturbation_sensitivity",
        "noise_amplification",
        "conversion_residual",
        "runtime_ms",
        "partition_reference",
        "partition_computed",
        "query_reference",
        "query_computed",
    ):
        if name in renamed.columns:
            renamed[name] = pd.to_numeric(renamed[name], errors="coerce")

    if "basis" in renamed.columns:
        renamed["basis"] = (
            renamed["basis"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    if "dtype" in renamed.columns:
        renamed["dtype"] = (
            renamed["dtype"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(
                {
                    "np.float32": "float32",
                    "numpy.float32": "float32",
                    "<class 'numpy.float32'>": "float32",
                    "torch.float32": "float32",
                    "fp32": "float32",
                    "single": "float32",
                    "np.float64": "float64",
                    "numpy.float64": "float64",
                    "<class 'numpy.float64'>": "float64",
                    "torch.float64": "float64",
                    "fp64": "float64",
                    "double": "float64",
                }
            )
        )

    if "scenario" in renamed.columns:
        renamed["scenario"] = (
            renamed["scenario"]
            .astype(str)
            .str.strip()
        )

    # Derive missing errors when reference and computed values are available.
    eps = np.finfo(np.float64).eps

    if (
        "partition_error" not in renamed.columns
        and {"partition_reference", "partition_computed"} <= set(renamed.columns)
    ):
        denominator = np.maximum(np.abs(renamed["partition_reference"]), eps)
        renamed["partition_error"] = (
            np.abs(renamed["partition_computed"] - renamed["partition_reference"])
            / denominator
        )

    if (
        "query_error" not in renamed.columns
        and {"query_reference", "query_computed"} <= set(renamed.columns)
    ):
        renamed["query_error"] = np.abs(
            renamed["query_computed"] - renamed["query_reference"]
        )

    if (
        "query_relative_error" not in renamed.columns
        and {"query_reference", "query_computed"} <= set(renamed.columns)
    ):
        denominator = np.maximum(np.abs(renamed["query_reference"]), eps)
        renamed["query_relative_error"] = (
            np.abs(renamed["query_computed"] - renamed["query_reference"])
            / denominator
        )

    # If only one degree column exists, expose it under both canonical names.
    if "density_degree" not in renamed.columns and "source_degree" in renamed.columns:
        renamed["density_degree"] = renamed["source_degree"]

    if "source_degree" not in renamed.columns and "density_degree" in renamed.columns:
        renamed["source_degree"] = renamed["density_degree"]

    return renamed, resolved


def validate_minimum_schema(df: pd.DataFrame) -> None:
    required = {"basis", "dtype"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "The results CSV is missing required columns after alias resolution: "
            + ", ".join(sorted(missing))
        )

    if "density_degree" not in df.columns and "dimension" not in df.columns:
        raise ValueError(
            "The results CSV must contain at least a degree column or a dimension column."
        )

    metric_columns = {
        "partition_error",
        "query_error",
        "condition_number",
        "rank_fraction",
        "perturbation_sensitivity",
        "noise_amplification",
        "conversion_residual",
        "runtime_ms",
    }
    if not (metric_columns & set(df.columns)):
        raise ValueError(
            "No plottable metric column was found. Available columns are:\n"
            + ", ".join(df.columns)
        )


def finite_record_mask(df: pd.DataFrame) -> pd.Series:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.Series(True, index=df.index)
    return np.isfinite(numeric).all(axis=1)


def filter_requested_values(
    df: pd.DataFrame,
    *,
    bases: Sequence[str],
    dtypes: Sequence[str],
) -> pd.DataFrame:
    result = df.copy()

    requested_bases = {str(v).lower() for v in bases}
    requested_dtypes = {str(v).lower() for v in dtypes}

    if requested_bases:
        result = result[result["basis"].isin(requested_bases)]
    if requested_dtypes:
        result = result[result["dtype"].isin(requested_dtypes)]

    if result.empty:
        raise ValueError(
            "No records remain after applying basis and dtype filters."
        )

    return result


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def choose_detail_dimension(df: pd.DataFrame, requested: int | None) -> int | None:
    if "dimension" not in df.columns:
        return None

    values = sorted(
        int(v)
        for v in df["dimension"].dropna().unique()
        if np.isfinite(v)
    )
    if not values:
        return None

    if requested is None:
        return values[-1]
    if requested not in values:
        raise ValueError(
            f"Requested detail dimension {requested} is unavailable. "
            f"Available dimensions: {values}"
        )
    return requested


def choose_detail_scenario(df: pd.DataFrame, requested: str | None) -> str | None:
    if "scenario" not in df.columns:
        return None

    values = [str(v) for v in df["scenario"].dropna().unique()]
    if not values:
        return None

    if requested is not None:
        if requested not in values:
            raise ValueError(
                f"Requested detail scenario {requested!r} is unavailable. "
                f"Available scenarios: {values}"
            )
        return requested

    dynamic_candidates = [
        value
        for value in values
        if "dynamic" in value.lower()
    ]
    if dynamic_candidates:
        return sorted(dynamic_candidates)[0]

    obstacle_candidates = [
        value
        for value in values
        if "obstacle" in value.lower()
    ]
    if obstacle_candidates:
        return sorted(obstacle_candidates)[0]

    return sorted(values)[0]


def choose_detail_query_scale(
    df: pd.DataFrame,
    requested: float | None,
) -> float | None:
    if "query_scale" not in df.columns:
        return None

    values = sorted(
        float(v)
        for v in df["query_scale"].dropna().unique()
        if np.isfinite(v)
    )
    if not values:
        return None

    target = 0.5 if requested is None else requested
    return min(values, key=lambda value: abs(value - target))


def apply_detail_filters(
    df: pd.DataFrame,
    *,
    dimension: int | None,
    scenario: str | None,
    query_scale: float | None = None,
) -> pd.DataFrame:
    result = df.copy()

    if dimension is not None and "dimension" in result.columns:
        result = result[result["dimension"] == dimension]

    if scenario is not None and "scenario" in result.columns:
        result = result[result["scenario"] == scenario]

    if query_scale is not None and "query_scale" in result.columns:
        result = result[np.isclose(result["query_scale"], query_scale)]

    return result


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    available_groups = [column for column in group_columns if column in df.columns]
    if metric not in df.columns:
        return pd.DataFrame()

    working = df[available_groups + [metric]].dropna(subset=[metric]).copy()
    if working.empty:
        return pd.DataFrame()

    grouped = (
        working.groupby(available_groups, dropna=False, observed=True)[metric]
        .agg(
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            minimum="min",
            maximum="max",
            mean="mean",
            count="count",
        )
        .reset_index()
    )
    return grouped


def aggregate_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")
    return [json_safe_mapping(record) for record in records]


def json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def json_safe_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): json_safe_value(value) for key, value in mapping.items()}


# ---------------------------------------------------------------------------
# Plot styling and file output
# ---------------------------------------------------------------------------

def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8.4, 5.2),
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "axes.grid": True,
            "axes.grid.axis": "both",
            "axes.grid.which": "major",
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
        }
    )


def save_figure(
    fig: plt.Figure,
    *,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        save_kwargs: dict[str, Any] = {}
        if fmt == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)
        written.append(path)
        print(f"Wrote {path}")

    plt.close(fig)
    return written


def positive_plot_values(
    values: pd.Series,
    *,
    floor: float | None = None,
) -> tuple[np.ndarray, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    positive = numeric[np.isfinite(numeric) & (numeric > 0)]

    if floor is None:
        if positive.size:
            floor = max(np.min(positive) * 0.1, np.finfo(np.float64).tiny)
        else:
            floor = 1e-18

    plotted = np.where(np.isfinite(numeric) & (numeric > 0), numeric, floor)
    return plotted, floor


def display_label(value: Any) -> str:
    text = str(value)
    return text.replace("_", " ").strip().title()


def legend_label(basis: str, dtype: str) -> str:
    basis_text = BASIS_LABELS.get(basis, display_label(basis))
    dtype_text = DTYPE_LABELS.get(dtype, dtype)
    return f"{basis_text}, {dtype_text}"


def basis_order(df: pd.DataFrame, requested: Sequence[str]) -> list[str]:
    present = set(df["basis"].dropna().astype(str))
    ordered = [basis for basis in requested if basis in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def dtype_order(df: pd.DataFrame, requested: Sequence[str]) -> list[str]:
    present = set(df["dtype"].dropna().astype(str))
    ordered = [dtype for dtype in requested if dtype in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


# ---------------------------------------------------------------------------
# Generic line figures
# ---------------------------------------------------------------------------

def plot_metric_vs_x(
    df: pd.DataFrame,
    *,
    metric_spec: MetricSpec,
    x_column: str,
    x_label: str,
    stem: str,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    requested_bases: Sequence[str],
    requested_dtypes: Sequence[str],
    title_suffix: str | None = None,
) -> list[Path]:
    metric = metric_spec.canonical_name
    if metric not in df.columns or x_column not in df.columns:
        print(f"Skipping {stem}: missing {metric!r} or {x_column!r}.")
        return []

    aggregated = aggregate_metric(
        df,
        metric=metric,
        group_columns=[x_column, "basis", "dtype"],
    )
    if aggregated.empty:
        print(f"Skipping {stem}: no finite data.")
        return []

    fig, ax = plt.subplots()

    bases = basis_order(aggregated, requested_bases)
    dtypes = dtype_order(aggregated, requested_dtypes)

    for basis in bases:
        for dtype in dtypes:
            subset = aggregated[
                (aggregated["basis"] == basis)
                & (aggregated["dtype"] == dtype)
            ].sort_values(x_column)

            if subset.empty:
                continue

            x = subset[x_column].to_numpy(dtype=float)
            median = subset["median"]
            q1 = subset["q1"]
            q3 = subset["q3"]

            if metric_spec.log_y:
                y, floor = positive_plot_values(
                    median,
                    floor=metric_spec.positive_floor,
                )
                lower, _ = positive_plot_values(q1, floor=floor)
                upper, _ = positive_plot_values(q3, floor=floor)
            else:
                y = median.to_numpy(dtype=float)
                lower = q1.to_numpy(dtype=float)
                upper = q3.to_numpy(dtype=float)

            ax.plot(
                x,
                y,
                marker=MARKERS.get(basis, "o"),
                linestyle=LINE_STYLES.get(dtype, "-"),
                label=legend_label(basis, dtype),
            )
            ax.fill_between(x, lower, upper, alpha=0.10)

    title = metric_spec.title
    if title_suffix:
        title = f"{title}\n{title_suffix}"

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric_spec.y_label)

    if metric_spec.log_y:
        ax.set_yscale("log")

    unique_x = sorted(
        value
        for value in aggregated[x_column].dropna().unique()
        if np.isfinite(value)
    )
    if len(unique_x) <= 12:
        ax.set_xticks(unique_x)

    ax.legend(ncol=2, loc="best")
    fig.tight_layout()

    return save_figure(
        fig,
        output_dir=output_dir,
        stem=stem,
        formats=formats,
        dpi=dpi,
    )


# ---------------------------------------------------------------------------
# Scenario figures
# ---------------------------------------------------------------------------

def plot_metric_by_scenario(
    df: pd.DataFrame,
    *,
    metric_spec: MetricSpec,
    stem: str,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    requested_bases: Sequence[str],
    requested_dtypes: Sequence[str],
) -> list[Path]:
    metric = metric_spec.canonical_name
    if metric not in df.columns or "scenario" not in df.columns:
        print(f"Skipping {stem}: scenario or metric unavailable.")
        return []

    aggregated = aggregate_metric(
        df,
        metric=metric,
        group_columns=["scenario", "basis", "dtype"],
    )
    if aggregated.empty:
        print(f"Skipping {stem}: no finite data.")
        return []

    scenarios = sorted(aggregated["scenario"].astype(str).unique())
    bases = basis_order(aggregated, requested_bases)
    dtypes = dtype_order(aggregated, requested_dtypes)

    series = [
        (basis, dtype)
        for basis in bases
        for dtype in dtypes
        if not aggregated[
            (aggregated["basis"] == basis)
            & (aggregated["dtype"] == dtype)
        ].empty
    ]

    x = np.arange(len(scenarios), dtype=float)
    group_width = 0.84
    bar_width = group_width / max(len(series), 1)

    fig_width = max(8.4, 1.35 * len(scenarios) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.4))

    for index, (basis, dtype) in enumerate(series):
        values: list[float] = []
        for scenario in scenarios:
            row = aggregated[
                (aggregated["scenario"].astype(str) == scenario)
                & (aggregated["basis"] == basis)
                & (aggregated["dtype"] == dtype)
            ]
            values.append(float(row["median"].iloc[0]) if not row.empty else np.nan)

        values_array = np.asarray(values, dtype=float)
        if metric_spec.log_y:
            values_array, _ = positive_plot_values(
                pd.Series(values_array),
                floor=metric_spec.positive_floor,
            )

        offset = (
            index - (len(series) - 1) / 2
        ) * bar_width

        ax.bar(
            x + offset,
            values_array,
            width=bar_width * 0.94,
            label=legend_label(basis, dtype),
        )

    ax.set_title(metric_spec.title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(metric_spec.y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [display_label(value) for value in scenarios],
        rotation=20,
        ha="right",
    )
    if metric_spec.log_y:
        ax.set_yscale("log")

    ax.legend(ncol=2, loc="best")
    fig.tight_layout()

    return save_figure(
        fig,
        output_dir=output_dir,
        stem=stem,
        formats=formats,
        dpi=dpi,
    )


# ---------------------------------------------------------------------------
# Combined summary figure
# ---------------------------------------------------------------------------

def plot_error_summary(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    requested_bases: Sequence[str],
    requested_dtypes: Sequence[str],
) -> list[Path]:
    metrics = [
        ("partition_error", "Partition-function relative error"),
        ("query_error", "Query-probability absolute error"),
    ]
    available = [
        (metric, label)
        for metric, label in metrics
        if metric in df.columns
    ]

    if not available:
        print("Skipping pal_error_summary: no downstream error metrics.")
        return []

    group_columns = ["basis", "dtype"]
    bases = basis_order(df, requested_bases)
    dtypes = dtype_order(df, requested_dtypes)
    series = [(basis, dtype) for basis in bases for dtype in dtypes]

    x = np.arange(len(available), dtype=float)
    group_width = 0.84
    bar_width = group_width / max(len(series), 1)

    fig, ax = plt.subplots(figsize=(8.6, 5.3))

    all_positive: list[float] = []
    prepared: dict[tuple[str, str], list[float]] = {}

    for basis, dtype in series:
        values: list[float] = []
        for metric, _ in available:
            subset = df[
                (df["basis"] == basis)
                & (df["dtype"] == dtype)
            ][metric].dropna()
            value = float(subset.median()) if not subset.empty else np.nan
            values.append(value)
            if np.isfinite(value) and value > 0:
                all_positive.append(value)
        prepared[(basis, dtype)] = values

    floor = (
        max(min(all_positive) * 0.1, np.finfo(np.float64).tiny)
        if all_positive
        else 1e-18
    )

    for index, (basis, dtype) in enumerate(series):
        values = np.asarray(prepared[(basis, dtype)], dtype=float)
        values = np.where(
            np.isfinite(values) & (values > 0),
            values,
            floor,
        )
        offset = (
            index - (len(series) - 1) / 2
        ) * bar_width

        ax.bar(
            x + offset,
            values,
            width=bar_width * 0.94,
            label=legend_label(basis, dtype),
        )

    ax.set_title("Representative PAL downstream error summary")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Median numerical error")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in available])
    ax.legend(ncol=2, loc="best")
    fig.tight_layout()

    return save_figure(
        fig,
        output_dir=output_dir,
        stem="pal_error_summary",
        formats=formats,
        dpi=dpi,
    )


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def print_configuration(
    *,
    config: PlotConfig,
    df: pd.DataFrame,
    detail_dimension: int | None,
    detail_scenario: str | None,
    detail_query_scale: float | None,
) -> None:
    print()
    print("Representative PAL plotting configuration")
    print("==========================================")
    print(f"results path: {config.results_path}")
    print(f"summary path: {config.summary_path}")
    print(f"output directory: {config.output_dir}")
    print(f"formats: {list(config.formats)}")
    print(f"records: {len(df)}")
    print(f"finite records: {int(finite_record_mask(df).sum())}")

    if "dimension" in df.columns:
        print(
            "dimensions: "
            + str(sorted(int(v) for v in df["dimension"].dropna().unique()))
        )
    if "source_degree" in df.columns:
        print(
            "source degrees: "
            + str(sorted(int(v) for v in df["source_degree"].dropna().unique()))
        )
    if "density_degree" in df.columns:
        print(
            "density degrees: "
            + str(sorted(int(v) for v in df["density_degree"].dropna().unique()))
        )
    if "scenario" in df.columns:
        print(
            "scenarios: "
            + str(sorted(str(v) for v in df["scenario"].dropna().unique()))
        )
    if "query_scale" in df.columns:
        print(
            "query scales: "
            + str(sorted(float(v) for v in df["query_scale"].dropna().unique()))
        )

    print(f"bases: {sorted(str(v) for v in df['basis'].dropna().unique())}")
    print(f"dtypes: {sorted(str(v) for v in df['dtype'].dropna().unique())}")
    print(f"detail dimension: {detail_dimension}")
    print(f"detail scenario: {detail_scenario}")
    print(f"detail query scale: {detail_query_scale}")
    print()


def print_metric_table(
    df: pd.DataFrame,
    *,
    metric: str,
    title: str,
) -> None:
    if metric not in df.columns:
        return

    grouped = aggregate_metric(
        df,
        metric=metric,
        group_columns=["scenario", "basis", "dtype"],
    )
    if grouped.empty:
        return

    print(title)
    print("=" * len(title))
    display_columns = [
        column
        for column in (
            "scenario",
            "basis",
            "dtype",
            "median",
            "q1",
            "q3",
            "count",
        )
        if column in grouped.columns
    ]

    with pd.option_context(
        "display.max_rows",
        500,
        "display.max_columns",
        20,
        "display.width",
        180,
        "display.float_format",
        lambda value: f"{value:.4e}",
    ):
        print(grouped[display_columns].to_string(index=False))
    print()


def load_optional_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"Could not read summary JSON {path}: {exc}")
        return None


def build_plot_summary(
    df: pd.DataFrame,
    *,
    config: PlotConfig,
    resolved_columns: Mapping[str, str],
    detail_dimension: int | None,
    detail_scenario: str | None,
    detail_query_scale: float | None,
    written_files: Sequence[Path],
    benchmark_summary: Any,
) -> dict[str, Any]:
    metric_names = (
        "partition_error",
        "query_error",
        "query_relative_error",
        "condition_number",
        "rank_fraction",
        "perturbation_sensitivity",
        "noise_amplification",
        "conversion_residual",
        "runtime_ms",
    )

    aggregate_statistics: dict[str, Any] = {}
    for metric in metric_names:
        if metric not in df.columns:
            continue
        grouped = aggregate_metric(
            df,
            metric=metric,
            group_columns=["scenario", "basis", "dtype"],
        )
        aggregate_statistics[metric] = aggregate_to_records(grouped)

    summary: dict[str, Any] = {
        "results_path": str(config.results_path),
        "input_summary_path": (
            str(config.summary_path)
            if config.summary_path is not None
            else None
        ),
        "output_dir": str(config.output_dir),
        "formats": list(config.formats),
        "record_count": int(len(df)),
        "finite_record_count": int(finite_record_mask(df).sum()),
        "resolved_input_columns": dict(resolved_columns),
        "detail_selection": {
            "dimension": detail_dimension,
            "scenario": detail_scenario,
            "query_scale": detail_query_scale,
        },
        "available_values": {
            "bases": sorted(str(v) for v in df["basis"].dropna().unique()),
            "dtypes": sorted(str(v) for v in df["dtype"].dropna().unique()),
            "dimensions": (
                sorted(int(v) for v in df["dimension"].dropna().unique())
                if "dimension" in df.columns
                else []
            ),
            "source_degrees": (
                sorted(int(v) for v in df["source_degree"].dropna().unique())
                if "source_degree" in df.columns
                else []
            ),
            "density_degrees": (
                sorted(int(v) for v in df["density_degree"].dropna().unique())
                if "density_degree" in df.columns
                else []
            ),
            "scenarios": (
                sorted(str(v) for v in df["scenario"].dropna().unique())
                if "scenario" in df.columns
                else []
            ),
            "query_scales": (
                sorted(float(v) for v in df["query_scale"].dropna().unique())
                if "query_scale" in df.columns
                else []
            ),
        },
        "aggregate_statistics": aggregate_statistics,
        "generated_files": [str(path) for path in written_files],
        "benchmark_summary": benchmark_summary,
    }
    return summary


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Main plotting pipeline
# ---------------------------------------------------------------------------

def run(config: PlotConfig) -> int:
    ensure_file(config.results_path, description="Results CSV")

    raw = pd.read_csv(config.results_path)
    df, resolved_columns = canonicalise_dataframe(raw)
    validate_minimum_schema(df)

    df = filter_requested_values(
        df,
        bases=config.bases,
        dtypes=config.dtypes,
    )

    detail_dimension = choose_detail_dimension(
        df,
        config.detail_dimension,
    )
    detail_scenario = choose_detail_scenario(
        df,
        config.detail_scenario,
    )
    detail_query_scale = choose_detail_query_scale(
        df,
        config.detail_query_scale,
    )

    print_configuration(
        config=config,
        df=df,
        detail_dimension=detail_dimension,
        detail_scenario=detail_scenario,
        detail_query_scale=detail_query_scale,
    )

    print_metric_table(
        df,
        metric="partition_error",
        title="Median partition-function error by scenario, basis, and dtype",
    )
    print_metric_table(
        df,
        metric="query_error",
        title="Median query-probability error by scenario, basis, and dtype",
    )
    print_metric_table(
        df,
        metric="condition_number",
        title="Median condition number by scenario, basis, and dtype",
    )
    print_metric_table(
        df,
        metric="runtime_ms",
        title="Median runtime by scenario, basis, and dtype",
    )

    configure_matplotlib()
    written_files: list[Path] = []

    degree_detail = apply_detail_filters(
        df,
        dimension=detail_dimension,
        scenario=detail_scenario,
    )

    query_degree_detail = apply_detail_filters(
        df,
        dimension=detail_dimension,
        scenario=detail_scenario,
        query_scale=detail_query_scale,
    )

    degree_suffix_parts: list[str] = []
    if detail_dimension is not None:
        degree_suffix_parts.append(f"dimension {detail_dimension}")
    if detail_scenario is not None:
        degree_suffix_parts.append(display_label(detail_scenario))
    degree_suffix = ", ".join(degree_suffix_parts) or None

    query_suffix_parts = list(degree_suffix_parts)
    if detail_query_scale is not None:
        query_suffix_parts.append(f"query scale {detail_query_scale:g}")
    query_suffix = ", ".join(query_suffix_parts) or None

    degree_specs = [
        (
            MetricSpec(
                "partition_error",
                "Partition-function relative error versus polynomial degree",
                "Relative partition-function error",
                log_y=True,
                positive_floor=1e-18,
            ),
            degree_detail,
            "pal_partition_function_error_vs_degree",
            degree_suffix,
        ),
        (
            MetricSpec(
                "query_error",
                "Query-probability absolute error versus polynomial degree",
                "Absolute query-probability error",
                log_y=True,
                positive_floor=1e-18,
            ),
            query_degree_detail,
            "pal_query_probability_error_vs_degree",
            query_suffix,
        ),
        (
            MetricSpec(
                "condition_number",
                "Basis condition number versus polynomial degree",
                "Estimated condition number",
                log_y=True,
                positive_floor=1.0,
            ),
            degree_detail,
            "pal_condition_number_vs_degree",
            degree_suffix,
        ),
        (
            MetricSpec(
                "rank_fraction",
                "Retained numerical-rank fraction versus polynomial degree",
                "Numerical-rank fraction",
                log_y=False,
            ),
            degree_detail,
            "pal_rank_fraction_vs_degree",
            degree_suffix,
        ),
        (
            MetricSpec(
                "perturbation_sensitivity",
                "Perturbation sensitivity versus polynomial degree",
                "Perturbation sensitivity",
                log_y=True,
                positive_floor=1e-12,
            ),
            degree_detail,
            "pal_perturbation_sensitivity_vs_degree",
            degree_suffix,
        ),
        (
            MetricSpec(
                "noise_amplification",
                "Coefficient-noise amplification versus polynomial degree",
                "Coefficient-noise amplification",
                log_y=True,
                positive_floor=1e-12,
            ),
            degree_detail,
            "pal_coefficient_noise_amplification_vs_degree",
            degree_suffix,
        ),
        (
            MetricSpec(
                "conversion_residual",
                "Basis-conversion residual versus polynomial degree",
                "Relative conversion residual",
                log_y=True,
                positive_floor=1e-18,
            ),
            degree_detail,
            "pal_basis_conversion_residual_vs_degree",
            degree_suffix,
        ),
        (
            MetricSpec(
                "runtime_ms",
                "Representative PAL runtime versus polynomial degree",
                "Total evaluation time (ms)",
                log_y=True,
                positive_floor=1e-6,
            ),
            degree_detail,
            "pal_runtime_vs_degree",
            degree_suffix,
        ),
    ]

    for spec, data, stem, suffix in degree_specs:
        written_files.extend(
            plot_metric_vs_x(
                data,
                metric_spec=spec,
                x_column="density_degree",
                x_label="Density polynomial degree",
                stem=stem,
                output_dir=config.output_dir,
                formats=config.formats,
                dpi=config.dpi,
                requested_bases=config.bases,
                requested_dtypes=config.dtypes,
                title_suffix=suffix,
            )
        )

    dimension_scenario_df = apply_detail_filters(
        df,
        dimension=None,
        scenario=detail_scenario,
    )

    dimension_specs = [
        MetricSpec(
            "partition_error",
            "Partition-function relative error versus dimension",
            "Relative partition-function error",
            log_y=True,
            positive_floor=1e-18,
        ),
        MetricSpec(
            "query_error",
            "Query-probability absolute error versus dimension",
            "Absolute query-probability error",
            log_y=True,
            positive_floor=1e-18,
        ),
        MetricSpec(
            "runtime_ms",
            "Representative PAL runtime versus dimension",
            "Total evaluation time (ms)",
            log_y=True,
            positive_floor=1e-6,
        ),
    ]

    # For dimension plots, use the largest common density degree unless there is
    # only one degree. This avoids mixing degree scaling into dimension scaling.
    dimension_detail = dimension_scenario_df.copy()
    if (
        "density_degree" in dimension_detail.columns
        and "dimension" in dimension_detail.columns
    ):
        degree_sets = []
        for _, group in dimension_detail.groupby("dimension"):
            degree_sets.append(
                set(int(v) for v in group["density_degree"].dropna().unique())
            )
        common_degrees = set.intersection(*degree_sets) if degree_sets else set()
        if common_degrees:
            selected_dimension_degree = max(common_degrees)
            dimension_detail = dimension_detail[
                dimension_detail["density_degree"] == selected_dimension_degree
            ]
        else:
            selected_dimension_degree = None
    else:
        selected_dimension_degree = None

    dimension_suffix_parts: list[str] = []
    if detail_scenario is not None:
        dimension_suffix_parts.append(display_label(detail_scenario))
    if selected_dimension_degree is not None:
        dimension_suffix_parts.append(
            f"density degree {selected_dimension_degree}"
        )
    dimension_suffix = ", ".join(dimension_suffix_parts) or None

    for spec in dimension_specs:
        data = dimension_detail
        if spec.canonical_name == "query_error":
            data = apply_detail_filters(
                data,
                dimension=None,
                scenario=None,
                query_scale=detail_query_scale,
            )

        written_files.extend(
            plot_metric_vs_x(
                data,
                metric_spec=spec,
                x_column="dimension",
                x_label="Dimension",
                stem={
                    "partition_error": "pal_partition_function_error_vs_dimension",
                    "query_error": "pal_query_probability_error_vs_dimension",
                    "runtime_ms": "pal_runtime_vs_dimension",
                }[spec.canonical_name],
                output_dir=config.output_dir,
                formats=config.formats,
                dpi=config.dpi,
                requested_bases=config.bases,
                requested_dtypes=config.dtypes,
                title_suffix=dimension_suffix,
            )
        )

    written_files.extend(
        plot_metric_by_scenario(
            df,
            metric_spec=MetricSpec(
                "partition_error",
                "Median partition-function error by PAL scenario",
                "Relative partition-function error",
                log_y=True,
                positive_floor=1e-18,
            ),
            stem="pal_error_by_scenario",
            output_dir=config.output_dir,
            formats=config.formats,
            dpi=config.dpi,
            requested_bases=config.bases,
            requested_dtypes=config.dtypes,
        )
    )

    written_files.extend(
        plot_metric_by_scenario(
            df,
            metric_spec=MetricSpec(
                "runtime_ms",
                "Median representative PAL runtime by scenario",
                "Total evaluation time (ms)",
                log_y=True,
                positive_floor=1e-6,
            ),
            stem="pal_runtime_by_scenario",
            output_dir=config.output_dir,
            formats=config.formats,
            dpi=config.dpi,
            requested_bases=config.bases,
            requested_dtypes=config.dtypes,
        )
    )

    written_files.extend(
        plot_error_summary(
            df,
            output_dir=config.output_dir,
            formats=config.formats,
            dpi=config.dpi,
            requested_bases=config.bases,
            requested_dtypes=config.dtypes,
        )
    )

    benchmark_summary = load_optional_json(config.summary_path)
    summary = build_plot_summary(
        df,
        config=config,
        resolved_columns=resolved_columns,
        detail_dimension=detail_dimension,
        detail_scenario=detail_scenario,
        detail_query_scale=detail_query_scale,
        written_files=written_files,
        benchmark_summary=benchmark_summary,
    )
    write_json(config.plot_summary_path, summary)

    print()
    print(f"Generated {len(written_files)} figure files.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    config = PlotConfig(
        results_path=args.results_path,
        summary_path=args.summary_path,
        output_dir=args.output_dir,
        plot_summary_path=args.plot_summary_path,
        formats=tuple(args.formats),
        dpi=int(args.dpi),
        detail_dimension=args.detail_dimension,
        detail_scenario=args.detail_scenario,
        detail_query_scale=args.detail_query_scale,
        bases=tuple(str(value).lower() for value in args.bases),
        dtypes=tuple(str(value).lower() for value in args.dtypes),
    )

    try:
        return run(config)
    except (FileNotFoundError, ValueError, KeyError, pd.errors.ParserError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
