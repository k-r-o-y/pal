#!/usr/bin/env python3
"""
Plot constrained-polytope runtime as a function of dimension.

This script is designed for the schema written by:

    analysis/polytope/run_polytope_benchmark.py

Expected input:
    results/polytope/polytope_results.csv

Default output:
    figures/polytope_runtime_vs_dimension.pdf
    figures/polytope_runtime_vs_dimension.png
    figures/polytope_runtime_vs_dimension.svg

Example:
    python -m analysis.plots.plot_polytope_runtime_vs_dimension \
        --input results/polytope/polytope_results.csv \
        --scenario dynamic_box_with_obstacle \
        --schedule oscillating \
        --degree 10 \
        --format pdf png svg

By default, --degree refers to density_polynomial_degree. To select using the
source polynomial degree instead, use:

    --degree-column source_polynomial_degree
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


# =============================================================================
# Defaults and display configuration
# =============================================================================

DEFAULT_INPUT_PATH = Path("results/polytope/polytope_results.csv")
DEFAULT_OUTPUT_DIRECTORY = Path("figures")
DEFAULT_SUMMARY_PATH = Path(
    "results/polytope/polytope_runtime_vs_dimension_summary.json"
)
DEFAULT_OUTPUT_STEM = "polytope_runtime_vs_dimension"
DEFAULT_FORMATS = ("pdf", "png", "svg")

SUPPORTED_FORMATS = ("pdf", "png", "svg")

BASIS_ORDER = ("monomial", "legendre", "chebyshev")
DTYPE_ORDER = ("float32", "float64")

SCENARIO_ORDER = (
    "convex_box",
    "box_with_obstacle",
    "dynamic_box_with_obstacle",
)

SCHEDULE_ORDER = (
    "static",
    "shrinking",
    "expanding",
    "oscillating",
    "pulsed",
    "random_walk",
)

BASIS_DISPLAY = {
    "monomial": "Monomial",
    "legendre": "Legendre",
    "chebyshev": "Chebyshev",
}

DTYPE_DISPLAY = {
    "float32": "float32",
    "float64": "float64",
}

SCENARIO_DISPLAY = {
    "convex_box": "Convex box",
    "box_with_obstacle": "Box with obstacle",
    "dynamic_box_with_obstacle": "Dynamic box with obstacle",
}

SCHEDULE_DISPLAY = {
    "static": "Static",
    "shrinking": "Shrinking",
    "expanding": "Expanding",
    "oscillating": "Oscillating",
    "pulsed": "Pulsed",
    "random_walk": "Random walk",
}

MARKERS = {
    "monomial": "o",
    "legendre": "s",
    "chebyshev": "^",
}

LINESTYLES = {
    "float32": "-",
    "float64": "--",
}


# These are the exact columns produced by the current benchmark runner.
REQUIRED_COLUMNS = {
    "scenario",
    "schedule",
    "dimension",
    "source_polynomial_degree",
    "density_polynomial_degree",
    "trial",
    "basis",
    "dtype",
    "total_evaluation_ms",
}

OPTIONAL_NUMERIC_COLUMNS = {
    "trajectory_step",
    "trajectory_steps",
    "basis_count",
    "recovery_point_count",
    "integration_ms",
    "conditioning_ms",
    "recovery_ms",
    "relative_integration_error",
    "perturbation_sensitivity",
    "basis_condition_number",
    "numerical_rank",
    "numerical_rank_fraction",
    "coefficient_noise_amplification",
    "coefficient_recovery_relative_error",
    "recovered_integral_relative_error",
    "obstacle_half_width",
}


# =============================================================================
# Data classes
# =============================================================================

@dataclass(frozen=True)
class PlotConfiguration:
    input_path: Path
    output_directory: Path
    summary_path: Path
    output_stem: str
    formats: tuple[str, ...]
    scenario: str
    schedule: str | None
    degree: int
    degree_column: str
    runtime_column: str
    bases: tuple[str, ...]
    dtypes: tuple[str, ...]
    dimensions: tuple[int, ...]
    dpi: int
    figure_width: float
    figure_height: float
    legend_columns: int
    log_y: bool
    include_iqr: bool
    include_runtime_components: bool
    quiet: bool


# =============================================================================
# Argument parsing
# =============================================================================

def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot constrained-polytope runtime against problem dimension "
            "using the current polytope_results.csv schema."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "--input-path",
        "--results-path",
        dest="input_path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to polytope_results.csv.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory in which figure files are written.",
    )

    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path for the JSON plot summary.",
    )

    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Filename stem used for generated figures.",
    )

    parser.add_argument(
        "--format",
        dest="formats",
        nargs="+",
        choices=SUPPORTED_FORMATS,
        default=list(DEFAULT_FORMATS),
        help="One or more output formats.",
    )

    parser.add_argument(
        "--scenario",
        choices=SCENARIO_ORDER,
        default="dynamic_box_with_obstacle",
        help="Polytope scenario to plot.",
    )

    parser.add_argument(
        "--schedule",
        choices=SCHEDULE_ORDER,
        default="oscillating",
        help=(
            "Dynamic obstacle schedule. For static scenarios, the script "
            "automatically uses the static schedule."
        ),
    )

    parser.add_argument(
        "--degree",
        type=int,
        default=10,
        help=(
            "Polynomial degree selected for the plot. By default this is "
            "matched against density_polynomial_degree."
        ),
    )

    parser.add_argument(
        "--degree-column",
        choices=(
            "density_polynomial_degree",
            "source_polynomial_degree",
        ),
        default="density_polynomial_degree",
        help="Column against which --degree is matched.",
    )

    parser.add_argument(
        "--runtime-column",
        choices=(
            "total_evaluation_ms",
            "integration_ms",
            "conditioning_ms",
            "recovery_ms",
        ),
        default="total_evaluation_ms",
        help="Runtime measurement used on the vertical axis.",
    )

    parser.add_argument(
        "--bases",
        nargs="+",
        choices=BASIS_ORDER,
        default=list(BASIS_ORDER),
        help="Basis representations to include.",
    )

    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=DTYPE_ORDER,
        default=list(DTYPE_ORDER),
        help="Arithmetic precisions to include.",
    )

    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional dimensions to include. When omitted, all available "
            "dimensions satisfying the filters are used."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution used for PNG output.",
    )

    parser.add_argument(
        "--figure-width",
        type=float,
        default=8.3,
        help="Figure width in inches.",
    )

    parser.add_argument(
        "--figure-height",
        type=float,
        default=5.4,
        help="Figure height in inches.",
    )

    parser.add_argument(
        "--legend-columns",
        type=int,
        default=3,
        help="Maximum number of legend columns.",
    )

    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use a linear runtime axis instead of a logarithmic axis.",
    )

    parser.add_argument(
        "--no-iqr",
        action="store_true",
        help="Disable the interquartile-range shading.",
    )

    parser.add_argument(
        "--include-runtime-components",
        action="store_true",
        help=(
            "Also generate separate dimension plots for integration_ms, "
            "conditioning_ms, and recovery_ms when those columns are present."
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the aggregate runtime table.",
    )

    args = parser.parse_args(argv)

    if args.degree < 0:
        parser.error("--degree must be non-negative")

    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    if args.figure_width <= 0:
        parser.error("--figure-width must be positive")

    if args.figure_height <= 0:
        parser.error("--figure-height must be positive")

    if args.legend_columns <= 0:
        parser.error("--legend-columns must be positive")

    if args.dimensions is not None and any(
        dimension <= 0 for dimension in args.dimensions
    ):
        parser.error("--dimensions values must be positive")

    args.formats = tuple(dict.fromkeys(args.formats))
    args.bases = tuple(dict.fromkeys(args.bases))
    args.dtypes = tuple(dict.fromkeys(args.dtypes))

    if args.dimensions is not None:
        args.dimensions = tuple(sorted(set(args.dimensions)))

    return args


# =============================================================================
# Data loading and validation
# =============================================================================

def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results file does not exist: {path}")

    frame = pd.read_csv(path)

    if frame.empty:
        raise ValueError(f"Results file is empty: {path}")

    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            "The input file does not match the expected polytope benchmark "
            "schema. Missing columns: "
            + ", ".join(missing)
            + "\nAvailable columns: "
            + ", ".join(str(column) for column in frame.columns)
        )

    frame = frame.copy()

    string_columns = ("scenario", "schedule", "basis", "dtype")
    for column in string_columns:
        frame[column] = (
            frame[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

    numeric_columns = {
        "dimension",
        "source_polynomial_degree",
        "density_polynomial_degree",
        "trial",
        "total_evaluation_ms",
    }
    numeric_columns |= OPTIONAL_NUMERIC_COLUMNS

    for column in sorted(numeric_columns & set(frame.columns)):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["is_finite_total_runtime"] = np.isfinite(
        frame["total_evaluation_ms"].to_numpy(dtype=float)
    )

    return frame


def ordered_present(
    values: Iterable[str],
    preferred_order: Sequence[str],
) -> list[str]:
    present = list(dict.fromkeys(str(value) for value in values))

    ordered = [
        value
        for value in preferred_order
        if value in present
    ]

    ordered.extend(
        value
        for value in present
        if value not in ordered
    )

    return ordered


def available_integer_values(
    frame: pd.DataFrame,
    column: str,
) -> list[int]:
    if column not in frame.columns:
        return []

    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[np.isfinite(values)]

    return sorted(set(int(value) for value in values))


def validate_requested_values(
    frame: pd.DataFrame,
    *,
    scenario: str,
    schedule: str | None,
    degree: int,
    degree_column: str,
    bases: Sequence[str],
    dtypes: Sequence[str],
) -> None:
    scenarios = ordered_present(frame["scenario"].unique(), SCENARIO_ORDER)

    if scenario not in scenarios:
        raise ValueError(
            f"Scenario {scenario!r} is unavailable. "
            f"Available scenarios: {scenarios}"
        )

    scenario_rows = frame[frame["scenario"] == scenario]

    if schedule is not None:
        schedules = ordered_present(
            scenario_rows["schedule"].unique(),
            SCHEDULE_ORDER,
        )

        if schedule not in schedules:
            raise ValueError(
                f"Schedule {schedule!r} is unavailable for scenario "
                f"{scenario!r}. Available schedules: {schedules}"
            )

    degree_values = available_integer_values(
        scenario_rows,
        degree_column,
    )

    if degree not in degree_values:
        raise ValueError(
            f"Degree {degree} is unavailable in column "
            f"{degree_column!r} for scenario {scenario!r}. "
            f"Available values: {degree_values}"
        )

    available_bases = ordered_present(
        scenario_rows["basis"].unique(),
        BASIS_ORDER,
    )
    missing_bases = [
        basis for basis in bases
        if basis not in available_bases
    ]

    if missing_bases:
        raise ValueError(
            "Unavailable bases requested: "
            + ", ".join(missing_bases)
            + f". Available bases: {available_bases}"
        )

    available_dtypes = ordered_present(
        scenario_rows["dtype"].unique(),
        DTYPE_ORDER,
    )
    missing_dtypes = [
        dtype for dtype in dtypes
        if dtype not in available_dtypes
    ]

    if missing_dtypes:
        raise ValueError(
            "Unavailable dtypes requested: "
            + ", ".join(missing_dtypes)
            + f". Available dtypes: {available_dtypes}"
        )


# =============================================================================
# Filtering and aggregation
# =============================================================================

def resolve_schedule(
    frame: pd.DataFrame,
    scenario: str,
    requested_schedule: str,
) -> str | None:
    if scenario != "dynamic_box_with_obstacle":
        scenario_rows = frame[frame["scenario"] == scenario]
        schedules = set(scenario_rows["schedule"].dropna().astype(str))

        if "static" in schedules:
            return "static"

        if schedules:
            return sorted(schedules)[0]

        return None

    return requested_schedule


def filter_results(
    frame: pd.DataFrame,
    configuration: PlotConfiguration,
) -> pd.DataFrame:
    filtered = frame[
        frame["scenario"].eq(configuration.scenario)
    ].copy()

    if configuration.schedule is not None:
        filtered = filtered[
            filtered["schedule"].eq(configuration.schedule)
        ]

    filtered = filtered[
        filtered[configuration.degree_column].eq(
            configuration.degree
        )
    ]

    filtered = filtered[
        filtered["basis"].isin(configuration.bases)
    ]

    filtered = filtered[
        filtered["dtype"].isin(configuration.dtypes)
    ]

    if configuration.dimensions:
        filtered = filtered[
            filtered["dimension"].isin(configuration.dimensions)
        ]

    runtime_values = pd.to_numeric(
        filtered[configuration.runtime_column],
        errors="coerce",
    )

    finite_mask = np.isfinite(runtime_values.to_numpy(dtype=float))
    positive_mask = runtime_values.to_numpy(dtype=float) > 0

    invalid_count = int(np.count_nonzero(~finite_mask))
    nonpositive_count = int(
        np.count_nonzero(finite_mask & ~positive_mask)
    )

    if invalid_count:
        warnings.warn(
            f"Omitting {invalid_count} non-finite "
            f"{configuration.runtime_column} values.",
            RuntimeWarning,
        )

    if configuration.log_y and nonpositive_count:
        warnings.warn(
            f"Omitting {nonpositive_count} non-positive "
            f"{configuration.runtime_column} values from the log plot.",
            RuntimeWarning,
        )

    filtered = filtered[finite_mask].copy()

    if configuration.log_y:
        filtered = filtered[
            filtered[configuration.runtime_column] > 0
        ]

    if filtered.empty:
        raise ValueError(
            "No usable records remain after applying the selected filters."
        )

    return filtered.reset_index(drop=True)


def aggregate_runtime(
    frame: pd.DataFrame,
    runtime_column: str,
) -> pd.DataFrame:
    groups = [
        "dimension",
        "basis",
        "dtype",
    ]

    grouped = (
        frame.groupby(
            groups,
            observed=True,
            dropna=False,
        )[runtime_column]
        .agg(
            median="median",
            q1=lambda series: series.quantile(0.25),
            q3=lambda series: series.quantile(0.75),
            mean="mean",
            standard_deviation="std",
            minimum="min",
            maximum="max",
            count="count",
        )
        .reset_index()
        .sort_values(
            ["basis", "dtype", "dimension"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return grouped


# =============================================================================
# Figure helpers
# =============================================================================

def display_name(
    mapping: Mapping[str, str],
    value: str,
) -> str:
    return mapping.get(
        value,
        value.replace("_", " ").title(),
    )


def legend_label(
    basis: str,
    dtype: str,
) -> str:
    return (
        f"{display_name(BASIS_DISPLAY, basis)} "
        f"({display_name(DTYPE_DISPLAY, dtype)})"
    )


def runtime_display_name(runtime_column: str) -> str:
    names = {
        "total_evaluation_ms": "Total evaluation runtime (ms)",
        "integration_ms": "Integration runtime (ms)",
        "conditioning_ms": "Conditioning runtime (ms)",
        "recovery_ms": "Coefficient-recovery runtime (ms)",
    }

    return names.get(
        runtime_column,
        runtime_column.replace("_", " ").title(),
    )


def runtime_title_fragment(runtime_column: str) -> str:
    names = {
        "total_evaluation_ms": "Total evaluation runtime",
        "integration_ms": "Integration runtime",
        "conditioning_ms": "Conditioning runtime",
        "recovery_ms": "Coefficient-recovery runtime",
    }

    return names.get(
        runtime_column,
        runtime_column.replace("_", " ").title(),
    )


def style_for_series(
    basis: str,
    dtype: str,
) -> dict[str, Any]:
    return {
        "marker": MARKERS.get(basis, "o"),
        "linestyle": LINESTYLES.get(dtype, "-"),
        "linewidth": 1.8,
        "markersize": 5.5,
    }


def set_integer_dimension_ticks(
    axis: plt.Axes,
    dimensions: Sequence[int],
) -> None:
    axis.set_xticks(list(dimensions))
    axis.set_xticklabels(
        [str(dimension) for dimension in dimensions]
    )


def add_legend_below(
    axis: plt.Axes,
    maximum_columns: int,
) -> None:
    handles, labels = axis.get_legend_handles_labels()

    if not handles:
        return

    axis.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(maximum_columns, len(handles)),
        frameon=False,
    )


def add_footer(
    figure: plt.Figure,
    text: str,
) -> None:
    figure.text(
        0.5,
        0.006,
        text,
        ha="center",
        va="bottom",
        fontsize=8,
    )


def save_figure(
    figure: plt.Figure,
    *,
    output_directory: Path,
    output_stem: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    written: list[Path] = []

    for file_format in formats:
        output_path = (
            output_directory
            / f"{output_stem}.{file_format}"
        )

        save_arguments: dict[str, Any] = {
            "bbox_inches": "tight",
            "pad_inches": 0.04,
        }

        if file_format == "png":
            save_arguments["dpi"] = dpi

        figure.savefig(
            output_path,
            **save_arguments,
        )

        written.append(output_path)
        print(f"Wrote {output_path}")

    plt.close(figure)
    return written


def plot_runtime_vs_dimension(
    grouped: pd.DataFrame,
    configuration: PlotConfiguration,
    *,
    runtime_column: str,
    output_stem: str,
) -> list[Path]:
    dimensions = sorted(
        int(value)
        for value in grouped["dimension"].dropna().unique()
    )

    if not dimensions:
        raise ValueError(
            "No dimensions are available for plotting."
        )

    figure, axis = plt.subplots(
        figsize=(
            configuration.figure_width,
            configuration.figure_height,
        )
    )

    bases = ordered_present(
        grouped["basis"].unique(),
        BASIS_ORDER,
    )

    dtypes = ordered_present(
        grouped["dtype"].unique(),
        DTYPE_ORDER,
    )

    plotted_series = 0

    for basis in bases:
        for dtype in dtypes:
            subset = grouped[
                grouped["basis"].eq(basis)
                & grouped["dtype"].eq(dtype)
            ].copy()

            subset = subset.sort_values(
                "dimension",
                kind="stable",
            )

            if subset.empty:
                continue

            x_values = subset[
                "dimension"
            ].to_numpy(dtype=float)

            median_values = subset[
                "median"
            ].to_numpy(dtype=float)

            q1_values = subset[
                "q1"
            ].to_numpy(dtype=float)

            q3_values = subset[
                "q3"
            ].to_numpy(dtype=float)

            line = axis.plot(
                x_values,
                median_values,
                label=legend_label(basis, dtype),
                **style_for_series(basis, dtype),
            )[0]

            if configuration.include_iqr:
                axis.fill_between(
                    x_values,
                    q1_values,
                    q3_values,
                    color=line.get_color(),
                    alpha=0.12,
                    linewidth=0,
                )

            plotted_series += 1

    if plotted_series == 0:
        plt.close(figure)
        raise ValueError(
            "No basis/dtype series were available to plot."
        )

    if configuration.log_y:
        axis.set_yscale("log")
        axis.grid(
            True,
            which="minor",
            alpha=0.12,
        )

    axis.grid(
        True,
        which="major",
        alpha=0.25,
    )

    axis.set_xlabel("Polytope dimension")
    axis.set_ylabel(runtime_display_name(runtime_column))

    axis.set_title(
        f"{runtime_title_fragment(runtime_column)} "
        "versus polytope dimension"
    )

    set_integer_dimension_ticks(
        axis,
        dimensions,
    )

    add_legend_below(
        axis,
        configuration.legend_columns,
    )

    scenario_text = display_name(
        SCENARIO_DISPLAY,
        configuration.scenario,
    )

    footer_parts = [
        scenario_text,
        (
            f"{configuration.degree_column.replace('_', ' ')} "
            f"{configuration.degree}"
        ),
        "median and IQR across trials and trajectory steps",
    ]

    if configuration.schedule is not None:
        footer_parts.insert(
            1,
            display_name(
                SCHEDULE_DISPLAY,
                configuration.schedule,
            ),
        )

    add_footer(
        figure,
        " | ".join(footer_parts),
    )

    figure.subplots_adjust(
        bottom=0.27,
    )

    return save_figure(
        figure,
        output_directory=configuration.output_directory,
        output_stem=output_stem,
        formats=configuration.formats,
        dpi=configuration.dpi,
    )


# =============================================================================
# Console and JSON summaries
# =============================================================================

def scientific(
    value: Any,
    digits: int = 4,
) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"

    if not math.isfinite(numeric):
        return "nan"

    if numeric == 0:
        return "0"

    return f"{numeric:.{digits}e}"


def print_configuration(
    frame: pd.DataFrame,
    configuration: PlotConfiguration,
) -> None:
    print(
        "\nPolytope runtime-versus-dimension plotting configuration"
    )
    print(
        "========================================================"
    )
    print(f"input path: {configuration.input_path}")
    print(
        f"output directory: {configuration.output_directory}"
    )
    print(f"formats: {list(configuration.formats)}")
    print(f"scenario: {configuration.scenario}")
    print(f"schedule: {configuration.schedule}")
    print(
        f"degree column: {configuration.degree_column}"
    )
    print(f"degree: {configuration.degree}")
    print(
        f"runtime column: {configuration.runtime_column}"
    )
    print(f"bases: {list(configuration.bases)}")
    print(f"dtypes: {list(configuration.dtypes)}")
    print(
        "dimensions: "
        f"{sorted(frame['dimension'].dropna().astype(int).unique())}"
    )
    print(f"filtered records: {len(frame)}")
    print(f"logarithmic y-axis: {configuration.log_y}")
    print(f"IQR shading: {configuration.include_iqr}")
    print()


def print_runtime_table(
    grouped: pd.DataFrame,
    runtime_column: str,
) -> None:
    heading = (
        f"Median {runtime_display_name(runtime_column).lower()} "
        "by dimension, basis, and dtype"
    )

    print(heading)
    print("=" * len(heading))

    display = grouped[
        [
            "dimension",
            "basis",
            "dtype",
            "median",
            "q1",
            "q3",
            "minimum",
            "maximum",
            "count",
        ]
    ].copy()

    for column in (
        "median",
        "q1",
        "q3",
        "minimum",
        "maximum",
    ):
        display[column] = display[column].map(scientific)

    print(
        display.to_string(index=False)
    )
    print()


def json_safe_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    if pd.isna(value):
        return None

    return value


def json_safe_records(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for record in frame.to_dict("records"):
        records.append(
            {
                key: json_safe_value(value)
                for key, value in record.items()
            }
        )

    return records


def write_summary(
    *,
    filtered: pd.DataFrame,
    grouped_metrics: Mapping[str, pd.DataFrame],
    configuration: PlotConfiguration,
    written_files: Sequence[Path],
) -> None:
    configuration.summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    runtime_summaries = {
        metric: json_safe_records(grouped)
        for metric, grouped in grouped_metrics.items()
    }

    maximum_records: dict[str, dict[str, Any] | None] = {}

    for runtime_column in grouped_metrics:
        usable = filtered[
            np.isfinite(
                pd.to_numeric(
                    filtered[runtime_column],
                    errors="coerce",
                ).to_numpy(dtype=float)
            )
        ]

        if usable.empty:
            maximum_records[runtime_column] = None
            continue

        row = usable.loc[
            usable[runtime_column].idxmax()
        ]

        maximum_records[runtime_column] = {
            key: json_safe_value(row[key])
            for key in (
                "scenario",
                "schedule",
                "trajectory_step",
                "dimension",
                "source_polynomial_degree",
                "density_polynomial_degree",
                "trial",
                "basis",
                "dtype",
                runtime_column,
            )
            if key in row.index
        }

    payload = {
        "input_path": str(configuration.input_path),
        "output_directory": str(
            configuration.output_directory
        ),
        "output_stem": configuration.output_stem,
        "scenario": configuration.scenario,
        "schedule": configuration.schedule,
        "degree": configuration.degree,
        "degree_column": configuration.degree_column,
        "primary_runtime_column": (
            configuration.runtime_column
        ),
        "bases": list(configuration.bases),
        "dtypes": list(configuration.dtypes),
        "dimensions": sorted(
            int(value)
            for value in filtered[
                "dimension"
            ].dropna().unique()
        ),
        "records": int(len(filtered)),
        "logarithmic_y_axis": configuration.log_y,
        "iqr_shading": configuration.include_iqr,
        "figures": [
            str(path)
            for path in written_files
        ],
        "runtime_summaries": runtime_summaries,
        "maximum_runtime_records": maximum_records,
    }

    configuration.summary_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Wrote {configuration.summary_path}"
    )


# =============================================================================
# Main program
# =============================================================================

def build_configuration(
    args: argparse.Namespace,
    frame: pd.DataFrame,
) -> PlotConfiguration:
    schedule = resolve_schedule(
        frame,
        args.scenario,
        args.schedule,
    )

    validate_requested_values(
        frame,
        scenario=args.scenario,
        schedule=schedule,
        degree=args.degree,
        degree_column=args.degree_column,
        bases=args.bases,
        dtypes=args.dtypes,
    )

    if args.runtime_column not in frame.columns:
        raise ValueError(
            f"Runtime column {args.runtime_column!r} is missing. "
            f"Available columns: {list(frame.columns)}"
        )

    available_dimensions = available_integer_values(
        frame,
        "dimension",
    )

    if args.dimensions is None:
        dimensions = tuple(available_dimensions)
    else:
        unavailable = [
            dimension
            for dimension in args.dimensions
            if dimension not in available_dimensions
        ]

        if unavailable:
            raise ValueError(
                "Unavailable dimensions requested: "
                + ", ".join(str(value) for value in unavailable)
                + f". Available dimensions: {available_dimensions}"
            )

        dimensions = tuple(args.dimensions)

    return PlotConfiguration(
        input_path=args.input_path,
        output_directory=args.output_directory,
        summary_path=args.summary_path,
        output_stem=args.output_stem,
        formats=args.formats,
        scenario=args.scenario,
        schedule=schedule,
        degree=args.degree,
        degree_column=args.degree_column,
        runtime_column=args.runtime_column,
        bases=args.bases,
        dtypes=args.dtypes,
        dimensions=dimensions,
        dpi=args.dpi,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        legend_columns=args.legend_columns,
        log_y=not args.linear_y,
        include_iqr=not args.no_iqr,
        include_runtime_components=(
            args.include_runtime_components
        ),
        quiet=args.quiet,
    )


def component_output_stem(
    base_stem: str,
    runtime_column: str,
) -> str:
    if runtime_column == "total_evaluation_ms":
        return base_stem

    suffix_mapping = {
        "integration_ms": "integration",
        "conditioning_ms": "conditioning",
        "recovery_ms": "recovery",
    }

    suffix = suffix_mapping.get(
        runtime_column,
        runtime_column.removesuffix("_ms"),
    )

    return f"{base_stem}_{suffix}"


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_arguments(argv)

    try:
        frame = load_results(args.input_path)

        configuration = build_configuration(
            args,
            frame,
        )

        filtered = filter_results(
            frame,
            configuration,
        )

        print_configuration(
            filtered,
            configuration,
        )

        runtime_columns = [
            configuration.runtime_column
        ]

        if configuration.include_runtime_components:
            for candidate in (
                "integration_ms",
                "conditioning_ms",
                "recovery_ms",
            ):
                if (
                    candidate in filtered.columns
                    and candidate not in runtime_columns
                ):
                    runtime_columns.append(candidate)

        written_files: list[Path] = []
        grouped_metrics: dict[str, pd.DataFrame] = {}

        for runtime_column in runtime_columns:
            usable = filtered.copy()

            numeric_runtime = pd.to_numeric(
                usable[runtime_column],
                errors="coerce",
            )

            finite_mask = np.isfinite(
                numeric_runtime.to_numpy(dtype=float)
            )

            usable = usable[finite_mask].copy()

            if configuration.log_y:
                usable = usable[
                    usable[runtime_column] > 0
                ]

            if usable.empty:
                warnings.warn(
                    f"Skipping {runtime_column}: no usable values.",
                    RuntimeWarning,
                )
                continue

            grouped = aggregate_runtime(
                usable,
                runtime_column,
            )

            grouped_metrics[runtime_column] = grouped

            if not configuration.quiet:
                print_runtime_table(
                    grouped,
                    runtime_column,
                )

            output_stem = component_output_stem(
                configuration.output_stem,
                runtime_column,
            )

            written_files.extend(
                plot_runtime_vs_dimension(
                    grouped,
                    configuration,
                    runtime_column=runtime_column,
                    output_stem=output_stem,
                )
            )

        if not written_files:
            warnings.warn(
                "No runtime figures were produced.",
                RuntimeWarning,
            )
            return 1

        write_summary(
            filtered=filtered,
            grouped_metrics=grouped_metrics,
            configuration=configuration,
            written_files=written_files,
        )

        print(
            f"\nGenerated {len(written_files)} figure files."
        )

        return 0

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        pd.errors.ParserError,
    ) as error:
        print(
            f"Error: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())