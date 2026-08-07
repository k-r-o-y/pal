"""
Plot downstream constrained-probability benchmark results.

Expected input
--------------
results/downstream/downstream_results.csv

Default outputs
---------------
figures/partition_function_error_vs_degree.pdf
figures/partition_function_error_vs_degree.png
figures/partition_function_error_vs_degree.svg

figures/query_probability_error_vs_degree.pdf
figures/query_probability_error_vs_degree.png
figures/query_probability_error_vs_degree.svg

figures/query_probability_error_vs_query_scale.pdf
figures/query_probability_error_vs_query_scale.png
figures/query_probability_error_vs_query_scale.svg

figures/query_probability_sensitivity_vs_degree.pdf
figures/query_probability_sensitivity_vs_degree.png
figures/query_probability_sensitivity_vs_degree.svg

figures/downstream_runtime_vs_degree.pdf
figures/downstream_runtime_vs_degree.png
figures/downstream_runtime_vs_degree.svg

figures/downstream_error_summary.pdf
figures/downstream_error_summary.png
figures/downstream_error_summary.svg

Run with
--------
python -m analysis.plots.plot_downstream_benchmark \
    --format pdf png svg
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULTS_PATH = Path(
    "results/downstream/downstream_results.csv"
)
DEFAULT_OUTPUT_DIRECTORY = Path("figures")

DEFAULT_FORMATS = ("pdf",)
SUPPORTED_FORMATS = ("pdf", "png", "svg")

DEFAULT_DETAIL_DIMENSION = 3
DEFAULT_QUERY_SCALE = 0.50
DEFAULT_ERROR_FLOOR = 1.0e-18
DEFAULT_SENSITIVITY_FLOOR = 1.0e-12
DEFAULT_RUNTIME_FLOOR_MS = 1.0e-6

BASIS_ORDER = ("monomial", "legendre", "chebyshev")
DTYPE_ORDER = ("float32", "float64")

BASIS_LABELS = {
    "monomial": "Monomial",
    "legendre": "Legendre",
    "chebyshev": "Chebyshev",
}

DTYPE_LABELS = {
    "float32": "float32",
    "float64": "float64",
}

BASIS_COLOURS = {
    "monomial": "tab:blue",
    "legendre": "tab:green",
    "chebyshev": "tab:purple",
}

FLOAT64_COLOURS = {
    "monomial": "tab:orange",
    "legendre": "tab:red",
    "chebyshev": "tab:brown",
}

BASIS_MARKERS = {
    "monomial": "s",
    "legendre": "o",
    "chebyshev": "s",
}

DTYPE_LINESTYLES = {
    "float32": "--",
    "float64": "-",
}

REQUIRED_COLUMNS = {
    "basis",
    "dtype",
    "dimension",
    "source_polynomial_degree",
    "density_polynomial_degree",
    "trial",
    "query_scale",
    "partition_function_relative_error",
    "query_probability_absolute_error",
    "query_probability_relative_error",
    "query_probability_perturbation_sensitivity",
    "total_downstream_evaluation_ms",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate figures and numerical summaries from the downstream "
            "constrained-probability benchmark."
        )
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=(
            "Path to downstream_results.csv. "
            f"Default: {DEFAULT_RESULTS_PATH}"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directory in which figures will be written. "
            f"Default: {DEFAULT_OUTPUT_DIRECTORY}"
        ),
    )
    parser.add_argument(
        "--format",
        nargs="+",
        choices=SUPPORTED_FORMATS,
        default=list(DEFAULT_FORMATS),
        help=(
            "One or more output formats. "
            "Supported values: pdf, png, svg."
        ),
    )
    parser.add_argument(
        "--detail-dimension",
        type=int,
        default=DEFAULT_DETAIL_DIMENSION,
        help=(
            "Dimension used for degree-dependent detail plots. "
            f"Default: {DEFAULT_DETAIL_DIMENSION}"
        ),
    )
    parser.add_argument(
        "--query-scale",
        type=float,
        default=DEFAULT_QUERY_SCALE,
        help=(
            "Query scale used for degree-dependent plots. "
            f"Default: {DEFAULT_QUERY_SCALE}"
        ),
    )
    parser.add_argument(
        "--error-floor",
        type=float,
        default=DEFAULT_ERROR_FLOOR,
        help=(
            "Positive display floor used for zero errors on logarithmic axes. "
            f"Default: {DEFAULT_ERROR_FLOOR:g}"
        ),
    )
    parser.add_argument(
        "--sensitivity-floor",
        type=float,
        default=DEFAULT_SENSITIVITY_FLOOR,
        help=(
            "Positive display floor for zero perturbation sensitivities. "
            f"Default: {DEFAULT_SENSITIVITY_FLOOR:g}"
        ),
    )
    parser.add_argument(
        "--runtime-floor-ms",
        type=float,
        default=DEFAULT_RUNTIME_FLOOR_MS,
        help=(
            "Positive display floor for zero runtimes on logarithmic axes. "
            f"Default: {DEFAULT_RUNTIME_FLOOR_MS:g}"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster-image resolution. Default: 300.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them.",
    )

    return parser.parse_args()


def validate_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive value")


def normalise_string_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    result["basis"] = (
        result["basis"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    result["dtype"] = (
        result["dtype"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return result


def ensure_numeric_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    result = frame.copy()

    for column in columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Downstream result file does not exist: {path}"
        )

    frame = pd.read_csv(path)

    missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        formatted = ", ".join(missing_columns)
        raise ValueError(
            "The downstream result file is missing required columns: "
            f"{formatted}"
        )

    frame = normalise_string_columns(frame)

    numeric_columns = [
        "dimension",
        "source_polynomial_degree",
        "density_polynomial_degree",
        "trial",
        "query_scale",
        "partition_function_relative_error",
        "query_probability_absolute_error",
        "query_probability_relative_error",
        "query_probability_perturbation_sensitivity",
        "total_downstream_evaluation_ms",
    ]

    frame = ensure_numeric_columns(frame, numeric_columns)

    invalid_basis = sorted(
        set(frame["basis"].dropna()).difference(BASIS_ORDER)
    )
    if invalid_basis:
        raise ValueError(
            "Unsupported basis names in result file: "
            + ", ".join(invalid_basis)
        )

    invalid_dtype = sorted(
        set(frame["dtype"].dropna()).difference(DTYPE_ORDER)
    )
    if invalid_dtype:
        raise ValueError(
            "Unsupported dtype names in result file: "
            + ", ".join(invalid_dtype)
        )

    non_finite_mask = ~np.isfinite(
        frame[numeric_columns].to_numpy(dtype=float)
    ).all(axis=1)

    if non_finite_mask.any():
        count = int(non_finite_mask.sum())
        warnings.warn(
            f"Dropping {count} rows containing non-finite numeric values.",
            RuntimeWarning,
            stacklevel=2,
        )
        frame = frame.loc[~non_finite_mask].copy()

    if frame.empty:
        raise ValueError(
            "No finite downstream benchmark rows remain after validation."
        )

    frame["dimension"] = frame["dimension"].astype(int)
    frame["source_polynomial_degree"] = (
        frame["source_polynomial_degree"].astype(int)
    )
    frame["density_polynomial_degree"] = (
        frame["density_polynomial_degree"].astype(int)
    )
    frame["trial"] = frame["trial"].astype(int)

    return frame


def nearest_available_value(
    values: Sequence[float],
    requested: float,
) -> float:
    if not values:
        raise ValueError("No candidate values are available")

    values_array = np.asarray(values, dtype=float)
    index = int(np.argmin(np.abs(values_array - requested)))
    return float(values_array[index])


def determine_detail_dimension(
    frame: pd.DataFrame,
    requested: int,
) -> int:
    dimensions = sorted(frame["dimension"].unique().tolist())

    if requested in dimensions:
        return requested

    nearest = min(
        dimensions,
        key=lambda value: abs(value - requested),
    )

    warnings.warn(
        f"Requested detail dimension {requested} is unavailable; "
        f"using dimension {nearest}.",
        RuntimeWarning,
        stacklevel=2,
    )

    return int(nearest)


def determine_query_scale(
    frame: pd.DataFrame,
    requested: float,
) -> float:
    scales = sorted(frame["query_scale"].unique().tolist())

    for scale in scales:
        if np.isclose(scale, requested, rtol=0.0, atol=1.0e-12):
            return float(scale)

    nearest = nearest_available_value(scales, requested)

    warnings.warn(
        f"Requested query scale {requested:g} is unavailable; "
        f"using query scale {nearest:g}.",
        RuntimeWarning,
        stacklevel=2,
    )

    return nearest


def quantile_summary(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    value_column: str,
) -> pd.DataFrame:
    grouped = (
        frame.groupby(
            list(group_columns),
            observed=True,
            sort=True,
        )[value_column]
        .agg(
            median="median",
            q1=lambda series: series.quantile(0.25),
            q3=lambda series: series.quantile(0.75),
            minimum="min",
            maximum="max",
            count="count",
        )
        .reset_index()
    )

    return grouped


def positive_plot_values(
    values: np.ndarray | pd.Series,
    floor: float,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)

    return np.where(
        np.isfinite(array) & (array > floor),
        array,
        floor,
    )


def line_colour(basis: str, dtype: str) -> str:
    if dtype == "float64":
        return FLOAT64_COLOURS[basis]

    return BASIS_COLOURS[basis]


def line_label(basis: str, dtype: str) -> str:
    return (
        f"{BASIS_LABELS[basis]}, "
        f"{DTYPE_LABELS[dtype]}"
    )


def configure_axis(
    axis: plt.Axes,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    logarithmic_y: bool = True,
) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)

    if logarithmic_y:
        axis.set_yscale("log")

    axis.grid(
        True,
        which="both",
        alpha=0.25,
        linewidth=0.8,
    )


def plot_grouped_metric(
    axis: plt.Axes,
    summary: pd.DataFrame,
    *,
    x_column: str,
    floor: float,
) -> None:
    for basis in BASIS_ORDER:
        for dtype in DTYPE_ORDER:
            subset = summary.loc[
                (summary["basis"] == basis)
                & (summary["dtype"] == dtype)
            ].sort_values(x_column)

            if subset.empty:
                continue

            x = subset[x_column].to_numpy(dtype=float)
            median = positive_plot_values(
                subset["median"],
                floor,
            )
            q1 = positive_plot_values(
                subset["q1"],
                floor,
            )
            q3 = positive_plot_values(
                subset["q3"],
                floor,
            )

            colour = line_colour(basis, dtype)

            axis.plot(
                x,
                median,
                linestyle=DTYPE_LINESTYLES[dtype],
                marker=BASIS_MARKERS[basis],
                linewidth=1.8,
                markersize=5.0,
                color=colour,
                label=line_label(basis, dtype),
            )

            axis.fill_between(
                x,
                q1,
                q3,
                color=colour,
                alpha=0.12,
                linewidth=0.0,
            )


def save_figure(
    figure: plt.Figure,
    output_directory: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    for file_format in formats:
        output_path = output_directory / f"{stem}.{file_format}"

        save_kwargs: dict[str, object] = {
            "bbox_inches": "tight",
        }

        if file_format == "png":
            save_kwargs["dpi"] = dpi

        figure.savefig(
            output_path,
            format=file_format,
            **save_kwargs,
        )

        print(f"Wrote {output_path}")


def add_external_legend(
    figure: plt.Figure,
    axis: plt.Axes,
    *,
    columns: int = 3,
) -> None:
    handles, labels = axis.get_legend_handles_labels()

    if not handles:
        return

    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=columns,
        frameon=True,
        title="Basis and arithmetic precision",
        bbox_to_anchor=(0.5, -0.01),
    )

    figure.subplots_adjust(bottom=0.24)


def plot_partition_function_error_vs_degree(
    frame: pd.DataFrame,
    *,
    dimension: int,
    query_scale: float,
    error_floor: float,
    output_directory: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    subset = frame.loc[
        (frame["dimension"] == dimension)
        & np.isclose(
            frame["query_scale"],
            query_scale,
            rtol=0.0,
            atol=1.0e-12,
        )
    ].copy()

    summary = quantile_summary(
        subset,
        [
            "basis",
            "dtype",
            "source_polynomial_degree",
            "density_polynomial_degree",
        ],
        "partition_function_relative_error",
    )

    figure, axis = plt.subplots(figsize=(9.0, 5.6))

    plot_grouped_metric(
        axis,
        summary,
        x_column="density_polynomial_degree",
        floor=error_floor,
    )

    configure_axis(
        axis,
        xlabel="Density polynomial degree",
        ylabel="Relative partition-function error",
        title=(
            "Partition-function error under float32 and float64 "
            f"(simplex dimension {dimension})"
        ),
    )

    density_degrees = sorted(
        subset["density_polynomial_degree"].unique()
    )
    axis.set_xticks(density_degrees)

    add_external_legend(figure, axis)

    save_figure(
        figure,
        output_directory,
        "partition_function_error_vs_degree",
        formats,
        dpi,
    )

    plt.close(figure)


def plot_query_probability_error_vs_degree(
    frame: pd.DataFrame,
    *,
    dimension: int,
    query_scale: float,
    error_floor: float,
    output_directory: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    subset = frame.loc[
        (frame["dimension"] == dimension)
        & np.isclose(
            frame["query_scale"],
            query_scale,
            rtol=0.0,
            atol=1.0e-12,
        )
    ].copy()

    summary = quantile_summary(
        subset,
        [
            "basis",
            "dtype",
            "source_polynomial_degree",
            "density_polynomial_degree",
        ],
        "query_probability_absolute_error",
    )

    figure, axis = plt.subplots(figsize=(9.0, 5.6))

    plot_grouped_metric(
        axis,
        summary,
        x_column="density_polynomial_degree",
        floor=error_floor,
    )

    configure_axis(
        axis,
        xlabel="Density polynomial degree",
        ylabel="Absolute query-probability error",
        title=(
            "Downstream query error under float32 and float64 "
            f"(simplex dimension {dimension}, query scale "
            f"{query_scale:g})"
        ),
    )

    density_degrees = sorted(
        subset["density_polynomial_degree"].unique()
    )
    axis.set_xticks(density_degrees)

    add_external_legend(figure, axis)

    save_figure(
        figure,
        output_directory,
        "query_probability_error_vs_degree",
        formats,
        dpi,
    )

    plt.close(figure)


def plot_query_probability_error_vs_query_scale(
    frame: pd.DataFrame,
    *,
    dimension: int,
    error_floor: float,
    output_directory: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    subset = frame.loc[
        frame["dimension"] == dimension
    ].copy()

    highest_degree = int(
        subset["density_polynomial_degree"].max()
    )

    subset = subset.loc[
        subset["density_polynomial_degree"] == highest_degree
    ].copy()

    summary = quantile_summary(
        subset,
        [
            "basis",
            "dtype",
            "query_scale",
        ],
        "query_probability_absolute_error",
    )

    figure, axis = plt.subplots(figsize=(9.0, 5.6))

    plot_grouped_metric(
        axis,
        summary,
        x_column="query_scale",
        floor=error_floor,
    )

    configure_axis(
        axis,
        xlabel="Query simplex scale",
        ylabel="Absolute query-probability error",
        title=(
            "Query error versus query-region scale "
            f"(simplex dimension {dimension}, density degree "
            f"{highest_degree})"
        ),
    )

    query_scales = sorted(subset["query_scale"].unique())
    axis.set_xticks(query_scales)

    add_external_legend(figure, axis)

    save_figure(
        figure,
        output_directory,
        "query_probability_error_vs_query_scale",
        formats,
        dpi,
    )

    plt.close(figure)


def plot_query_probability_sensitivity_vs_degree(
    frame: pd.DataFrame,
    *,
    dimension: int,
    query_scale: float,
    sensitivity_floor: float,
    output_directory: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    subset = frame.loc[
        (frame["dimension"] == dimension)
        & np.isclose(
            frame["query_scale"],
            query_scale,
            rtol=0.0,
            atol=1.0e-12,
        )
    ].copy()

    zero_count = int(
        (
            subset[
                "query_probability_perturbation_sensitivity"
            ]
            == 0.0
        ).sum()
    )

    if zero_count:
        warnings.warn(
            f"{zero_count} downstream perturbation-sensitivity "
            "values are exactly zero. They are shown at the "
            "configured positive plotting floor.",
            RuntimeWarning,
            stacklevel=2,
        )

    summary = quantile_summary(
        subset,
        [
            "basis",
            "dtype",
            "source_polynomial_degree",
            "density_polynomial_degree",
        ],
        "query_probability_perturbation_sensitivity",
    )

    figure, axis = plt.subplots(figsize=(9.0, 5.6))

    plot_grouped_metric(
        axis,
        summary,
        x_column="density_polynomial_degree",
        floor=sensitivity_floor,
    )

    configure_axis(
        axis,
        xlabel="Density polynomial degree",
        ylabel="Query-probability perturbation sensitivity",
        title=(
            "Downstream perturbation sensitivity "
            f"(simplex dimension {dimension}, query scale "
            f"{query_scale:g})"
        ),
    )

    density_degrees = sorted(
        subset["density_polynomial_degree"].unique()
    )
    axis.set_xticks(density_degrees)

    add_external_legend(figure, axis)

    save_figure(
        figure,
        output_directory,
        "query_probability_sensitivity_vs_degree",
        formats,
        dpi,
    )

    plt.close(figure)


def plot_downstream_runtime_vs_degree(
    frame: pd.DataFrame,
    *,
    dimension: int,
    query_scale: float,
    runtime_floor_ms: float,
    output_directory: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    subset = frame.loc[
        (frame["dimension"] == dimension)
        & np.isclose(
            frame["query_scale"],
            query_scale,
            rtol=0.0,
            atol=1.0e-12,
        )
    ].copy()

    summary = quantile_summary(
        subset,
        [
            "basis",
            "dtype",
            "source_polynomial_degree",
            "density_polynomial_degree",
        ],
        "total_downstream_evaluation_ms",
    )

    figure, axis = plt.subplots(figsize=(9.0, 5.6))

    plot_grouped_metric(
        axis,
        summary,
        x_column="density_polynomial_degree",
        floor=runtime_floor_ms,
    )

    configure_axis(
        axis,
        xlabel="Density polynomial degree",
        ylabel="Median downstream runtime (ms)",
        title=(
            "Downstream evaluation runtime under float32 and "
            f"float64 (simplex dimension {dimension})"
        ),
    )

    density_degrees = sorted(
        subset["density_polynomial_degree"].unique()
    )
    axis.set_xticks(density_degrees)

    add_external_legend(figure, axis)

    save_figure(
        figure,
        output_directory,
        "downstream_runtime_vs_degree",
        formats,
        dpi,
    )

    plt.close(figure)


def plot_downstream_error_summary(
    frame: pd.DataFrame,
    *,
    error_floor: float,
    output_directory: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    metrics = [
        (
            "Partition function",
            "partition_function_relative_error",
        ),
        (
            "Query probability",
            "query_probability_relative_error",
        ),
    ]

    summary_rows: list[dict[str, object]] = []

    for metric_label, metric_column in metrics:
        grouped = (
            frame.groupby(
                ["basis", "dtype"],
                observed=True,
            )[metric_column]
            .median()
            .reset_index()
        )

        for row in grouped.itertuples(index=False):
            summary_rows.append(
                {
                    "metric": metric_label,
                    "basis": row.basis,
                    "dtype": row.dtype,
                    "median": getattr(
                        row,
                        metric_column,
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows)

    categories = [
        f"{BASIS_LABELS[basis]}\n{DTYPE_LABELS[dtype]}"
        for basis in BASIS_ORDER
        for dtype in DTYPE_ORDER
    ]

    x = np.arange(len(categories), dtype=float)
    bar_width = 0.34

    figure, axis = plt.subplots(figsize=(10.0, 5.7))

    for metric_index, (metric_label, _) in enumerate(metrics):
        values: list[float] = []

        for basis in BASIS_ORDER:
            for dtype in DTYPE_ORDER:
                selected = summary.loc[
                    (summary["metric"] == metric_label)
                    & (summary["basis"] == basis)
                    & (summary["dtype"] == dtype),
                    "median",
                ]

                if selected.empty:
                    values.append(error_floor)
                else:
                    values.append(
                        max(
                            float(selected.iloc[0]),
                            error_floor,
                        )
                    )

        offset = (
            metric_index - (len(metrics) - 1) / 2.0
        ) * bar_width

        axis.bar(
            x + offset,
            values,
            width=bar_width,
            label=metric_label,
            alpha=0.85,
        )

    axis.set_xticks(x)
    axis.set_xticklabels(categories)
    axis.set_yscale("log")
    axis.set_ylabel("Median relative error")
    axis.set_title(
        "Aggregate downstream numerical error by basis and precision"
    )
    axis.grid(
        True,
        axis="y",
        which="both",
        alpha=0.25,
    )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=True,
    )

    figure.subplots_adjust(bottom=0.25)

    save_figure(
        figure,
        output_directory,
        "downstream_error_summary",
        formats,
        dpi,
    )

    plt.close(figure)


def print_configuration(
    frame: pd.DataFrame,
    *,
    results_path: Path,
    output_directory: Path,
    formats: Sequence[str],
    detail_dimension: int,
    query_scale: float,
) -> None:
    print()
    print("Downstream plotting configuration")
    print("=================================")
    print(f"results path: {results_path}")
    print(f"output directory: {output_directory}")
    print(f"formats: {list(formats)}")
    print(
        "dimensions: "
        f"{sorted(frame['dimension'].unique().tolist())}"
    )
    print(
        "source degrees: "
        f"{sorted(frame['source_polynomial_degree'].unique().tolist())}"
    )
    print(
        "density degrees: "
        f"{sorted(frame['density_polynomial_degree'].unique().tolist())}"
    )
    print(
        "bases: "
        f"{sorted(frame['basis'].unique().tolist())}"
    )
    print(
        "dtypes: "
        f"{sorted(frame['dtype'].unique().tolist())}"
    )
    print(
        "query scales: "
        f"{sorted(frame['query_scale'].unique().tolist())}"
    )
    print(f"records: {len(frame)}")
    print(f"detail dimension: {detail_dimension}")
    print(f"detail query scale: {query_scale:g}")


def print_metric_summary(frame: pd.DataFrame) -> None:
    columns = [
        "partition_function_relative_error",
        "query_probability_absolute_error",
        "query_probability_relative_error",
        "query_probability_perturbation_sensitivity",
        "total_downstream_evaluation_ms",
    ]

    summary = (
        frame.groupby(
            ["basis", "dtype"],
            observed=True,
        )[columns]
        .median()
        .reset_index()
        .sort_values(["basis", "dtype"])
    )

    print()
    print("Median downstream metrics by basis and dtype")
    print("============================================")
    print(summary.to_string(index=False))


def print_error_quantiles(frame: pd.DataFrame) -> None:
    rows: list[dict[str, object]] = []

    for dtype in DTYPE_ORDER:
        subset = frame.loc[frame["dtype"] == dtype]

        if subset.empty:
            continue

        values = subset[
            "query_probability_absolute_error"
        ]

        rows.append(
            {
                "dtype": dtype,
                "median_absolute_query_error": values.median(),
                "q1_absolute_query_error": values.quantile(0.25),
                "q3_absolute_query_error": values.quantile(0.75),
                "maximum_absolute_query_error": values.max(),
            }
        )

    summary = pd.DataFrame(rows)

    print()
    print("Query-probability absolute error by dtype")
    print("=========================================")
    print(summary.to_string(index=False))


def print_maximum_error_record(frame: pd.DataFrame) -> None:
    index = frame[
        "query_probability_absolute_error"
    ].idxmax()

    columns = [
        "basis",
        "dtype",
        "dimension",
        "source_polynomial_degree",
        "density_polynomial_degree",
        "trial",
        "query_scale",
        "partition_function_relative_error",
        "query_probability_absolute_error",
        "query_probability_relative_error",
        "query_probability_perturbation_sensitivity",
        "total_downstream_evaluation_ms",
    ]

    record = frame.loc[index, columns]

    print()
    print("Maximum query-probability error")
    print("===============================")
    print(record.to_string())


def main() -> None:
    arguments = parse_arguments()

    validate_positive(
        arguments.error_floor,
        "--error-floor",
    )
    validate_positive(
        arguments.sensitivity_floor,
        "--sensitivity-floor",
    )
    validate_positive(
        arguments.runtime_floor_ms,
        "--runtime-floor-ms",
    )

    frame = load_results(arguments.results)

    detail_dimension = determine_detail_dimension(
        frame,
        arguments.detail_dimension,
    )
    query_scale = determine_query_scale(
        frame,
        arguments.query_scale,
    )

    formats = tuple(dict.fromkeys(arguments.format))

    print_configuration(
        frame,
        results_path=arguments.results,
        output_directory=arguments.output_directory,
        formats=formats,
        detail_dimension=detail_dimension,
        query_scale=query_scale,
    )

    print_metric_summary(frame)
    print_error_quantiles(frame)
    print_maximum_error_record(frame)

    plot_partition_function_error_vs_degree(
        frame,
        dimension=detail_dimension,
        query_scale=query_scale,
        error_floor=arguments.error_floor,
        output_directory=arguments.output_directory,
        formats=formats,
        dpi=arguments.dpi,
    )

    plot_query_probability_error_vs_degree(
        frame,
        dimension=detail_dimension,
        query_scale=query_scale,
        error_floor=arguments.error_floor,
        output_directory=arguments.output_directory,
        formats=formats,
        dpi=arguments.dpi,
    )

    plot_query_probability_error_vs_query_scale(
        frame,
        dimension=detail_dimension,
        error_floor=arguments.error_floor,
        output_directory=arguments.output_directory,
        formats=formats,
        dpi=arguments.dpi,
    )

    plot_query_probability_sensitivity_vs_degree(
        frame,
        dimension=detail_dimension,
        query_scale=query_scale,
        sensitivity_floor=arguments.sensitivity_floor,
        output_directory=arguments.output_directory,
        formats=formats,
        dpi=arguments.dpi,
    )

    plot_downstream_runtime_vs_degree(
        frame,
        dimension=detail_dimension,
        query_scale=query_scale,
        runtime_floor_ms=arguments.runtime_floor_ms,
        output_directory=arguments.output_directory,
        formats=formats,
        dpi=arguments.dpi,
    )

    plot_downstream_error_summary(
        frame,
        error_floor=arguments.error_floor,
        output_directory=arguments.output_directory,
        formats=formats,
        dpi=arguments.dpi,
    )

    if arguments.show:
        plt.show()


if __name__ == "__main__":
    main()