from __future__ import annotations

"""
Plot the unified n-dimensional simplex benchmark.

Expected input:
    results/simplex/dimension_results.csv

Default outputs:
    figures/condition_number_vs_dimension.pdf
    figures/relative_error_vs_dimension.pdf
    figures/perturbation_sensitivity_vs_dimension.pdf
    figures/runtime_and_basis_count_vs_dimension.pdf

The script is intentionally tolerant of small differences in column naming. It supports
common alternatives such as:

    dimension / n
    degree / d
    basis_count / num_basis_functions / M
    condition_number / cond
    relative_error / relerr
    perturbation_sensitivity / sensitivity / sens
    runtime_seconds / mean time(s) / runtime_ms / time_ms

Examples:
    python -m analysis.plots.plot_dimension_benchmark

    python -m analysis.plots.plot_dimension_benchmark \
        --input results/simplex/dimension_results.csv \
        --degree 5 \
        --output-dir figures

    python -m analysis.plots.plot_dimension_benchmark \
        --degrees 1 2 3 5 \
        --format pdf png svg
"""

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASIS_ORDER = ("monomial", "legendre", "chebyshev")

BASIS_LABELS = {
    "monomial": "Monomial",
    "legendre": "Legendre",
    "chebyshev": "Chebyshev",
}

COLUMN_ALIASES = {
    "basis": ("basis",),
    "dimension": ("dimension", "n", "simplex_dimension"),
    "degree": ("degree", "d", "polynomial_degree"),
    "trial": ("trial", "trial_index"),
    "basis_count": (
        "basis_count",
        "num_basis_functions",
        "number_of_basis_functions",
        "n_basis",
        "M",
    ),
    "condition_number": (
        "condition_number",
        "estimated_condition_number",
        "cond",
        "kappa",
    ),
    "relative_error": (
        "relative_error",
        "relative_integration_error",
        "relerr",
    ),
    "perturbation_sensitivity": (
        "perturbation_sensitivity",
        "sensitivity",
        "sens",
    ),
    "runtime_seconds": (
        "runtime_seconds",
        "runtime_s",
        "elapsed_seconds",
        "mean time(s)",
        "mean_time_s",
    ),
    "runtime_milliseconds": (
        "runtime_ms",
        "time_ms",
        "elapsed_ms",
        "mean_time_ms",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Chapter 5 plots from the unified n-dimensional "
            "simplex benchmark."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/simplex/dimension_results.csv"),
        help="Path to dimension_results.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory in which figures are written.",
    )

    parser.add_argument(
        "--degree",
        type=int,
        default=None,
        help=(
            "Plot one degree across dimensions. If omitted, the script chooses "
            "the largest degree available in every plotted dimension."
        ),
    )

    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Plot several degrees. This overrides --degree. Each requested degree "
            "must be available in at least two dimensions."
        ),
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of simplex dimensions.",
    )

    parser.add_argument(
        "--bases",
        nargs="+",
        choices=BASIS_ORDER,
        default=list(BASIS_ORDER),
        help="Basis representations to include.",
    )

    parser.add_argument(
        "--format",
        nargs="+",
        choices=("pdf", "png", "svg"),
        default=["pdf"],
        help="One or more output formats.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster DPI for PNG output.",
    )

    parser.add_argument(
        "--error-floor",
        type=float,
        default=1.0e-18,
        help="Positive display floor used for exact zero relative errors.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving.",
    )

    return parser.parse_args()


def canonicalise_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Rename supported input columns to a stable internal schema."""
    rename_map: dict[str, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in data.columns:
                rename_map[alias] = canonical
                break

    result = data.rename(columns=rename_map).copy()

    required = {
        "basis",
        "dimension",
        "degree",
        "condition_number",
        "relative_error",
        "perturbation_sensitivity",
    }

    missing = sorted(required.difference(result.columns))

    if missing:
        raise KeyError(
            "The benchmark CSV is missing required columns after alias resolution: "
            f"{missing}\nAvailable columns: {list(data.columns)}"
        )

    if "runtime_milliseconds" not in result.columns:
        if "runtime_seconds" not in result.columns:
            raise KeyError(
                "Could not find a runtime column. Expected one of: "
                f"{COLUMN_ALIASES['runtime_seconds'] + COLUMN_ALIASES['runtime_milliseconds']}"
            )

        result["runtime_milliseconds"] = (
            pd.to_numeric(result["runtime_seconds"], errors="coerce") * 1_000.0
        )

    if "basis_count" not in result.columns:
        result["basis_count"] = [
            math.comb(int(n) + int(d), int(d))
            for n, d in zip(result["dimension"], result["degree"])
        ]

    return result


def validate_and_clean(data: pd.DataFrame) -> pd.DataFrame:
    """Validate finite values and normalise categorical fields."""
    cleaned = data.copy()

    cleaned["basis"] = (
        cleaned["basis"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    numeric_columns = [
        "dimension",
        "degree",
        "basis_count",
        "condition_number",
        "relative_error",
        "perturbation_sensitivity",
        "runtime_milliseconds",
    ]

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    invalid_basis = sorted(
        set(cleaned["basis"]) - set(BASIS_ORDER)
    )

    if invalid_basis:
        raise ValueError(
            f"Unsupported basis values: {invalid_basis}. "
            f"Expected values from {BASIS_ORDER}."
        )

    if cleaned[numeric_columns].isna().any().any():
        bad_columns = cleaned[numeric_columns].columns[
            cleaned[numeric_columns].isna().any()
        ].tolist()

        raise ValueError(
            f"Non-numeric or missing values were found in columns: {bad_columns}"
        )

    values = cleaned[numeric_columns].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise FloatingPointError(
            "The result file contains NaN or infinite numeric values."
        )

    if (cleaned["condition_number"] <= 0.0).any():
        raise ValueError(
            "Condition numbers must be strictly positive."
        )

    if (cleaned["relative_error"] < 0.0).any():
        raise ValueError(
            "Relative errors cannot be negative."
        )

    if (cleaned["perturbation_sensitivity"] < 0.0).any():
        raise ValueError(
            "Perturbation sensitivities cannot be negative."
        )

    if (cleaned["runtime_milliseconds"] < 0.0).any():
        raise ValueError(
            "Runtime values cannot be negative."
        )

    cleaned["dimension"] = cleaned["dimension"].astype(int)
    cleaned["degree"] = cleaned["degree"].astype(int)
    cleaned["basis_count"] = cleaned["basis_count"].astype(int)

    return cleaned


def filter_data(
    data: pd.DataFrame,
    dimensions: list[int] | None,
    bases: list[str],
) -> pd.DataFrame:
    result = data[data["basis"].isin(bases)].copy()

    if dimensions is not None:
        result = result[
            result["dimension"].isin(dimensions)
        ].copy()

    if result.empty:
        raise ValueError(
            "No records remain after applying the requested filters."
        )

    return result


def choose_degrees(
    data: pd.DataFrame,
    one_degree: int | None,
    several_degrees: list[int] | None,
) -> list[int]:
    if several_degrees is not None:
        requested = sorted(set(several_degrees))
    elif one_degree is not None:
        requested = [one_degree]
    else:
        degree_sets = [
            set(group["degree"].unique())
            for _, group in data.groupby("dimension")
        ]

        common = (
            set.intersection(*degree_sets)
            if degree_sets
            else set()
        )

        if not common:
            raise ValueError(
                "No polynomial degree is shared by all selected dimensions. "
                "Pass --degree or --degrees explicitly."
            )

        requested = [max(common)]

    available_counts = (
        data.groupby("degree")["dimension"]
        .nunique()
        .to_dict()
    )

    usable = [
        degree
        for degree in requested
        if available_counts.get(degree, 0) >= 2
    ]

    missing = sorted(set(requested) - set(usable))

    if missing:
        raise ValueError(
            "The following requested degrees are not available in at least two "
            f"selected dimensions: {missing}"
        )

    return usable


def aggregate(
    data: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Return median, first quartile, and third quartile by configuration."""
    grouped = (
        data
        .groupby(
            ["basis", "dimension", "degree"],
            as_index=False,
        )[metric]
        .agg(
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        )
        .sort_values(
            ["degree", "basis", "dimension"]
        )
    )

    return grouped


def basis_count_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    result = (
        data
        .groupby(
            ["dimension", "degree"],
            as_index=False,
        )["basis_count"]
        .median()
        .sort_values(
            ["degree", "dimension"]
        )
    )

    result["basis_count"] = (
        result["basis_count"].astype(int)
    )

    return result


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Iterable[str],
    dpi: int,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for extension in formats:
        path = output_dir / f"{stem}.{extension}"

        save_kwargs: dict[str, object] = {
            "bbox_inches": "tight",
            "pad_inches": 0.12,
        }

        if extension == "png":
            save_kwargs["dpi"] = dpi

        figure.savefig(
            path,
            **save_kwargs,
        )

        print(f"Wrote {path}")


def line_label(
    basis: str,
    degree: int,
    number_of_degrees: int,
) -> str:
    basis_label = BASIS_LABELS.get(
        basis,
        basis.title(),
    )

    if number_of_degrees == 1:
        return basis_label

    return f"{basis_label}, d={degree}"


def legend_column_count(
    number_of_entries: int,
    preferred_columns: int = 4,
) -> int:
    """
    Choose a compact legend width.

    Four columns work well for the common twelve-entry case produced by
    three bases and four degrees.
    """
    if number_of_entries <= 3:
        return number_of_entries

    return min(
        preferred_columns,
        number_of_entries,
    )


def plot_metric_vs_dimension(
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_stem: str,
    degrees: list[int],
    output_dir: Path,
    formats: list[str],
    dpi: int,
    log_y: bool,
    display_floor: float | None = None,
) -> None:
    selected = data[
        data["degree"].isin(degrees)
    ].copy()

    if display_floor is not None:
        selected[metric] = selected[metric].clip(
            lower=display_floor
        )

    summary = aggregate(
        selected,
        metric,
    )

    figure, axis = plt.subplots(
        figsize=(9.0, 6.8)
    )

    for degree in degrees:
        for basis in BASIS_ORDER:
            if basis not in selected["basis"].unique():
                continue

            group = summary[
                (summary["degree"] == degree)
                & (summary["basis"] == basis)
            ].sort_values("dimension")

            if group.empty:
                continue

            x = group["dimension"].to_numpy(
                dtype=float
            )
            median = group["median"].to_numpy(
                dtype=float
            )
            q1 = group["q1"].to_numpy(
                dtype=float
            )
            q3 = group["q3"].to_numpy(
                dtype=float
            )

            axis.plot(
                x,
                median,
                marker="o",
                linewidth=1.5,
                markersize=5,
                label=line_label(
                    basis,
                    degree,
                    len(degrees),
                ),
            )

            axis.fill_between(
                x,
                q1,
                q3,
                alpha=0.12,
            )

    axis.set_xlabel(
        "Simplex dimension"
    )
    axis.set_ylabel(
        ylabel
    )
    axis.set_title(
        title
    )
    axis.set_xticks(
        sorted(
            selected["dimension"].unique()
        )
    )
    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    if log_y:
        axis.set_yscale("log")

    handles, labels = (
        axis.get_legend_handles_labels()
    )

    if handles:
        figure.legend(
            handles,
            labels,
            title="Basis and degree",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=legend_column_count(
                len(handles),
                preferred_columns=4,
            ),
            frameon=True,
            columnspacing=1.4,
            handlelength=2.0,
            handletextpad=0.6,
            borderaxespad=0.0,
        )

    # Reserve space below the axes for the figure-level legend.
    figure.subplots_adjust(
        left=0.11,
        right=0.97,
        top=0.91,
        bottom=0.27 if len(handles) > 6 else 0.22,
    )

    save_figure(
        figure=figure,
        output_dir=output_dir,
        stem=output_stem,
        formats=formats,
        dpi=dpi,
    )

    plt.close(figure)


def plot_runtime_and_basis_count(
    data: pd.DataFrame,
    degrees: list[int],
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    selected = data[
        data["degree"].isin(degrees)
    ].copy()

    runtime_summary = aggregate(
        selected,
        "runtime_milliseconds",
    )

    counts = basis_count_table(
        selected
    )

    figure, runtime_axis = plt.subplots(
        figsize=(9.2, 7.8)
    )

    count_axis = runtime_axis.twinx()

    for degree in degrees:
        for basis in BASIS_ORDER:
            if basis not in selected["basis"].unique():
                continue

            group = runtime_summary[
                (runtime_summary["degree"] == degree)
                & (runtime_summary["basis"] == basis)
            ].sort_values("dimension")

            if group.empty:
                continue

            x = group["dimension"].to_numpy(
                dtype=float
            )
            median = group["median"].to_numpy(
                dtype=float
            )
            q1 = group["q1"].to_numpy(
                dtype=float
            )
            q3 = group["q3"].to_numpy(
                dtype=float
            )

            runtime_axis.plot(
                x,
                median,
                marker="o",
                linewidth=1.5,
                markersize=5,
                label=line_label(
                    basis,
                    degree,
                    len(degrees),
                ),
            )

            runtime_axis.fill_between(
                x,
                q1,
                q3,
                alpha=0.10,
            )

    for degree in degrees:
        degree_counts = counts[
            counts["degree"] == degree
        ].sort_values("dimension")

        if degree_counts.empty:
            continue

        label = (
            "Basis count"
            if len(degrees) == 1
            else f"d={degree}"
        )

        count_axis.plot(
            degree_counts["dimension"],
            degree_counts["basis_count"],
            linestyle="--",
            marker="s",
            linewidth=1.4,
            markersize=5,
            label=label,
        )

    runtime_axis.set_xlabel(
        "Simplex dimension"
    )
    runtime_axis.set_ylabel(
        "Median runtime (ms)"
    )
    count_axis.set_ylabel(
        "Number of basis functions"
    )

    runtime_axis.set_yscale("log")
    count_axis.set_yscale("log")

    runtime_axis.set_xticks(
        sorted(
            selected["dimension"].unique()
        )
    )

    runtime_axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    runtime_axis.set_title(
        "Runtime and basis-count scaling with simplex dimension"
    )

    runtime_handles, runtime_labels = (
        runtime_axis.get_legend_handles_labels()
    )

    count_handles, count_labels = (
        count_axis.get_legend_handles_labels()
    )

    if runtime_handles:
        runtime_legend = figure.legend(
            runtime_handles,
            runtime_labels,
            title="Runtime: basis and degree",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.105),
            ncol=legend_column_count(
                len(runtime_handles),
                preferred_columns=4,
            ),
            frameon=True,
            columnspacing=1.35,
            handlelength=2.0,
            handletextpad=0.6,
            borderaxespad=0.0,
        )

        figure.add_artist(runtime_legend)

    if count_handles:
        figure.legend(
            count_handles,
            count_labels,
            title="Basis count by degree",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=legend_column_count(
                len(count_handles),
                preferred_columns=4,
            ),
            frameon=True,
            columnspacing=1.8,
            handlelength=2.0,
            handletextpad=0.6,
            borderaxespad=0.0,
        )

    # Reserve two legend rows below the axes.
    figure.subplots_adjust(
        left=0.11,
        right=0.89,
        top=0.91,
        bottom=0.35,
    )

    save_figure(
        figure=figure,
        output_dir=output_dir,
        stem="runtime_and_basis_count_vs_dimension",
        formats=formats,
        dpi=dpi,
    )

    plt.close(figure)


def print_summary(
    data: pd.DataFrame,
    degrees: list[int],
) -> None:
    selected = data[
        data["degree"].isin(degrees)
    ].copy()

    print("\nPlotting configuration")
    print(
        "dimensions: "
        f"{sorted(selected['dimension'].unique().tolist())}"
    )
    print(
        f"degrees: {degrees}"
    )
    print(
        "bases: "
        f"{sorted(selected['basis'].unique().tolist())}"
    )
    print(
        f"records: {len(selected)}"
    )

    for degree in degrees:
        print(
            f"\nDegree {degree}"
        )

        degree_data = selected[
            selected["degree"] == degree
        ]

        table = (
            degree_data
            .groupby(
                ["dimension", "basis"],
                as_index=False,
            )
            .agg(
                median_condition=(
                    "condition_number",
                    "median",
                ),
                median_error=(
                    "relative_error",
                    "median",
                ),
                median_sensitivity=(
                    "perturbation_sensitivity",
                    "median",
                ),
                median_runtime_ms=(
                    "runtime_milliseconds",
                    "median",
                ),
                basis_count=(
                    "basis_count",
                    "median",
                ),
            )
            .sort_values(
                ["dimension", "basis"]
            )
        )

        print(
            table.to_string(
                index=False
            )
        )


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Could not find benchmark CSV: {args.input}"
        )

    raw = pd.read_csv(
        args.input
    )

    data = canonicalise_column_names(
        raw
    )

    data = validate_and_clean(
        data
    )

    data = filter_data(
        data=data,
        dimensions=args.dimensions,
        bases=args.bases,
    )

    degrees = choose_degrees(
        data=data,
        one_degree=args.degree,
        several_degrees=args.degrees,
    )

    print_summary(
        data,
        degrees,
    )

    plot_metric_vs_dimension(
        data=data,
        metric="condition_number",
        ylabel="Estimated condition number",
        title="Condition number as a function of simplex dimension",
        output_stem="condition_number_vs_dimension",
        degrees=degrees,
        output_dir=args.output_dir,
        formats=args.format,
        dpi=args.dpi,
        log_y=True,
    )

    plot_metric_vs_dimension(
        data=data,
        metric="relative_error",
        ylabel="Relative integration error",
        title="Relative integration error as a function of simplex dimension",
        output_stem="relative_error_vs_dimension",
        degrees=degrees,
        output_dir=args.output_dir,
        formats=args.format,
        dpi=args.dpi,
        log_y=True,
        display_floor=args.error_floor,
    )

    plot_metric_vs_dimension(
        data=data,
        metric="perturbation_sensitivity",
        ylabel="Perturbation sensitivity",
        title="Perturbation sensitivity as a function of simplex dimension",
        output_stem="perturbation_sensitivity_vs_dimension",
        degrees=degrees,
        output_dir=args.output_dir,
        formats=args.format,
        dpi=args.dpi,
        log_y=True,
        display_floor=np.finfo(
            np.float64
        ).tiny,
    )

    plot_runtime_and_basis_count(
        data=data,
        degrees=degrees,
        output_dir=args.output_dir,
        formats=args.format,
        dpi=args.dpi,
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()