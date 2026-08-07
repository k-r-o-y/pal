from __future__ import annotations

"""
Plot the float32-versus-float64 simplex precision benchmark.

Expected input
--------------
    results/simplex/precision_results.csv

Default outputs
---------------
    figures/condition_number_vs_precision.pdf
    figures/relative_error_vs_precision.pdf
    figures/rank_fraction_vs_precision.pdf
    figures/perturbation_sensitivity_vs_precision.pdf
    figures/runtime_vs_precision.pdf

The figures compare float32 and float64 while preserving basis, degree, and
dimension information.

Recommended benchmark rerun
---------------------------
The perturbation magnitude should be comfortably above float32 machine epsilon.
For example:

    python -m analysis.simplex.run_precision_benchmark \
        --trials 10 \
        --perturbation-magnitude 1e-5

Examples
--------
Generate the default PDF figures:

    python -m analysis.plots.plot_precision_benchmark

Generate PDF, PNG, and SVG figures:

    python -m analysis.plots.plot_precision_benchmark \
        --format pdf png svg

Restrict the plotted dimensions and degrees:

    python -m analysis.plots.plot_precision_benchmark \
        --dimensions 1 2 3 \
        --degrees 1 2 3 5 8

Select one dimension for detailed degree curves:

    python -m analysis.plots.plot_precision_benchmark \
        --detail-dimension 3
"""

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


BASIS_ORDER: tuple[str, ...] = (
    "monomial",
    "legendre",
    "chebyshev",
)

BASIS_LABELS: dict[str, str] = {
    "monomial": "Monomial",
    "legendre": "Legendre",
    "chebyshev": "Chebyshev",
}

DTYPE_ORDER: tuple[str, ...] = (
    "float32",
    "float64",
)

DTYPE_LABELS: dict[str, str] = {
    "float32": "float32",
    "float64": "float64",
}

DTYPE_LINESTYLES: dict[str, str] = {
    "float32": "--",
    "float64": "-",
}

DTYPE_MARKERS: dict[str, str] = {
    "float32": "s",
    "float64": "o",
}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "basis": (
        "basis",
        "basis_name",
    ),
    "dimension": (
        "dimension",
        "n",
        "simplex_dimension",
    ),
    "degree": (
        "degree",
        "d",
        "polynomial_degree",
    ),
    "trial": (
        "trial",
        "trial_index",
    ),
    "dtype": (
        "dtype",
        "precision",
        "floating_point_dtype",
    ),
    "basis_count": (
        "basis_count",
        "num_basis_functions",
        "number_of_basis_functions",
        "n_basis",
        "M",
    ),
    "matrix_rank": (
        "matrix_rank",
        "rank",
        "numerical_rank",
    ),
    "matrix_columns": (
        "matrix_columns",
        "num_columns",
        "basis_matrix_columns",
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
    "perturbation_magnitude": (
        "perturbation_magnitude",
        "delta",
        "coefficient_perturbation",
    ),
    "runtime_seconds": (
        "runtime_seconds",
        "runtime_s",
        "elapsed_seconds",
        "mean_time_s",
        "mean time(s)",
    ),
    "runtime_milliseconds": (
        "runtime_milliseconds",
        "runtime_ms",
        "time_ms",
        "elapsed_ms",
        "mean_time_ms",
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Chapter 5 figures from the float32-versus-float64 "
            "simplex precision benchmark."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/simplex/precision_results.csv"),
        help="Path to the detailed precision benchmark CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory in which figures are written.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of simplex dimensions.",
    )
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of polynomial degrees.",
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
        help="Floating-point precisions to include.",
    )
    parser.add_argument(
        "--detail-dimension",
        type=int,
        default=None,
        help=(
            "Dimension used for the degree-based condition, error, sensitivity, "
            "and runtime figures. If omitted, the largest selected dimension "
            "having at least two degrees is used."
        ),
    )
    parser.add_argument(
        "--rank-degree",
        type=int,
        default=None,
        help=(
            "Degree used for the rank-fraction-versus-dimension figure. "
            "If omitted, the largest degree available in every selected "
            "dimension is used."
        ),
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
        help="Raster DPI used for PNG output.",
    )
    parser.add_argument(
        "--error-floor",
        type=float,
        default=1.0e-18,
        help="Positive display floor for exact zero relative errors.",
    )
    parser.add_argument(
        "--sensitivity-floor",
        type=float,
        default=1.0e-12,
        help=(
            "Positive display floor for zero perturbation sensitivities. "
            "Zero values can occur when a perturbation rounds away in float32."
        ),
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
        "trial",
        "dtype",
        "condition_number",
        "relative_error",
        "perturbation_sensitivity",
    }

    missing = sorted(required.difference(result.columns))
    if missing:
        raise KeyError(
            "The precision benchmark CSV is missing required columns after "
            f"alias resolution: {missing}\n"
            f"Available columns: {list(data.columns)}"
        )

    if "runtime_milliseconds" not in result.columns:
        if "runtime_seconds" not in result.columns:
            runtime_aliases = (
                COLUMN_ALIASES["runtime_seconds"]
                + COLUMN_ALIASES["runtime_milliseconds"]
            )
            raise KeyError(
                "Could not find a runtime column. Expected one of: "
                f"{runtime_aliases}"
            )

        result["runtime_milliseconds"] = (
            pd.to_numeric(
                result["runtime_seconds"],
                errors="coerce",
            )
            * 1_000.0
        )

    if "matrix_columns" not in result.columns:
        if "basis_count" not in result.columns:
            raise KeyError(
                "Could not infer matrix column count because neither "
                "'matrix_columns' nor 'basis_count' is present."
            )
        result["matrix_columns"] = result["basis_count"]

    if "matrix_rank" not in result.columns:
        raise KeyError(
            "The precision benchmark must include matrix rank information."
        )

    if "basis_count" not in result.columns:
        result["basis_count"] = result["matrix_columns"]

    return result


def normalise_dtype_name(value: object) -> str:
    """Convert common precision labels to float32 or float64."""
    text = str(value).strip().lower()

    aliases = {
        "float32": "float32",
        "np.float32": "float32",
        "<class 'numpy.float32'>": "float32",
        "single": "float32",
        "32": "float32",
        "fp32": "float32",
        "float64": "float64",
        "np.float64": "float64",
        "<class 'numpy.float64'>": "float64",
        "double": "float64",
        "64": "float64",
        "fp64": "float64",
    }

    return aliases.get(text, text)


def validate_and_clean(data: pd.DataFrame) -> pd.DataFrame:
    """Validate numeric fields and normalise categorical values."""
    cleaned = data.copy()

    cleaned["basis"] = (
        cleaned["basis"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    cleaned["dtype"] = cleaned["dtype"].map(normalise_dtype_name)

    numeric_columns = [
        "dimension",
        "degree",
        "trial",
        "basis_count",
        "matrix_rank",
        "matrix_columns",
        "condition_number",
        "relative_error",
        "perturbation_sensitivity",
        "runtime_milliseconds",
    ]

    if "perturbation_magnitude" in cleaned.columns:
        numeric_columns.append("perturbation_magnitude")

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    invalid_bases = sorted(
        set(cleaned["basis"]) - set(BASIS_ORDER)
    )
    if invalid_bases:
        raise ValueError(
            f"Unsupported basis values: {invalid_bases}. "
            f"Expected values from {BASIS_ORDER}."
        )

    invalid_dtypes = sorted(
        set(cleaned["dtype"]) - set(DTYPE_ORDER)
    )
    if invalid_dtypes:
        raise ValueError(
            f"Unsupported dtype values: {invalid_dtypes}. "
            f"Expected values from {DTYPE_ORDER}."
        )

    if cleaned[numeric_columns].isna().any().any():
        bad_columns = cleaned[numeric_columns].columns[
            cleaned[numeric_columns].isna().any()
        ].tolist()
        raise ValueError(
            "Non-numeric or missing values were found in columns: "
            f"{bad_columns}"
        )

    numeric_values = cleaned[numeric_columns].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric_values).all():
        raise FloatingPointError(
            "The precision result file contains NaN or infinite values."
        )

    if (cleaned["dimension"] < 1).any():
        raise ValueError("Dimensions must be at least one.")

    if (cleaned["degree"] < 0).any():
        raise ValueError("Degrees cannot be negative.")

    if (cleaned["basis_count"] < 1).any():
        raise ValueError("Basis counts must be positive.")

    if (cleaned["matrix_columns"] < 1).any():
        raise ValueError("Matrix column counts must be positive.")

    if (cleaned["matrix_rank"] < 0).any():
        raise ValueError("Matrix ranks cannot be negative.")

    if (
        cleaned["matrix_rank"]
        > cleaned["matrix_columns"]
    ).any():
        raise ValueError(
            "A matrix rank exceeds its matrix column count."
        )

    if (cleaned["condition_number"] <= 0.0).any():
        raise ValueError(
            "Condition numbers must be strictly positive."
        )

    if (cleaned["relative_error"] < 0.0).any():
        raise ValueError("Relative errors cannot be negative.")

    if (cleaned["perturbation_sensitivity"] < 0.0).any():
        raise ValueError(
            "Perturbation sensitivities cannot be negative."
        )

    if (cleaned["runtime_milliseconds"] < 0.0).any():
        raise ValueError("Runtime values cannot be negative.")

    integer_columns = [
        "dimension",
        "degree",
        "trial",
        "basis_count",
        "matrix_rank",
        "matrix_columns",
    ]

    for column in integer_columns:
        cleaned[column] = cleaned[column].astype(int)

    cleaned["rank_fraction"] = (
        cleaned["matrix_rank"]
        / cleaned["matrix_columns"]
    )

    return cleaned


def filter_data(
    data: pd.DataFrame,
    *,
    dimensions: Sequence[int] | None,
    degrees: Sequence[int] | None,
    bases: Sequence[str],
    dtypes: Sequence[str],
) -> pd.DataFrame:
    """Apply requested dimension, degree, basis, and dtype filters."""
    result = data[
        data["basis"].isin(bases)
        & data["dtype"].isin(dtypes)
    ].copy()

    if dimensions is not None:
        result = result[
            result["dimension"].isin(dimensions)
        ].copy()

    if degrees is not None:
        result = result[
            result["degree"].isin(degrees)
        ].copy()

    if result.empty:
        raise ValueError(
            "No records remain after applying the requested filters."
        )

    return result


def resolve_detail_dimension(
    data: pd.DataFrame,
    requested_dimension: int | None,
) -> int:
    """Choose a dimension for degree-based precision plots."""
    available_dimensions = sorted(
        int(value)
        for value in data["dimension"].unique()
    )

    if requested_dimension is not None:
        if requested_dimension not in available_dimensions:
            raise ValueError(
                f"Requested detail dimension {requested_dimension} "
                f"is unavailable. Available dimensions: "
                f"{available_dimensions}"
            )
        return requested_dimension

    degree_counts = (
        data.groupby("dimension")["degree"]
        .nunique()
        .sort_index()
    )
    candidates = degree_counts[
        degree_counts >= 2
    ].index.tolist()

    if not candidates:
        raise ValueError(
            "No selected dimension contains at least two degrees."
        )

    return int(max(candidates))


def resolve_rank_degree(
    data: pd.DataFrame,
    requested_degree: int | None,
) -> int:
    """Choose one degree shared across all selected dimensions."""
    dimensions = sorted(
        int(value)
        for value in data["dimension"].unique()
    )

    if requested_degree is not None:
        missing_dimensions = [
            dimension
            for dimension in dimensions
            if requested_degree
            not in set(
                data.loc[
                    data["dimension"] == dimension,
                    "degree",
                ].unique()
            )
        ]

        if missing_dimensions:
            raise ValueError(
                f"Rank degree {requested_degree} is unavailable in "
                f"dimensions {missing_dimensions}."
            )

        return requested_degree

    degree_sets = [
        set(
            int(value)
            for value in data.loc[
                data["dimension"] == dimension,
                "degree",
            ].unique()
        )
        for dimension in dimensions
    ]

    common_degrees = (
        set.intersection(*degree_sets)
        if degree_sets
        else set()
    )

    if not common_degrees:
        raise ValueError(
            "No polynomial degree is shared by every selected dimension. "
            "Pass --rank-degree explicitly or change the dimension filter."
        )

    nonzero_degrees = [
        degree
        for degree in common_degrees
        if degree > 0
    ]

    if nonzero_degrees:
        return max(nonzero_degrees)

    return max(common_degrees)


def aggregate_metric(
    data: pd.DataFrame,
    *,
    metric: str,
    grouping_columns: Sequence[str],
) -> pd.DataFrame:
    """Return median and interquartile range for one metric."""
    summary = (
        data.groupby(
            list(grouping_columns),
            as_index=False,
        )[metric]
        .agg(
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        )
        .sort_values(list(grouping_columns))
    )

    return summary


def save_figure(
    figure: plt.Figure,
    *,
    output_dir: Path,
    stem: str,
    formats: Iterable[str],
    dpi: int,
) -> None:
    """Save one figure in every requested format."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for extension in formats:
        output_path = output_dir / f"{stem}.{extension}"

        save_kwargs: dict[str, object] = {
            "bbox_inches": "tight",
        }

        if extension == "png":
            save_kwargs["dpi"] = dpi

        figure.savefig(
            output_path,
            **save_kwargs,
        )
        print(f"Wrote {output_path}")


def basis_dtype_label(
    basis: str,
    dtype: str,
) -> str:
    """Return a concise legend label."""
    return (
        f"{BASIS_LABELS.get(basis, basis.title())}, "
        f"{DTYPE_LABELS.get(dtype, dtype)}"
    )


def ordered_basis_values(data: pd.DataFrame) -> list[str]:
    """Return selected bases in the preferred display order."""
    available = set(data["basis"].unique())
    return [
        basis
        for basis in BASIS_ORDER
        if basis in available
    ]


def ordered_dtype_values(data: pd.DataFrame) -> list[str]:
    """Return selected dtypes in the preferred display order."""
    available = set(data["dtype"].unique())
    return [
        dtype
        for dtype in DTYPE_ORDER
        if dtype in available
    ]


def add_bottom_legend(
    figure: plt.Figure,
    axis: plt.Axes,
    *,
    title: str,
    number_of_columns: int,
    bottom_margin: float,
) -> None:
    """
    Place the legend below the axes so that it cannot overlap plotted data.
    """
    handles, labels = axis.get_legend_handles_labels()

    if handles:
        figure.legend(
            handles,
            labels,
            title=title,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=max(1, number_of_columns),
            frameon=True,
            fontsize=9,
            title_fontsize=10,
            handlelength=2.4,
            columnspacing=1.3,
        )

    figure.subplots_adjust(
        bottom=bottom_margin,
        top=0.90,
        left=0.12,
        right=0.97,
    )


def plot_degree_metric(
    data: pd.DataFrame,
    *,
    dimension: int,
    metric: str,
    ylabel: str,
    title: str,
    output_stem: str,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    log_y: bool,
    display_floor: float | None = None,
) -> None:
    """Plot one metric against degree for one selected dimension."""
    selected = data[
        data["dimension"] == dimension
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No records are available for dimension {dimension}."
        )

    if display_floor is not None:
        selected[metric] = selected[metric].clip(
            lower=display_floor
        )

    summary = aggregate_metric(
        selected,
        metric=metric,
        grouping_columns=(
            "basis",
            "dtype",
            "degree",
        ),
    )

    figure, axis = plt.subplots(
        figsize=(9.4, 6.7)
    )

    for basis in ordered_basis_values(selected):
        for dtype in ordered_dtype_values(selected):
            group = summary[
                (summary["basis"] == basis)
                & (summary["dtype"] == dtype)
            ].sort_values("degree")

            if group.empty:
                continue

            x = group["degree"].to_numpy(
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

            line = axis.plot(
                x,
                median,
                marker=DTYPE_MARKERS[dtype],
                linestyle=DTYPE_LINESTYLES[dtype],
                linewidth=1.8,
                markersize=5.5,
                label=basis_dtype_label(
                    basis,
                    dtype,
                ),
            )[0]

            axis.fill_between(
                x,
                q1,
                q3,
                alpha=0.12,
                color=line.get_color(),
            )

    axis.set_xlabel("Polynomial degree")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_xticks(
        sorted(
            int(value)
            for value in selected["degree"].unique()
        )
    )
    axis.grid(
        True,
        which="both",
        alpha=0.28,
    )

    if log_y:
        axis.set_yscale("log")

    add_bottom_legend(
        figure,
        axis,
        title="Basis and arithmetic precision",
        number_of_columns=3,
        bottom_margin=0.26,
    )

    save_figure(
        figure,
        output_dir=output_dir,
        stem=output_stem,
        formats=formats,
        dpi=dpi,
    )
    plt.close(figure)


def plot_condition_number(
    data: pd.DataFrame,
    *,
    detail_dimension: int,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    """Plot condition number against degree."""
    plot_degree_metric(
        data,
        dimension=detail_dimension,
        metric="condition_number",
        ylabel="Estimated condition number",
        title=(
            "Conditioning under float32 and float64 "
            f"(simplex dimension {detail_dimension})"
        ),
        output_stem="condition_number_vs_precision",
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
        log_y=True,
    )


def plot_relative_error(
    data: pd.DataFrame,
    *,
    detail_dimension: int,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    error_floor: float,
) -> None:
    """Plot relative integration error against degree."""
    plot_degree_metric(
        data,
        dimension=detail_dimension,
        metric="relative_error",
        ylabel="Relative integration error",
        title=(
            "Integration error under float32 and float64 "
            f"(simplex dimension {detail_dimension})"
        ),
        output_stem="relative_error_vs_precision",
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
        log_y=True,
        display_floor=error_floor,
    )


def plot_perturbation_sensitivity(
    data: pd.DataFrame,
    *,
    detail_dimension: int,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    sensitivity_floor: float,
) -> None:
    """Plot coefficient-perturbation sensitivity against degree."""
    selected = data.copy()

    zero_count = int(
        (
            selected["perturbation_sensitivity"]
            == 0.0
        ).sum()
    )

    if zero_count:
        print(
            "Warning: "
            f"{zero_count} perturbation-sensitivity values are exactly zero. "
            "They are displayed at the configured positive plotting floor. "
            "For float32, this can indicate that the requested coefficient "
            "perturbation rounded away."
        )

    plot_degree_metric(
        selected,
        dimension=detail_dimension,
        metric="perturbation_sensitivity",
        ylabel="Perturbation sensitivity",
        title=(
            "Coefficient-perturbation sensitivity under float32 and float64 "
            f"(simplex dimension {detail_dimension})"
        ),
        output_stem="perturbation_sensitivity_vs_precision",
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
        log_y=True,
        display_floor=sensitivity_floor,
    )


def plot_runtime(
    data: pd.DataFrame,
    *,
    detail_dimension: int,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    """Plot integration-stage runtime against degree."""
    plot_degree_metric(
        data,
        dimension=detail_dimension,
        metric="runtime_milliseconds",
        ylabel="Median runtime (ms)",
        title=(
            "Integration-stage runtime under float32 and float64 "
            f"(simplex dimension {detail_dimension})"
        ),
        output_stem="runtime_vs_precision",
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
        log_y=True,
    )


def plot_rank_fraction(
    data: pd.DataFrame,
    *,
    degree: int,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    """
    Plot numerical rank divided by matrix column count against dimension.

    The figure focuses on one common degree so that every dimension is compared
    using the same polynomial complexity.
    """
    selected = data[
        data["degree"] == degree
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No records are available at degree {degree}."
        )

    summary = aggregate_metric(
        selected,
        metric="rank_fraction",
        grouping_columns=(
            "basis",
            "dtype",
            "dimension",
        ),
    )

    figure, axis = plt.subplots(
        figsize=(9.4, 6.7)
    )

    for basis in ordered_basis_values(selected):
        for dtype in ordered_dtype_values(selected):
            group = summary[
                (summary["basis"] == basis)
                & (summary["dtype"] == dtype)
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

            line = axis.plot(
                x,
                median,
                marker=DTYPE_MARKERS[dtype],
                linestyle=DTYPE_LINESTYLES[dtype],
                linewidth=1.8,
                markersize=5.5,
                label=basis_dtype_label(
                    basis,
                    dtype,
                ),
            )[0]

            axis.fill_between(
                x,
                q1,
                q3,
                alpha=0.12,
                color=line.get_color(),
            )

    axis.axhline(
        1.0,
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
    )

    axis.set_xlabel("Simplex dimension")
    axis.set_ylabel(
        "Numerical rank fraction "
        r"$\mathrm{rank}(V)/M$"
    )
    axis.set_title(
        "Numerical rank retained under float32 and float64 "
        f"(degree {degree})"
    )
    axis.set_xticks(
        sorted(
            int(value)
            for value in selected["dimension"].unique()
        )
    )
    axis.set_ylim(
        bottom=0.0,
        top=1.04,
    )
    axis.grid(
        True,
        which="both",
        alpha=0.28,
    )

    add_bottom_legend(
        figure,
        axis,
        title="Basis and arithmetic precision",
        number_of_columns=3,
        bottom_margin=0.26,
    )

    save_figure(
        figure,
        output_dir=output_dir,
        stem="rank_fraction_vs_precision",
        formats=formats,
        dpi=dpi,
    )
    plt.close(figure)


def print_configuration_summary(
    data: pd.DataFrame,
    *,
    detail_dimension: int,
    rank_degree: int,
) -> None:
    """Print the selected plotting configuration and compact diagnostics."""
    print("\nPrecision plotting configuration")
    print(
        "dimensions:",
        sorted(
            int(value)
            for value in data["dimension"].unique()
        ),
    )
    print(
        "degrees:",
        sorted(
            int(value)
            for value in data["degree"].unique()
        ),
    )
    print(
        "bases:",
        ordered_basis_values(data),
    )
    print(
        "dtypes:",
        ordered_dtype_values(data),
    )
    print(f"records: {len(data)}")
    print(
        f"detail dimension for degree curves: {detail_dimension}"
    )
    print(
        f"degree for rank-fraction curve: {rank_degree}"
    )

    if "perturbation_magnitude" in data.columns:
        magnitudes = sorted(
            float(value)
            for value in data[
                "perturbation_magnitude"
            ].unique()
        )
        print(
            "perturbation magnitudes:",
            magnitudes,
        )

        float32_epsilon = float(
            np.finfo(np.float32).eps
        )

        if any(
            magnitude <= float32_epsilon
            for magnitude in magnitudes
        ):
            print(
                "Warning: at least one perturbation magnitude is not "
                "larger than float32 machine epsilon "
                f"({float32_epsilon:.3e}). Some float32 perturbations "
                "may round away."
            )

    print("\nMedian relative error by dtype")
    error_table = (
        data.groupby("dtype", as_index=False)
        .agg(
            median_relative_error=(
                "relative_error",
                "median",
            ),
            q1_relative_error=(
                "relative_error",
                lambda values: values.quantile(0.25),
            ),
            q3_relative_error=(
                "relative_error",
                lambda values: values.quantile(0.75),
            ),
            maximum_relative_error=(
                "relative_error",
                "max",
            ),
        )
        .sort_values("dtype")
    )
    print(
        error_table.to_string(
            index=False,
        )
    )

    print("\nMinimum observed rank fraction")
    rank_table = (
        data.groupby(
            [
                "basis",
                "dtype",
                "dimension",
            ],
            as_index=False,
        )
        .agg(
            minimum_rank_fraction=(
                "rank_fraction",
                "min",
            ),
            minimum_rank=(
                "matrix_rank",
                "min",
            ),
            maximum_columns=(
                "matrix_columns",
                "max",
            ),
        )
        .sort_values(
            [
                "dimension",
                "dtype",
                "basis",
            ]
        )
    )
    print(
        rank_table.to_string(
            index=False,
        )
    )

    zero_sensitivity = (
        data.groupby(
            [
                "basis",
                "dtype",
            ],
            as_index=False,
        )
        .agg(
            zero_sensitivity_count=(
                "perturbation_sensitivity",
                lambda values: int(
                    np.count_nonzero(
                        np.asarray(values) == 0.0
                    )
                ),
            ),
            record_count=(
                "perturbation_sensitivity",
                "size",
            ),
        )
    )
    zero_sensitivity["zero_fraction"] = (
        zero_sensitivity["zero_sensitivity_count"]
        / zero_sensitivity["record_count"]
    )

    print("\nExactly zero perturbation sensitivities")
    print(
        zero_sensitivity.to_string(
            index=False,
        )
    )


def main() -> None:
    """Command-line entry point."""
    args = parse_args()

    if args.error_floor <= 0.0:
        raise ValueError(
            "--error-floor must be strictly positive."
        )

    if args.sensitivity_floor <= 0.0:
        raise ValueError(
            "--sensitivity-floor must be strictly positive."
        )

    if not args.input.exists():
        raise FileNotFoundError(
            f"Could not find precision benchmark CSV: {args.input}"
        )

    raw = pd.read_csv(args.input)

    data = canonicalise_column_names(raw)
    data = validate_and_clean(data)
    data = filter_data(
        data,
        dimensions=args.dimensions,
        degrees=args.degrees,
        bases=args.bases,
        dtypes=args.dtypes,
    )

    detail_dimension = resolve_detail_dimension(
        data,
        args.detail_dimension,
    )
    rank_degree = resolve_rank_degree(
        data,
        args.rank_degree,
    )

    print_configuration_summary(
        data,
        detail_dimension=detail_dimension,
        rank_degree=rank_degree,
    )

    plot_condition_number(
        data,
        detail_dimension=detail_dimension,
        output_dir=args.output_dir,
        formats=args.format,
        dpi=args.dpi,
    )

    plot_relative_error(
        data,
        detail_dimension=detail_dimension,
        output_dir=args.output_dir,
        formats=args.format,
        dpi=args.dpi,
        error_floor=args.error_floor,
    )

    plot_rank_fraction(
        data,
        degree=rank_degree,
        output_dir=args.output_dir,
        formats=args.format,
        dpi=args.dpi,
    )

    plot_perturbation_sensitivity(
        data,
        detail_dimension=detail_dimension,
        output_dir=args.output_dir,
        formats=args.format,
        dpi=args.dpi,
        sensitivity_floor=args.sensitivity_floor,
    )

    plot_runtime(
        data,
        detail_dimension=detail_dimension,
        output_dir=args.output_dir,
        formats=args.format,
        dpi=args.dpi,
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()