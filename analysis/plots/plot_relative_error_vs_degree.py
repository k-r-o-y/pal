"""
Generate the Chapter 5 relative-integration-error-versus-degree figure.

The script reads the detailed unit-simplex benchmark CSV, selects one
controlled configuration, aggregates repeated trials, and compares the
three polynomial basis representations.

Outputs:
    figures/chapter5/relative_error_vs_degree.pdf
    figures/chapter5/relative_error_vs_degree.png
    figures/chapter5/relative_error_vs_degree.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import pandas as pd


SUPPORTED_BASES = (
    "monomial",
    "legendre",
    "chebyshev",
)


PLOT_STYLE = {
    "monomial": {
        "label": "Monomial basis",
        "marker": "o",
        "linestyle": "-",
    },
    "legendre": {
        "label": "Legendre basis",
        "marker": "s",
        "linestyle": "--",
    },
    "chebyshev": {
        "label": "Chebyshev basis",
        "marker": "^",
        "linestyle": "-.",
    },
}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot relative integration error against polynomial degree "
            "for the unit-simplex benchmark."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/simplex/unit_simplex_results.csv"
        ),
        help="Path to the detailed unit-simplex benchmark CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/chapter5"),
        help="Directory in which the generated figures are saved.",
    )

    parser.add_argument(
        "--dimension",
        type=int,
        default=2,
        help="Simplex dimension to include.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Geometric scale to include.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        help="Floating-point precision to include.",
    )

    parser.add_argument(
        "--error-floor",
        type=float,
        default=1.0e-18,
        help=(
            "Positive floor used when plotting exact zero errors "
            "on a logarithmic axis."
        ),
    )

    return parser.parse_args()


def normalise_column_names(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Normalise CSV column names for reliable lookup."""

    dataframe = dataframe.copy()

    dataframe.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for column in dataframe.columns
    ]

    return dataframe


def find_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
    description: str,
) -> str:
    """Return the first matching column from candidate names."""

    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    raise KeyError(
        f"Could not find the {description} column.\n"
        f"Tried: {list(candidates)}\n"
        f"Available columns: {list(dataframe.columns)}"
    )


def load_results(
    csv_path: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load benchmark results and identify required columns."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Benchmark CSV not found:\n"
            f"{csv_path.resolve()}\n\n"
            "Run the unit-simplex benchmark first, or provide "
            "the correct path using --input."
        )

    dataframe = pd.read_csv(csv_path)
    dataframe = normalise_column_names(dataframe)

    columns = {
        "basis": find_column(
            dataframe,
            (
                "basis",
                "basis_name",
            ),
            "basis",
        ),
        "degree": find_column(
            dataframe,
            (
                "degree",
                "polynomial_degree",
                "total_degree",
            ),
            "polynomial degree",
        ),
        "dimension": find_column(
            dataframe,
            (
                "dimension",
                "dim",
                "simplex_dimension",
            ),
            "simplex dimension",
        ),
        "scale": find_column(
            dataframe,
            (
                "scale",
                "domain_scale",
                "simplex_scale",
                "geometric_scale",
                "constraint_scale",
            ),
            "scale",
        ),
        "dtype": find_column(
            dataframe,
            (
                "dtype",
                "precision",
                "floating_point_dtype",
            ),
            "floating-point precision",
        ),
        "relative_error": find_column(
            dataframe,
            (
                "relative_error",
                "rel_error",
                "relative_integration_error",
                "integration_relative_error",
                "relative_integral_error",
                "rel_err",
            ),
            "relative integration error",
        ),
    }

    return dataframe, columns


def filter_configuration(
    dataframe: pd.DataFrame,
    columns: dict[str, str],
    dimension: int,
    scale: float,
    dtype: str,
) -> pd.DataFrame:
    """Select one controlled experimental configuration."""

    dimension_values = pd.to_numeric(
        dataframe[columns["dimension"]],
        errors="coerce",
    )

    scale_values = pd.to_numeric(
        dataframe[columns["scale"]],
        errors="coerce",
    )

    dtype_values = (
        dataframe[columns["dtype"]]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    basis_values = (
        dataframe[columns["basis"]]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    filtered = dataframe[
        (dimension_values == dimension)
        & np.isclose(scale_values, scale)
        & dtype_values.str.contains(
            dtype.lower(),
            regex=False,
        )
        & basis_values.isin(SUPPORTED_BASES)
    ].copy()

    filtered[columns["basis"]] = (
        filtered[columns["basis"]]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    filtered[columns["degree"]] = pd.to_numeric(
        filtered[columns["degree"]],
        errors="coerce",
    )

    filtered[columns["relative_error"]] = pd.to_numeric(
        filtered[columns["relative_error"]],
        errors="coerce",
    )

    filtered = filtered.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    filtered = filtered.dropna(
        subset=[
            columns["degree"],
            columns["relative_error"],
        ]
    )

    # Relative errors should not be negative.
    filtered = filtered[
        filtered[columns["relative_error"]] >= 0
    ]

    if filtered.empty:
        available = (
            dataframe[
                [
                    columns["dimension"],
                    columns["scale"],
                    columns["dtype"],
                ]
            ]
            .drop_duplicates()
            .sort_values(
                [
                    columns["dimension"],
                    columns["scale"],
                    columns["dtype"],
                ]
            )
        )

        raise ValueError(
            "No records matched the requested configuration:\n"
            f"  dimension = {dimension}\n"
            f"  scale     = {scale}\n"
            f"  dtype     = {dtype}\n\n"
            "Available configurations include:\n"
            f"{available.head(20).to_string(index=False)}"
        )

    return filtered


def apply_error_floor(
    dataframe: pd.DataFrame,
    error_column: str,
    error_floor: float,
) -> pd.DataFrame:
    """
    Replace exact zero errors with a small positive plotting floor.

    This affects only the visualisation. The original benchmark values
    remain unchanged in the CSV.
    """

    if not np.isfinite(error_floor) or error_floor <= 0:
        raise ValueError(
            "--error-floor must be a finite positive number."
        )

    dataframe = dataframe.copy()

    zero_count = int(
        (dataframe[error_column] == 0).sum()
    )

    dataframe["plot_relative_error"] = dataframe[
        error_column
    ].clip(lower=error_floor)

    if zero_count > 0:
        print(
            f"Note: {zero_count} exact zero errors were displayed at "
            f"the plotting floor {error_floor:.1e}."
        )

    return dataframe


def aggregate_trials(
    dataframe: pd.DataFrame,
    basis_column: str,
    degree_column: str,
) -> pd.DataFrame:
    """
    Aggregate repeated trials by basis and polynomial degree.

    The median is used as the central estimate. The interquartile range
    is shown as an uncertainty band.
    """

    grouped = (
        dataframe.groupby(
            [
                basis_column,
                degree_column,
            ],
            as_index=False,
        )["plot_relative_error"]
        .agg(
            median="median",
            lower_quartile=lambda values: values.quantile(
                0.25
            ),
            upper_quartile=lambda values: values.quantile(
                0.75
            ),
            minimum="min",
            maximum="max",
            count="count",
        )
        .sort_values(
            [
                basis_column,
                degree_column,
            ]
        )
    )

    return grouped


def create_plot(
    summary: pd.DataFrame,
    basis_column: str,
    degree_column: str,
    output_directory: Path,
    error_floor: float,
) -> None:
    """Create and save the dissertation-quality figure."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(8.6, 5.4),
        constrained_layout=True,
    )

    plotted_basis_count = 0

    for basis in SUPPORTED_BASES:
        basis_data = summary[
            summary[basis_column] == basis
        ].sort_values(degree_column)

        if basis_data.empty:
            print(
                f"Warning: no records were available for basis "
                f"'{basis}'."
            )
            continue

        plotted_basis_count += 1

        degrees = basis_data[
            degree_column
        ].to_numpy(dtype=float)

        medians = basis_data[
            "median"
        ].to_numpy(dtype=float)

        lower = basis_data[
            "lower_quartile"
        ].to_numpy(dtype=float)

        upper = basis_data[
            "upper_quartile"
        ].to_numpy(dtype=float)

        style = PLOT_STYLE[basis]

        line = axis.plot(
            degrees,
            medians,
            label=style["label"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.6,
            markersize=7.5,
            markerfacecolor="white",
            markeredgewidth=1.5,
        )[0]

        axis.fill_between(
            degrees,
            lower,
            upper,
            alpha=0.18,
            color=line.get_color(),
            linewidth=0,
        )

    if plotted_basis_count == 0:
        raise ValueError(
            "No supported basis data were available to plot."
        )

    axis.set_yscale("log")

    axis.set_xlabel("Polynomial degree")
    axis.set_ylabel("Relative integration error (log scale)")

    axis.set_title(
        "Relative Integration Error versus Polynomial Degree",
        pad=12,
    )

    axis.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.7,
        alpha=0.5,
    )

    axis.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.4,
        alpha=0.25,
    )

    axis.yaxis.set_major_locator(
        LogLocator(
            base=10.0,
            numticks=16,
        )
    )

    axis.yaxis.set_minor_locator(
        LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1,
            numticks=100,
        )
    )

    degree_values = sorted(
        summary[degree_column].unique()
    )

    axis.set_xticks(degree_values)

    minimum_degree = min(degree_values)
    maximum_degree = max(degree_values)

    axis.set_xlim(
        minimum_degree - 0.25,
        maximum_degree + 0.25,
    )

    plotted_values = summary[
        [
            "median",
            "lower_quartile",
            "upper_quartile",
        ]
    ].to_numpy(dtype=float)

    finite_positive_values = plotted_values[
        np.isfinite(plotted_values)
        & (plotted_values > 0)
    ]

    if finite_positive_values.size > 0:
        minimum_error = max(
            error_floor,
            float(finite_positive_values.min()),
        )

        maximum_error = float(
            finite_positive_values.max()
        )

        lower_limit = 10.0 ** (
            np.floor(np.log10(minimum_error)) - 0.25
        )

        upper_limit = 10.0 ** (
            np.ceil(np.log10(maximum_error)) + 0.25
        )

        axis.set_ylim(
            lower_limit,
            upper_limit,
        )

    legend = axis.legend(
        title="Polynomial basis",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        frameon=True,
        fancybox=False,
    )

    legend.get_frame().set_alpha(0.95)

    pdf_path = (
        output_directory
        / "relative_error_vs_degree.pdf"
    )

    png_path = (
        output_directory
        / "relative_error_vs_degree.png"
    )

    svg_path = (
        output_directory
        / "relative_error_vs_degree.svg"
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    figure.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    figure.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        "Relative-integration-error figure generated successfully."
    )
    print(f"PDF: {pdf_path.resolve()}")
    print(f"PNG: {png_path.resolve()}")
    print(f"SVG: {svg_path.resolve()}")


def main() -> None:
    """Run the complete plotting pipeline."""

    arguments = parse_arguments()

    dataframe, columns = load_results(
        arguments.input
    )

    filtered = filter_configuration(
        dataframe=dataframe,
        columns=columns,
        dimension=arguments.dimension,
        scale=arguments.scale,
        dtype=arguments.dtype,
    )

    filtered = apply_error_floor(
        dataframe=filtered,
        error_column=columns["relative_error"],
        error_floor=arguments.error_floor,
    )

    summary = aggregate_trials(
        dataframe=filtered,
        basis_column=columns["basis"],
        degree_column=columns["degree"],
    )

    print("Selected configuration")
    print("----------------------")
    print(
        f"Input:       {arguments.input.resolve()}"
    )
    print(
        f"Dimension:   {arguments.dimension}"
    )
    print(
        f"Scale:       {arguments.scale:g}"
    )
    print(
        f"Precision:   {arguments.dtype}"
    )
    print(
        f"Error floor: {arguments.error_floor:.1e}"
    )
    print(
        f"Records:     {len(filtered)}"
    )
    print()
    print(
        summary.to_string(index=False)
    )
    print()

    create_plot(
        summary=summary,
        basis_column=columns["basis"],
        degree_column=columns["degree"],
        output_directory=arguments.output_dir,
        error_floor=arguments.error_floor,
    )


if __name__ == "__main__":
    main()