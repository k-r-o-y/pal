"""
Generate the Chapter 5 condition-number-versus-degree figure.

The script reads the detailed unit-simplex benchmark CSV, selects one
controlled configuration, aggregates repeated trials, and compares the
three polynomial basis representations.

Outputs:
    figures/chapter5/condition_number_vs_degree.pdf
    figures/chapter5/condition_number_vs_degree.png
    figures/chapter5/condition_number_vs_degree.svg
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
            "Plot estimated condition number against polynomial degree "
            "for the unit-simplex benchmark."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/simplex/unit_simplex_results.csv"
        ),
        help=(
            "Path to the detailed unit-simplex benchmark CSV."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/chapter5"),
        help=(
            "Directory in which the PDF, PNG, and SVG "
            "figures are saved."
        ),
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
    """
    Return the first matching column from candidate names.

    This permits small naming differences between benchmark-output
    versions without requiring the plotting script to be rewritten.
    """

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
    """Load the benchmark CSV and identify required columns."""

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
        "condition": find_column(
            dataframe,
            (
                "condition_number",
                "condition",
                "cond",
                "estimated_condition_number",
                "design_condition_number",
            ),
            "condition number",
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
    """Select one controlled benchmark configuration."""

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

    filtered[columns["condition"]] = pd.to_numeric(
        filtered[columns["condition"]],
        errors="coerce",
    )

    filtered = filtered.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    filtered = filtered.dropna(
        subset=[
            columns["degree"],
            columns["condition"],
        ]
    )

    filtered = filtered[
        filtered[columns["condition"]] > 0
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


def aggregate_trials(
    dataframe: pd.DataFrame,
    columns: dict[str, str],
) -> pd.DataFrame:
    """
    Aggregate repeated trials by basis and polynomial degree.

    The median is used as the central estimate because condition
    numbers may vary across several orders of magnitude. The
    interquartile range is retained as an uncertainty band.
    """

    grouped = (
        dataframe.groupby(
            [
                columns["basis"],
                columns["degree"],
            ],
            as_index=False,
        )[columns["condition"]]
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
                columns["basis"],
                columns["degree"],
            ]
        )
    )

    return grouped


def create_plot(
    summary: pd.DataFrame,
    basis_column: str,
    degree_column: str,
    output_directory: Path,
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
                "Warning: no records were available for "
                f"basis '{basis}'."
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
    axis.set_ylabel(
        "Estimated condition number (log scale)"
    )

    axis.set_title(
        "Estimated Condition Number versus Polynomial Degree",
        pad=14,
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
            numticks=12,
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

    legend = axis.legend(
        title="Polynomial basis",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
        fancybox=False,
    )

    legend.get_frame().set_alpha(0.95)

    pdf_path = (
        output_directory
        / "condition_number_vs_degree.pdf"
    )

    png_path = (
        output_directory
        / "condition_number_vs_degree.png"
    )

    svg_path = (
        output_directory
               / "condition_number_vs_degree.svg"
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
        "Condition-number figure generated successfully."
    )

    print(
        f"PDF: {pdf_path.resolve()}"
    )

    print(
        f"PNG: {png_path.resolve()}"
    )

    print(
        f"SVG: {svg_path.resolve()}"
    )


def main() -> None:
    """Run the plotting pipeline."""

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

    summary = aggregate_trials(
        dataframe=filtered,
        columns=columns,
    )

    print("Selected configuration")
    print("----------------------")
    print(
        f"Input:      {arguments.input.resolve()}"
    )
    print(
        f"Dimension:  {arguments.dimension}"
    )
    print(
        f"Scale:      {arguments.scale:g}"
    )
    print(
        f"Precision:  {arguments.dtype}"
    )
    print(
        f"Records:    {len(filtered)}"
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
    )


if __name__ == "__main__":
    main()