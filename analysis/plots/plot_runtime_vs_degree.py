"""
Generate the Chapter 5 runtime-versus-polynomial-degree figure.

The script reads the detailed unit-simplex benchmark CSV, selects one
controlled configuration, aggregates repeated trials, and compares the
execution time of the three polynomial basis representations.

The benchmark CSV is assumed to store runtime in seconds. By default,
the figure displays runtime in milliseconds.

Outputs:
    figures/chapter5/runtime_vs_degree.pdf
    figures/chapter5/runtime_vs_degree.png
    figures/chapter5/runtime_vs_degree.svg
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


RUNTIME_UNITS = {
    "seconds": {
        "factor": 1.0,
        "axis_label": "Execution time (seconds)",
    },
    "milliseconds": {
        "factor": 1.0e3,
        "axis_label": "Execution time (milliseconds)",
    },
    "microseconds": {
        "factor": 1.0e6,
        "axis_label": "Execution time (microseconds)",
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
            "Plot execution time against polynomial degree for the "
            "unit-simplex benchmark."
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
        help=(
            "Directory in which the PDF, PNG, and SVG figures "
            "are saved."
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

    parser.add_argument(
        "--runtime-unit",
        choices=tuple(RUNTIME_UNITS),
        default="milliseconds",
        help="Unit used to display execution time.",
    )

    parser.add_argument(
        "--runtime-floor",
        type=float,
        default=1.0e-12,
        help=(
            "Positive floor in seconds used for exact-zero runtime "
            "values on a logarithmic axis."
        ),
    )

    parser.add_argument(
        "--log-y-axis",
        action="store_true",
        help=(
            "Force a logarithmic y-axis. By default, the script chooses "
            "a logarithmic axis only when runtime spans at least one "
            "order of magnitude."
        ),
    )

    parser.add_argument(
        "--linear-y-axis",
        action="store_true",
        help=(
            "Force a linear y-axis. This cannot be combined with "
            "--log-y-axis."
        ),
    )

    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Validate mutually exclusive and numeric arguments."""

    if arguments.log_y_axis and arguments.linear_y_axis:
        raise ValueError(
            "--log-y-axis and --linear-y-axis cannot be used together."
        )

    if (
        not np.isfinite(arguments.runtime_floor)
        or arguments.runtime_floor <= 0
    ):
        raise ValueError(
            "--runtime-floor must be a finite positive number."
        )


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
    Return the first matching CSV column.

    The candidate list permits minor naming differences between benchmark
    output versions.
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
    """Load benchmark data and identify the required columns."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Benchmark CSV not found:\n"
            f"{csv_path.resolve()}\n\n"
            "Run the unit-simplex benchmark first, or provide "
            "the correct path with --input."
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
        "runtime": find_column(
            dataframe,
            (
                "runtime_seconds",
                "runtime_s",
                "runtime",
                "elapsed_seconds",
                "elapsed_time_seconds",
                "execution_time_seconds",
                "integration_runtime_seconds",
                "basis_runtime_seconds",
                "elapsed_time",
                "execution_time",
                "time_seconds",
                "duration_seconds",
            ),
            "runtime",
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
        & np.isclose(
            scale_values,
            scale,
            rtol=1.0e-9,
            atol=1.0e-12,
        )
        & dtype_values.str.contains(
            dtype.strip().lower(),
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

    filtered[columns["runtime"]] = pd.to_numeric(
        filtered[columns["runtime"]],
        errors="coerce",
    )

    filtered = filtered.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    filtered = filtered.dropna(
        subset=[
            columns["degree"],
            columns["runtime"],
        ]
    )

    filtered = filtered[
        filtered[columns["runtime"]] >= 0
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
            f"{available.head(30).to_string(index=False)}"
        )

    return filtered


def prepare_runtime_values(
    dataframe: pd.DataFrame,
    runtime_column: str,
    runtime_unit: str,
    runtime_floor_seconds: float,
) -> tuple[pd.DataFrame, int, float, str]:
    """
    Convert runtime values from seconds into the requested display unit.

    Exact-zero values are replaced only in the in-memory plotting data.
    The source CSV remains unchanged.
    """

    unit_configuration = RUNTIME_UNITS[runtime_unit]

    conversion_factor = float(
        unit_configuration["factor"]
    )

    axis_label = str(
        unit_configuration["axis_label"]
    )

    prepared = dataframe.copy()

    zero_mask = prepared[runtime_column] == 0
    zero_count = int(zero_mask.sum())

    runtime_seconds = prepared[
        runtime_column
    ].clip(lower=runtime_floor_seconds)

    prepared["plot_runtime"] = (
        runtime_seconds * conversion_factor
    )

    display_floor = (
        runtime_floor_seconds * conversion_factor
    )

    return (
        prepared,
        zero_count,
        display_floor,
        axis_label,
    )


def aggregate_trials(
    dataframe: pd.DataFrame,
    basis_column: str,
    degree_column: str,
) -> pd.DataFrame:
    """
    Aggregate repeated runtime measurements.

    The median is used as the principal estimate. The interquartile range
    is retained as an uncertainty band because runtime may be affected by
    operating-system scheduling and other transient activity.
    """

    grouped = (
        dataframe.groupby(
            [
                basis_column,
                degree_column,
            ],
            as_index=False,
        )["plot_runtime"]
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
            mean="mean",
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


def automatic_log_axis_choice(
    summary: pd.DataFrame,
) -> bool:
    """
    Decide whether the runtime range warrants a logarithmic axis.

    A logarithmic scale is selected when the plotted values span at
    least one order of magnitude.
    """

    values = summary[
        [
            "median",
            "lower_quartile",
            "upper_quartile",
        ]
    ].to_numpy(dtype=float)

    finite_positive = values[
        np.isfinite(values)
        & (values > 0)
    ]

    if finite_positive.size == 0:
        return False

    minimum_value = float(
        finite_positive.min()
    )

    maximum_value = float(
        finite_positive.max()
    )

    if minimum_value <= 0:
        return False

    return maximum_value / minimum_value >= 10.0


def determine_axis_type(
    summary: pd.DataFrame,
    force_log: bool,
    force_linear: bool,
) -> bool:
    """Return True when the plot should use a logarithmic y-axis."""

    if force_log:
        return True

    if force_linear:
        return False

    return automatic_log_axis_choice(summary)


def configure_logarithmic_axis(
    axis: plt.Axes,
    summary: pd.DataFrame,
    display_floor: float,
    axis_label: str,
) -> None:
    """Configure a logarithmic runtime axis."""

    axis.set_yscale("log")

    axis.set_ylabel(
        f"{axis_label} (log scale)"
    )

    axis.yaxis.set_major_locator(
        LogLocator(
            base=10.0,
            numticks=14,
        )
    )

    axis.yaxis.set_minor_locator(
        LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1,
            numticks=100,
        )
    )

    values = summary[
        [
            "median",
            "lower_quartile",
            "upper_quartile",
        ]
    ].to_numpy(dtype=float)

    finite_positive = values[
        np.isfinite(values)
        & (values > 0)
    ]

    if finite_positive.size == 0:
        return

    minimum_value = max(
        display_floor,
        float(finite_positive.min()),
    )

    maximum_value = float(
        finite_positive.max()
    )

    lower_limit = 10.0 ** np.floor(
        np.log10(minimum_value)
    )

    upper_limit = 10.0 ** np.ceil(
        np.log10(maximum_value)
    )

    if np.isclose(
        lower_limit,
        upper_limit,
        rtol=1.0e-12,
        atol=0.0,
    ):
        lower_limit /= 10.0
        upper_limit *= 10.0

    axis.set_ylim(
        lower_limit / 1.25,
        upper_limit * 1.25,
    )


def configure_linear_axis(
    axis: plt.Axes,
    summary: pd.DataFrame,
    axis_label: str,
) -> None:
    """Configure a linear runtime axis."""

    axis.set_ylabel(axis_label)

    values = summary[
        [
            "median",
            "lower_quartile",
            "upper_quartile",
        ]
    ].to_numpy(dtype=float)

    finite_values = values[
        np.isfinite(values)
    ]

    if finite_values.size == 0:
        return

    minimum_value = float(
        finite_values.min()
    )

    maximum_value = float(
        finite_values.max()
    )

    value_range = maximum_value - minimum_value

    if value_range <= 0:
        padding = max(
            abs(maximum_value) * 0.1,
            0.01,
        )
    else:
        padding = value_range * 0.08

    lower_limit = max(
        0.0,
        minimum_value - padding,
    )

    upper_limit = maximum_value + padding

    if upper_limit <= lower_limit:
        upper_limit = lower_limit + 1.0

    axis.set_ylim(
        lower_limit,
        upper_limit,
    )


def create_plot(
    summary: pd.DataFrame,
    basis_column: str,
    degree_column: str,
    output_directory: Path,
    axis_label: str,
    display_floor: float,
    zero_count: int,
    use_log_axis: bool,
) -> None:
    """Create and save the dissertation-quality runtime figure."""

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

    axis.set_xlabel("Polynomial degree")

    axis.set_title(
        "Execution Time versus Polynomial Degree",
        pad=12,
    )

    if use_log_axis:
        configure_logarithmic_axis(
            axis=axis,
            summary=summary,
            display_floor=display_floor,
            axis_label=axis_label,
        )
    else:
        configure_linear_axis(
            axis=axis,
            summary=summary,
            axis_label=axis_label,
        )

    axis.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.7,
        alpha=0.5,
    )

    if use_log_axis:
        axis.grid(
            True,
            which="minor",
            linestyle=":",
            linewidth=0.4,
            alpha=0.25,
        )

    degree_values = sorted(
        summary[degree_column].unique()
    )

    axis.set_xticks(degree_values)

    minimum_degree = float(
        min(degree_values)
    )

    maximum_degree = float(
        max(degree_values)
    )

    axis.set_xlim(
        minimum_degree - 0.25,
        maximum_degree + 0.25,
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

    if zero_count > 0:
        axis.text(
            0.01,
            0.02,
            (
                f"{zero_count} exact-zero runtime value(s) "
                f"displayed at {display_floor:.2e}"
            ),
            transform=axis.transAxes,
            horizontalalignment="left",
            verticalalignment="bottom",
            fontsize=9,
        )

    pdf_path = (
        output_directory
        / "runtime_vs_degree.pdf"
    )

    png_path = (
        output_directory
        / "runtime_vs_degree.png"
    )

    svg_path = (
        output_directory
        / "runtime_vs_degree.svg"
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

    print("Runtime figure generated successfully.")
    print(f"PDF: {pdf_path.resolve()}")
    print(f"PNG: {png_path.resolve()}")
    print(f"SVG: {svg_path.resolve()}")
    print(
        "Y-axis scale: "
        f"{'logarithmic' if use_log_axis else 'linear'}"
    )


def main() -> None:
    """Run the complete plotting pipeline."""

    arguments = parse_arguments()
    validate_arguments(arguments)

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

    (
        plotting_data,
        zero_count,
        display_floor,
        axis_label,
    ) = prepare_runtime_values(
        dataframe=filtered,
        runtime_column=columns["runtime"],
        runtime_unit=arguments.runtime_unit,
        runtime_floor_seconds=arguments.runtime_floor,
    )

    summary = aggregate_trials(
        dataframe=plotting_data,
        basis_column=columns["basis"],
        degree_column=columns["degree"],
    )

    use_log_axis = determine_axis_type(
        summary=summary,
        force_log=arguments.log_y_axis,
        force_linear=arguments.linear_y_axis,
    )

    print("Selected configuration")
    print("----------------------")
    print(
        f"Input:         {arguments.input.resolve()}"
    )
    print(
        f"Runtime field: {columns['runtime']}"
    )
    print(
        f"Dimension:     {arguments.dimension}"
    )
    print(
        f"Scale:         {arguments.scale:g}"
    )
    print(
        f"Precision:     {arguments.dtype}"
    )
    print(
        f"Display unit:  {arguments.runtime_unit}"
    )
    print(
        f"Records:       {len(filtered)}"
    )
    print(
        f"Zero runtimes: {zero_count}"
    )
    print(
        "Y-axis:        "
        f"{'logarithmic' if use_log_axis else 'linear'}"
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
        axis_label=axis_label,
        display_floor=display_floor,
        zero_count=zero_count,
        use_log_axis=use_log_axis,
    )


if __name__ == "__main__":
    main()