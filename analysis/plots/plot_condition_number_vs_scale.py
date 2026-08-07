#!/usr/bin/env python3
"""
Plot estimated condition number versus geometric scale.

The script reads the geometric-scale benchmark output, filters one controlled
configuration, aggregates repeated trials using the median and interquartile
range, and writes publication-quality PDF, PNG, and SVG figures.

Expected input
--------------
results/simplex/geometric_scale_results.csv

Default outputs
---------------
figures/chapter5/condition_number_vs_scale.pdf
figures/chapter5/condition_number_vs_scale.png
figures/chapter5/condition_number_vs_scale.svg

Run from the repository root
----------------------------
    python -m analysis.plots.plot_condition_number_vs_scale

Optional example
----------------
    python -m analysis.plots.plot_condition_number_vs_scale \
        --input results/simplex/geometric_scale_results.csv \
        --output-directory figures/chapter5 \
        --degree 10 \
        --dimension 2 \
        --dtype float64

Notes
-----
- Both axes use logarithmic scaling because scale and condition number span
  several orders of magnitude.
- Lines show the median across repeated trials.
- Shaded regions show the interquartile range.
- The script accepts several plausible column-name variants so that it remains
  compatible with minor changes to the benchmark CSV schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASIS_ORDER = ("monomial", "legendre", "chebyshev")

BASIS_LABELS = {
    "monomial": "Monomial basis",
    "legendre": "Legendre basis",
    "chebyshev": "Chebyshev basis",
}

COLUMN_CANDIDATES = {
    "basis": (
        "basis",
        "polynomial_basis",
        "basis_name",
    ),
    "degree": (
        "degree",
        "polynomial_degree",
        "total_degree",
    ),
    "dimension": (
        "dimension",
        "simplex_dimension",
        "dim",
    ),
    "scale": (
        "scale",
        "geometric_scale",
        "simplex_scale",
        "domain_scale",
    ),
    "dtype": (
        "dtype",
        "precision",
        "floating_point_precision",
    ),
    "condition_number": (
        "condition_number",
        "estimated_condition_number",
        "condition",
        "cond",
    ),
}


def normalise_name(name: str) -> str:
    """Return a normalised identifier for tolerant column matching."""
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def find_column(
    dataframe: pd.DataFrame,
    logical_name: str,
    *,
    required: bool = True,
) -> str | None:
    """Locate a CSV column using the configured candidate names."""
    normalised_columns = {
        normalise_name(column): column for column in dataframe.columns
    }

    for candidate in COLUMN_CANDIDATES[logical_name]:
        match = normalised_columns.get(normalise_name(candidate))
        if match is not None:
            return match

    if required:
        candidates = ", ".join(COLUMN_CANDIDATES[logical_name])
        available = ", ".join(map(str, dataframe.columns))
        raise KeyError(
            f"Could not find the {logical_name!r} column. "
            f"Expected one of: {candidates}. "
            f"Available columns: {available}"
        )

    return None


def canonical_basis(value: object) -> str:
    """Normalise basis labels used by the benchmark."""
    text = normalise_name(str(value))

    aliases = {
        "monomial": "monomial",
        "monomial_basis": "monomial",
        "power": "monomial",
        "power_basis": "monomial",
        "legendre": "legendre",
        "legendre_basis": "legendre",
        "chebyshev": "chebyshev",
        "chebyshev_basis": "chebyshev",
    }

    return aliases.get(text, text)


def canonical_dtype(value: object) -> str:
    """Normalise floating-point type labels."""
    text = normalise_name(str(value))

    aliases = {
        "float": "float64",
        "double": "float64",
        "np.float64": "float64",
        "numpy.float64": "float64",
        "torch.float64": "float64",
        "single": "float32",
        "np.float32": "float32",
        "numpy.float32": "float32",
        "torch.float32": "float32",
    }

    return aliases.get(text, text)


def finite_numeric(series: pd.Series, name: str) -> pd.Series:
    """Convert a series to numeric values and reject non-finite entries."""
    numeric = pd.to_numeric(series, errors="coerce")

    invalid = numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype=float))
    if invalid.any():
        count = int(invalid.sum())
        raise ValueError(
            f"Column {name!r} contains {count} non-numeric or non-finite values."
        )

    return numeric.astype(float)


def load_and_filter(
    *,
    input_path: Path,
    degree: int,
    dimension: int,
    dtype: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load the benchmark CSV and select the requested configuration."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Run the geometric-scale benchmark first."
        )

    dataframe = pd.read_csv(input_path)

    if dataframe.empty:
        raise ValueError(f"Input CSV is empty: {input_path}")

    columns = {
        name: find_column(dataframe, name)
        for name in COLUMN_CANDIDATES
    }

    working = dataframe.copy()

    working["_basis"] = working[columns["basis"]].map(canonical_basis)
    working["_degree"] = finite_numeric(
        working[columns["degree"]],
        columns["degree"],
    ).astype(int)
    working["_dimension"] = finite_numeric(
        working[columns["dimension"]],
        columns["dimension"],
    ).astype(int)
    working["_scale"] = finite_numeric(
        working[columns["scale"]],
        columns["scale"],
    )
    working["_condition_number"] = finite_numeric(
        working[columns["condition_number"]],
        columns["condition_number"],
    )

    dtype_column = columns["dtype"]
    working["_dtype"] = working[dtype_column].map(canonical_dtype)

    requested_dtype = canonical_dtype(dtype)

    selected = working[
        (working["_degree"] == degree)
        & (working["_dimension"] == dimension)
        & (working["_dtype"] == requested_dtype)
        & (working["_basis"].isin(BASIS_ORDER))
    ].copy()

    if selected.empty:
        available = (
            working[
                ["_degree", "_dimension", "_dtype"]
            ]
            .drop_duplicates()
            .sort_values(["_degree", "_dimension", "_dtype"])
        )
        raise ValueError(
            "No rows match the requested configuration:\n"
            f"  degree={degree}\n"
            f"  dimension={dimension}\n"
            f"  dtype={requested_dtype}\n\n"
            "Available configurations:\n"
            f"{available.to_string(index=False)}"
        )

    if (selected["_scale"] <= 0).any():
        raise ValueError("All geometric scales must be positive for a log axis.")

    if (selected["_condition_number"] <= 0).any():
        raise ValueError(
            "All condition numbers must be positive for a log axis."
        )

    return selected, columns


def aggregate_results(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Compute median, first quartile, third quartile, and trial count."""
    aggregated = (
        dataframe.groupby(
            ["_basis", "_scale"],
            as_index=False,
            observed=True,
        )["_condition_number"]
        .agg(
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
            count="count",
        )
        .sort_values(["_basis", "_scale"])
    )

    return aggregated


def print_summary_table(aggregated: pd.DataFrame) -> None:
    """Print a compact publication-check table to the terminal."""
    table = aggregated.copy()
    table = table.rename(
        columns={
            "_basis": "basis",
            "_scale": "scale",
        }
    )

    print("\nAggregated condition-number results")
    print("-----------------------------------")
    print(
        table[
            ["basis", "scale", "median", "q1", "q3", "count"]
        ].to_string(
            index=False,
            formatters={
                "scale": lambda value: f"{value:g}",
                "median": lambda value: f"{value:.6e}",
                "q1": lambda value: f"{value:.6e}",
                "q3": lambda value: f"{value:.6e}",
            },
        )
    )


def plot_condition_number(
    *,
    aggregated: pd.DataFrame,
    degree: int,
    dimension: int,
    dtype: str,
    output_directory: Path,
    filename_stem: str,
    dpi: int,
) -> list[Path]:
    """Create and save the condition-number figure."""
    output_directory.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8.2, 5.2))

    plotted_bases = 0

    for basis in BASIS_ORDER:
        subset = aggregated[aggregated["_basis"] == basis].sort_values(
            "_scale"
        )

        if subset.empty:
            continue

        scales = subset["_scale"].to_numpy(dtype=float)
        medians = subset["median"].to_numpy(dtype=float)
        q1 = subset["q1"].to_numpy(dtype=float)
        q3 = subset["q3"].to_numpy(dtype=float)

        line = axis.plot(
            scales,
            medians,
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            label=BASIS_LABELS[basis],
        )[0]

        axis.fill_between(
            scales,
            q1,
            q3,
            alpha=0.18,
            color=line.get_color(),
            linewidth=0,
        )

        plotted_bases += 1

    if plotted_bases == 0:
        plt.close(figure)
        raise ValueError("No recognised basis rows were available to plot.")

    axis.set_xscale("log")
    axis.set_yscale("log")

    axis.set_xlabel("Geometric scale")
    axis.set_ylabel("Estimated condition number (log scale)")
    axis.set_title("Estimated Condition Number versus Geometric Scale")

    axis.grid(True, which="major", linewidth=0.7, alpha=0.35)
    axis.grid(True, which="minor", linewidth=0.4, alpha=0.15)

    axis.legend(
        title="Polynomial basis",
        loc="best",
        frameon=True,
    )

    figure.text(
        0.5,
        0.005,
        (
            f"Fixed configuration: degree {degree}, dimension {dimension}, "
            f"{canonical_dtype(dtype)}"
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

    figure.tight_layout(rect=(0, 0.025, 1, 1))

    output_paths = [
        output_directory / f"{filename_stem}.pdf",
        output_directory / f"{filename_stem}.png",
        output_directory / f"{filename_stem}.svg",
    ]

    figure.savefig(
        output_paths[0],
        bbox_inches="tight",
    )
    figure.savefig(
        output_paths[1],
        dpi=dpi,
        bbox_inches="tight",
    )
    figure.savefig(
        output_paths[2],
        bbox_inches="tight",
    )

    plt.close(figure)
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot median estimated condition number and interquartile range "
            "against geometric simplex scale."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/simplex/geometric_scale_results.csv"
        ),
        help="Path to the geometric-scale benchmark CSV.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("figures/chapter5"),
        help="Directory for generated PDF, PNG, and SVG files.",
    )

    parser.add_argument(
        "--filename-stem",
        default="condition_number_vs_scale",
        help="Output filename without an extension.",
    )

    parser.add_argument(
        "--degree",
        type=int,
        default=10,
        help="Fixed polynomial degree to plot.",
    )

    parser.add_argument(
        "--dimension",
        type=int,
        default=2,
        help="Fixed simplex dimension to plot.",
    )

    parser.add_argument(
        "--dtype",
        default="float64",
        help="Fixed floating-point precision to plot.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution of the PNG output.",
    )

    return parser


def main() -> int:
    """Load, aggregate, plot, and report the selected results."""
    arguments = build_parser().parse_args()

    if arguments.degree < 0:
        raise ValueError("--degree must be non-negative.")

    if arguments.dimension <= 0:
        raise ValueError("--dimension must be positive.")

    if arguments.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    selected, columns = load_and_filter(
        input_path=arguments.input,
        degree=arguments.degree,
        dimension=arguments.dimension,
        dtype=arguments.dtype,
    )

    aggregated = aggregate_results(selected)

    print("Condition number versus geometric scale")
    print("----------------------------------------")
    print(f"Input:       {arguments.input}")
    print(f"Degree:      {arguments.degree}")
    print(f"Dimension:   {arguments.dimension}")
    print(f"Precision:   {canonical_dtype(arguments.dtype)}")
    print(f"Rows:        {len(selected)}")
    print(f"Scales:      {selected['_scale'].nunique()}")
    print(f"Bases:       {selected['_basis'].nunique()}")
    print(f"Condition column: {columns['condition_number']}")

    print_summary_table(aggregated)

    output_paths = plot_condition_number(
        aggregated=aggregated,
        degree=arguments.degree,
        dimension=arguments.dimension,
        dtype=arguments.dtype,
        output_directory=arguments.output_directory,
        filename_stem=arguments.filename_stem,
        dpi=arguments.dpi,
    )

    print("\nGenerated files")
    print("---------------")
    for path in output_paths:
        print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
