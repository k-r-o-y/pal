#!/usr/bin/env python3
"""
Plot constraint-induced Gram-matrix conditioning results.

Reads:
    results/conditioning/constraint_gram_results.csv

Writes:
    figures/constraint_gram_condition_number_vs_degree.pdf
    figures/constraint_gram_condition_number_vs_degree.png
    figures/constraint_gram_condition_number_vs_degree.svg

The figure contains two panels:
    (a) unit simplex
    (b) box with obstacle

Each panel compares monomial, Legendre, and Chebyshev values of

    kappa_2(G_C)

as polynomial degree increases.

Infinite condition numbers are preserved in the source CSV. For plotting only,
they are placed at a finite ceiling above the largest finite value and marked
with an "x" so that numerical singularity remains visually explicit.

Example
-------
Run from the repository root:

    python -m analysis.plots.plot_constraint_gram_benchmark \
        --format pdf png svg
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_INPUT = Path(
    "results/conditioning/constraint_gram_results.csv"
)

DEFAULT_OUTPUT_DIR = Path("figures")

DEFAULT_OUTPUT_STEM = (
    "constraint_gram_condition_number_vs_degree"
)

DEFAULT_DIMENSION = 2

DEFAULT_DOMAINS = (
    "unit_simplex",
    "box_with_obstacle",
)

BASIS_ORDER = (
    "monomial",
    "legendre",
    "chebyshev",
)

BASIS_LABELS = {
    "monomial": "Monomial",
    "legendre": "Legendre",
    "chebyshev": "Chebyshev",
}

DOMAIN_LABELS = {
    "unit_simplex": "(a) Unit simplex",
    "box_with_obstacle": "(b) Box with obstacle",
}

MARKERS = {
    "monomial": "o",
    "legendre": "s",
    "chebyshev": "^",
}

LINESTYLES = {
    "monomial": "-",
    "legendre": "--",
    "chebyshev": "-.",
}

SUPPORTED_FORMATS = (
    "pdf",
    "png",
    "svg",
)


# =============================================================================
# Validation
# =============================================================================

def validate_columns(
    dataframe: pd.DataFrame,
) -> None:
    required = {
        "domain",
        "dimension",
        "degree",
        "basis",
        "condition_number",
        "numerical_rank_fraction",
        "basis_count",
    }

    missing = required.difference(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            "input CSV is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )


def validate_basis_values(
    dataframe: pd.DataFrame,
) -> None:
    present = set(
        dataframe["basis"].astype(str)
    )

    unknown = present.difference(
        BASIS_ORDER
    )

    if unknown:
        raise ValueError(
            "unknown basis values in input: "
            + ", ".join(
                sorted(unknown)
            )
        )


# =============================================================================
# Helpers
# =============================================================================

def finite_condition_values(
    dataframe: pd.DataFrame,
) -> np.ndarray:
    values = pd.to_numeric(
        dataframe["condition_number"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    return values[
        np.isfinite(values)
        & (values > 0.0)
    ]


def compute_plot_ceiling(
    dataframe: pd.DataFrame,
) -> float:
    """
    Compute a finite plotting height used only for configurations with
    condition_number == inf.

    The ceiling is one decade above the largest finite condition number.
    """
    finite = finite_condition_values(
        dataframe
    )

    if finite.size == 0:
        return 10.0

    maximum = float(
        np.max(finite)
    )

    exponent = math.ceil(
        math.log10(maximum)
    )

    return 10.0 ** (
        exponent + 1
    )


def display_condition_values(
    condition_values: Sequence[float],
    ceiling: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replace infinite values with the plotting ceiling.

    Returns:
        displayed_values
        infinite_mask
    """
    values = np.asarray(
        condition_values,
        dtype=float,
    )

    infinite_mask = ~np.isfinite(
        values
    )

    displayed = values.copy()

    displayed[
        infinite_mask
    ] = ceiling

    return (
        displayed,
        infinite_mask,
    )


def load_results(
    path: Path,
) -> pd.DataFrame:
    dataframe = pd.read_csv(
        path
    )

    validate_columns(
        dataframe
    )

    validate_basis_values(
        dataframe
    )

    dataframe["dimension"] = pd.to_numeric(
        dataframe["dimension"],
        errors="raise",
    ).astype(int)

    dataframe["degree"] = pd.to_numeric(
        dataframe["degree"],
        errors="raise",
    ).astype(int)

    dataframe["condition_number"] = pd.to_numeric(
        dataframe["condition_number"],
        errors="coerce",
    )

    dataframe["numerical_rank_fraction"] = (
        pd.to_numeric(
            dataframe[
                "numerical_rank_fraction"
            ],
            errors="raise",
        )
    )

    return dataframe


def filter_results(
    dataframe: pd.DataFrame,
    *,
    dimension: int,
    domains: Sequence[str],
) -> pd.DataFrame:
    filtered = dataframe[
        (
            dataframe["dimension"]
            == dimension
        )
        & (
            dataframe["domain"]
            .isin(domains)
        )
    ].copy()

    if filtered.empty:
        raise ValueError(
            "no matching results found for "
            f"dimension={dimension}, "
            f"domains={list(domains)}"
        )

    return filtered


# =============================================================================
# Main condition-number figure
# =============================================================================

def plot_condition_number_figure(
    dataframe: pd.DataFrame,
    *,
    dimension: int,
    domains: Sequence[str],
) -> plt.Figure:
    """
    Create the main two-panel thesis figure.
    """
    ceiling = compute_plot_ceiling(
        dataframe
    )

    figure, axes = plt.subplots(
        1,
        len(domains),
        figsize=(
            11.0,
            4.4,
        ),
        sharey=True,
        constrained_layout=True,
    )

    if len(domains) == 1:
        axes = np.asarray(
            [axes]
        )

    for axis, domain in zip(
        axes,
        domains,
    ):
        domain_data = dataframe[
            dataframe["domain"]
            == domain
        ].copy()

        if domain_data.empty:
            raise ValueError(
                f"no results found for domain "
                f"{domain}"
            )

        degrees = sorted(
            domain_data[
                "degree"
            ].unique()
        )

        for basis in BASIS_ORDER:
            basis_data = domain_data[
                domain_data["basis"]
                == basis
            ].sort_values(
                "degree"
            )

            if basis_data.empty:
                continue

            x = basis_data[
                "degree"
            ].to_numpy(
                dtype=float
            )

            raw_y = basis_data[
                "condition_number"
            ].to_numpy(
                dtype=float
            )

            y, infinite_mask = (
                display_condition_values(
                    raw_y,
                    ceiling,
                )
            )

            axis.plot(
                x,
                y,
                marker=MARKERS[basis],
                linestyle=LINESTYLES[basis],
                linewidth=1.8,
                markersize=5.5,
                label=BASIS_LABELS[basis],
            )

            if np.any(
                infinite_mask
            ):
                axis.scatter(
                    x[
                        infinite_mask
                    ],
                    y[
                        infinite_mask
                    ],
                    marker="x",
                    s=70,
                    linewidths=2.0,
                    zorder=5,
                )

        axis.set_yscale(
            "log"
        )

        axis.set_xlabel(
            "Polynomial degree"
        )

        axis.set_title(
            DOMAIN_LABELS.get(
                domain,
                domain,
            )
        )

        axis.set_xticks(
            degrees
        )

        axis.grid(
            True,
            which="both",
            axis="y",
            alpha=0.25,
        )

        axis.grid(
            True,
            which="major",
            axis="x",
            alpha=0.15,
        )

        axis.set_ylim(
            bottom=0.8,
            top=ceiling * 2.0,
        )

    axes[0].set_ylabel(
        r"Condition number $\kappa_2(G_C)$"
    )

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(
            0.5,
            -0.03,
        ),
        frameon=False,
    )

    figure.suptitle(
        (
            "Conditioning of the "
            "constraint-induced integration matrix "
            f"$G_C$ in dimension {dimension}"
        ),
        y=1.02,
    )

    # Explain x-markers without overloading the legend.
    figure.text(
        0.5,
        -0.105,
        (
            r"$\times$ indicates a numerically singular "
            r"$G_C$ for which $\kappa_2(G_C)=\infty$; "
            "the marker is shown at a finite plotting ceiling."
        ),
        ha="center",
        va="top",
        fontsize=9,
    )

    return figure


# =============================================================================
# Optional rank-fraction figure
# =============================================================================

def plot_rank_fraction_figure(
    dataframe: pd.DataFrame,
    *,
    dimension: int,
    domains: Sequence[str],
) -> plt.Figure:
    """
    Optional companion figure showing numerical-rank retention.
    """
    figure, axes = plt.subplots(
        1,
        len(domains),
        figsize=(
            11.0,
            4.2,
        ),
        sharey=True,
        constrained_layout=True,
    )

    if len(domains) == 1:
        axes = np.asarray(
            [axes]
        )

    for axis, domain in zip(
        axes,
        domains,
    ):
        domain_data = dataframe[
            dataframe["domain"]
            == domain
        ].copy()

        degrees = sorted(
            domain_data[
                "degree"
            ].unique()
        )

        for basis in BASIS_ORDER:
            basis_data = domain_data[
                domain_data["basis"]
                == basis
            ].sort_values(
                "degree"
            )

            if basis_data.empty:
                continue

            axis.plot(
                basis_data[
                    "degree"
                ].to_numpy(
                    dtype=float
                ),
                basis_data[
                    "numerical_rank_fraction"
                ].to_numpy(
                    dtype=float
                ),
                marker=MARKERS[basis],
                linestyle=LINESTYLES[basis],
                linewidth=1.8,
                markersize=5.5,
                label=BASIS_LABELS[basis],
            )

        axis.set_xlabel(
            "Polynomial degree"
        )

        axis.set_title(
            DOMAIN_LABELS.get(
                domain,
                domain,
            )
        )

        axis.set_xticks(
            degrees
        )

        axis.set_ylim(
            0.0,
            1.05,
        )

        axis.grid(
            True,
            alpha=0.25,
        )

    axes[0].set_ylabel(
        "Retained numerical-rank fraction"
    )

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(
            0.5,
            -0.02,
        ),
        frameon=False,
    )

    figure.suptitle(
        (
            "Numerical-rank retention of "
            f"$G_C$ in dimension {dimension}"
        ),
        y=1.02,
    )

    return figure


# =============================================================================
# Console summary
# =============================================================================

def print_summary(
    dataframe: pd.DataFrame,
    *,
    dimension: int,
    domains: Sequence[str],
) -> None:
    print(
        "Constraint-induced Gram conditioning summary"
    )
    print(
        "============================================"
    )
    print(
        f"dimension: {dimension}"
    )
    print()

    for domain in domains:
        print(
            f"{domain}"
        )
        print(
            "-" * len(domain)
        )

        domain_data = dataframe[
            dataframe["domain"]
            == domain
        ]

        for degree in sorted(
            domain_data[
                "degree"
            ].unique()
        ):
            degree_data = domain_data[
                domain_data["degree"]
                == degree
            ]

            print(
                f"degree={degree}"
            )

            for basis in BASIS_ORDER:
                row = degree_data[
                    degree_data["basis"]
                    == basis
                ]

                if row.empty:
                    continue

                condition = float(
                    row.iloc[0][
                        "condition_number"
                    ]
                )

                rank_fraction = float(
                    row.iloc[0][
                        "numerical_rank_fraction"
                    ]
                )

                if math.isinf(
                    condition
                ):
                    condition_text = "inf"
                else:
                    condition_text = (
                        f"{condition:.6e}"
                    )

                print(
                    f"  "
                    f"{basis:9s} "
                    f"kappa={condition_text:>13s} "
                    f"rank_frac={rank_fraction:.3f}"
                )

        print()


# =============================================================================
# Output
# =============================================================================

def save_figure(
    figure: plt.Figure,
    *,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for file_format in formats:
        output_path = (
            output_dir
            / f"{stem}.{file_format}"
        )

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Wrote {output_path}"
        )


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the condition number of the "
            "constraint-induced Gram matrix G_C."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Input CSV written by "
            "run_constraint_gram_benchmark.py"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--dimension",
        type=int,
        default=DEFAULT_DIMENSION,
    )

    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(
            DEFAULT_DOMAINS
        ),
        choices=DEFAULT_DOMAINS,
    )

    parser.add_argument(
        "--format",
        nargs="+",
        dest="formats",
        default=[
            "pdf",
            "png",
            "svg",
        ],
        choices=SUPPORTED_FORMATS,
    )

    parser.add_argument(
        "--plot-rank",
        action="store_true",
        help=(
            "Also generate the optional "
            "rank-fraction companion figure."
        ),
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.dimension < 1:
        raise ValueError(
            "--dimension must be positive"
        )

    dataframe = load_results(
        args.input
    )

    filtered = filter_results(
        dataframe,
        dimension=args.dimension,
        domains=args.domains,
    )

    print_summary(
        filtered,
        dimension=args.dimension,
        domains=args.domains,
    )

    condition_figure = (
        plot_condition_number_figure(
            filtered,
            dimension=args.dimension,
            domains=args.domains,
        )
    )

    save_figure(
        condition_figure,
        output_dir=args.output_dir,
        stem=DEFAULT_OUTPUT_STEM,
        formats=args.formats,
    )

    plt.close(
        condition_figure
    )

    if args.plot_rank:
        rank_figure = (
            plot_rank_fraction_figure(
                filtered,
                dimension=args.dimension,
                domains=args.domains,
            )
        )

        save_figure(
            rank_figure,
            output_dir=args.output_dir,
            stem=(
                "constraint_gram_rank_fraction_vs_degree"
            ),
            formats=args.formats,
        )

        plt.close(
            rank_figure
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )