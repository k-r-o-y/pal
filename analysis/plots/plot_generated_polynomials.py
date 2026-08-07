#!/usr/bin/env python3
"""Visualise representative q(x)^2 + eta polynomial densities.

Run from the repository root:

    python -m analysis.plots.plot_generated_polynomials \
        --degrees 1 3 5 \
        --trials 0 1 2 \
        --format pdf png svg

Generated files:
    generated_polynomial_1d_examples.*
    generated_polynomial_2d_heatmaps.*
    generated_polynomial_2d_surfaces.*
    generated_polynomial_basis_equivalence.*
    generated_polynomial_constrained_region.*

This file is self-contained and does not require changes to the benchmark runner.
To reproduce an exact benchmark density, replace derive_seed() with the exact seed
formula used by run_polytope_benchmark.py if that runner differs from this script.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import chebyshev, legendre, polynomial

DEFAULT_OUTPUT_DIRECTORY = Path("figures")
DEFAULT_FORMATS = ("pdf", "png", "svg")
BASIS_ORDER = ("monomial", "legendre", "chebyshev")
BASIS_DISPLAY = {
    "monomial": "Monomial",
    "legendre": "Legendre",
    "chebyshev": "Chebyshev",
}
MARKERS = {"monomial": "o", "legendre": "s", "chebyshev": "^"}


@dataclass(frozen=True)
class Density:
    dimension: int
    source_degree: int
    density_degree: int
    trial: int
    seed: int
    offset: float
    coefficient_scale: float
    source_indices: tuple[tuple[int, ...], ...]
    source_coefficients: np.ndarray
    density_indices: tuple[tuple[int, ...], ...]
    density_coefficients: np.ndarray


@dataclass(frozen=True)
class PlotConfig:
    output_directory: Path
    formats: tuple[str, ...]
    dpi: int
    figure_width: float
    figure_height: float
    grid_size_1d: int
    grid_size_2d: int
    obstacle_half_width: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise representative generated polynomial densities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--degrees", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--trials", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--base-seed", type=int, default=20260804)
    parser.add_argument("--density-offset", type=float, default=0.1)
    parser.add_argument("--coefficient-scale", type=float, default=0.35)
    parser.add_argument("--obstacle-half-width", type=float, default=0.15)
    parser.add_argument("--grid-size-1d", type=int, default=800)
    parser.add_argument("--grid-size-2d", type=int, default=220)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--format",
        dest="formats",
        nargs="+",
        choices=DEFAULT_FORMATS,
        default=list(DEFAULT_FORMATS),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figure-width", type=float, default=8.3)
    parser.add_argument("--figure-height", type=float, default=5.4)
    parser.add_argument("--equivalence-degree", type=int, default=None)
    parser.add_argument("--equivalence-trial", type=int, default=None)
    args = parser.parse_args(argv)

    if any(value < 0 for value in args.degrees):
        parser.error("All degrees must be non-negative.")
    if any(value < 0 for value in args.trials):
        parser.error("All trials must be non-negative.")
    if args.density_offset <= 0:
        parser.error("--density-offset must be positive.")
    if args.coefficient_scale < 0:
        parser.error("--coefficient-scale must be non-negative.")
    if not 0 < args.obstacle_half_width < 0.5:
        parser.error("--obstacle-half-width must lie in (0, 0.5).")
    if args.grid_size_1d < 10 or args.grid_size_2d < 20:
        parser.error("Grid sizes are too small.")
    if args.dpi <= 0:
        parser.error("--dpi must be positive.")
    if args.figure_width <= 0 or args.figure_height <= 0:
        parser.error("Figure dimensions must be positive.")

    args.degrees = sorted(dict.fromkeys(args.degrees))
    args.trials = list(dict.fromkeys(args.trials))
    args.formats = tuple(dict.fromkeys(args.formats))
    args.equivalence_degree = (
        max(args.degrees) if args.equivalence_degree is None else args.equivalence_degree
    )
    args.equivalence_trial = (
        args.trials[0] if args.equivalence_trial is None else args.equivalence_trial
    )
    if args.equivalence_degree not in args.degrees:
        parser.error("--equivalence-degree must also appear in --degrees.")
    if args.equivalence_trial not in args.trials:
        parser.error("--equivalence-trial must also appear in --trials.")
    return args


def total_degree_indices(dimension: int, degree: int) -> tuple[tuple[int, ...], ...]:
    indices = [
        item
        for item in itertools.product(range(degree + 1), repeat=dimension)
        if sum(item) <= degree
    ]
    indices.sort(key=lambda item: (sum(item), item))
    return tuple(indices)


def derive_seed(base_seed: int, dimension: int, degree: int, trial: int) -> int:
    """Deterministic instance seed; replace with the runner's formula if needed."""
    return (
        int(base_seed)
        + 1_000_003 * int(dimension)
        + 10_007 * int(degree)
        + 101 * int(trial)
    ) % (2**32 - 5)


def square_polynomial(
    indices: Sequence[tuple[int, ...]], coefficients: np.ndarray
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    terms: dict[tuple[int, ...], float] = {}
    dimension = len(indices[0])
    for left_index, left_value in zip(indices, coefficients):
        for right_index, right_value in zip(indices, coefficients):
            exponent = tuple(
                left_index[axis] + right_index[axis] for axis in range(dimension)
            )
            terms[exponent] = terms.get(exponent, 0.0) + float(left_value * right_value)
    ordered = tuple(sorted(terms, key=lambda item: (sum(item), item)))
    values = np.asarray([terms[item] for item in ordered], dtype=np.float64)
    return ordered, values


def generate_density(
    *,
    dimension: int,
    source_degree: int,
    trial: int,
    base_seed: int,
    offset: float,
    coefficient_scale: float,
) -> Density:
    source_indices = total_degree_indices(dimension, source_degree)
    seed = derive_seed(base_seed, dimension, source_degree, trial)
    rng = np.random.default_rng(seed)
    source_coefficients = rng.normal(
        0.0, coefficient_scale, size=len(source_indices)
    ).astype(np.float64)

    squared_indices, squared_coefficients = square_polynomial(
        source_indices, source_coefficients
    )
    terms = {
        index: float(value)
        for index, value in zip(squared_indices, squared_coefficients)
    }
    zero = (0,) * dimension
    terms[zero] = terms.get(zero, 0.0) + offset
    density_indices = tuple(sorted(terms, key=lambda item: (sum(item), item)))
    density_coefficients = np.asarray(
        [terms[index] for index in density_indices], dtype=np.float64
    )

    return Density(
        dimension=dimension,
        source_degree=source_degree,
        density_degree=2 * source_degree,
        trial=trial,
        seed=seed,
        offset=offset,
        coefficient_scale=coefficient_scale,
        source_indices=source_indices,
        source_coefficients=source_coefficients,
        density_indices=density_indices,
        density_coefficients=density_coefficients,
    )


def evaluate_sparse(
    points: np.ndarray,
    indices: Sequence[tuple[int, ...]],
    coefficients: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    result = np.zeros(points.shape[0], dtype=np.float64)
    for exponent, coefficient in zip(indices, coefficients):
        term = np.ones(points.shape[0], dtype=np.float64)
        for axis, power in enumerate(exponent):
            if power:
                term *= points[:, axis] ** power
        result += coefficient * term
    return result


def dense_1d_coefficients(density: Density) -> np.ndarray:
    coefficients = np.zeros(density.density_degree + 1, dtype=np.float64)
    for exponent, value in zip(density.density_indices, density.density_coefficients):
        coefficients[exponent[0]] = value
    return coefficients


def convert_1d_basis(monomial_coefficients: np.ndarray, basis: str) -> np.ndarray:
    """Represent p(x), x in [0,1], in t=2x-1 coordinates for orthogonal bases."""
    physical = polynomial.Polynomial(monomial_coefficients)
    canonical = physical(polynomial.Polynomial([0.5, 0.5]))
    power_coefficients = np.asarray(canonical.coef, dtype=np.float64)
    if basis == "monomial":
        return monomial_coefficients.copy()
    if basis == "legendre":
        return legendre.poly2leg(power_coefficients)
    if basis == "chebyshev":
        return chebyshev.poly2cheb(power_coefficients)
    raise ValueError(f"Unsupported basis: {basis}")


def evaluate_1d_basis(x: np.ndarray, coefficients: np.ndarray, basis: str) -> np.ndarray:
    if basis == "monomial":
        return polynomial.polyval(x, coefficients)
    canonical_x = 2.0 * x - 1.0
    if basis == "legendre":
        return legendre.legval(canonical_x, coefficients)
    if basis == "chebyshev":
        return chebyshev.chebval(canonical_x, coefficients)
    raise ValueError(f"Unsupported basis: {basis}")


def save_figure(figure: plt.Figure, config: PlotConfig, stem: str) -> list[Path]:
    config.output_directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for extension in config.formats:
        path = config.output_directory / f"{stem}.{extension}"
        kwargs: dict[str, object] = {"bbox_inches": "tight", "pad_inches": 0.05}
        if extension == "png":
            kwargs["dpi"] = config.dpi
        figure.savefig(path, **kwargs)
        written.append(path)
        print(f"Wrote {path}")
    plt.close(figure)
    return written


def add_footer(figure: plt.Figure, text: str) -> None:
    figure.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=8)


def plot_1d_examples(densities: Sequence[Density], config: PlotConfig) -> list[Path]:
    x = np.linspace(0.0, 1.0, config.grid_size_1d)
    figure, axis = plt.subplots(figsize=(config.figure_width, config.figure_height))
    for density in densities:
        values = evaluate_sparse(
            x[:, None], density.density_indices, density.density_coefficients
        )
        axis.plot(
            x,
            values,
            linewidth=1.7,
            label=f"source degree {density.source_degree}, trial {density.trial}",
        )
    axis.set_title("Representative generated one-dimensional densities")
    axis.set_xlabel("x")
    axis.set_ylabel(r"$p(x)=q(x)^2+\eta$")
    axis.grid(True, alpha=0.25)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(3, len(densities)),
        frameon=False,
    )
    add_footer(figure, "Every plotted density is strictly positive because eta > 0.")
    figure.subplots_adjust(bottom=0.28)
    return save_figure(figure, config, "generated_polynomial_1d_examples")


def make_2d_grid(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.linspace(0.0, 1.0, size)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates, indexing="xy")
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    return grid_x, grid_y, points


def plot_2d_heatmaps(densities: Sequence[Density], config: PlotConfig) -> list[Path]:
    count = len(densities)
    columns = min(3, count)
    rows = int(math.ceil(count / columns))
    grid_x, _, points = make_2d_grid(config.grid_size_2d)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(config.figure_width, max(config.figure_height, 3.7 * rows)),
        squeeze=False,
    )
    for axis, density in zip(axes.ravel(), densities):
        values = evaluate_sparse(
            points, density.density_indices, density.density_coefficients
        ).reshape(grid_x.shape)
        image = axis.imshow(
            values,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            aspect="equal",
        )
        axis.set_title(f"source degree {density.source_degree}, trial {density.trial}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes.ravel()[count:]:
        axis.set_visible(False)
    figure.suptitle("Representative generated two-dimensional polynomial densities")
    add_footer(figure, "Heatmap intensity shows p(x,y)=q(x,y)^2+eta on [0,1]^2.")
    figure.subplots_adjust(bottom=0.06, top=0.92)
    return save_figure(figure, config, "generated_polynomial_2d_heatmaps")


def plot_2d_surfaces(densities: Sequence[Density], config: PlotConfig) -> list[Path]:
    count = len(densities)
    columns = min(3, count)
    rows = int(math.ceil(count / columns))
    size = min(config.grid_size_2d, 120)
    grid_x, grid_y, points = make_2d_grid(size)
    figure = plt.figure(
        figsize=(config.figure_width, max(config.figure_height, 3.8 * rows))
    )
    for index, density in enumerate(densities, start=1):
        axis = figure.add_subplot(rows, columns, index, projection="3d")
        values = evaluate_sparse(
            points, density.density_indices, density.density_coefficients
        ).reshape(grid_x.shape)
        axis.plot_surface(grid_x, grid_y, values, linewidth=0, antialiased=True)
        axis.set_title(f"source degree {density.source_degree}, trial {density.trial}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("p(x,y)")
    figure.suptitle("Surface views of generated polynomial densities")
    add_footer(figure, "These surfaces show the same densities as the heatmap figure.")
    figure.subplots_adjust(bottom=0.06, top=0.92)
    return save_figure(figure, config, "generated_polynomial_2d_surfaces")


def plot_basis_equivalence(density: Density, config: PlotConfig) -> list[Path]:
    x = np.linspace(0.0, 1.0, config.grid_size_1d)
    monomial_coefficients = dense_1d_coefficients(density)
    figure, axis = plt.subplots(figsize=(config.figure_width, config.figure_height))
    reference: np.ndarray | None = None
    maximum_difference = 0.0
    for basis in BASIS_ORDER:
        coefficients = convert_1d_basis(monomial_coefficients, basis)
        values = evaluate_1d_basis(x, coefficients, basis)
        if reference is None:
            reference = values
        else:
            maximum_difference = max(
                maximum_difference, float(np.max(np.abs(values - reference)))
            )
        axis.plot(
            x,
            values,
            marker=MARKERS[basis],
            markevery=max(1, config.grid_size_1d // 18),
            markersize=4,
            linewidth=1.5,
            label=BASIS_DISPLAY[basis],
        )
    axis.set_title("Equivalent monomial, Legendre, and Chebyshev representations")
    axis.set_xlabel("x")
    axis.set_ylabel(r"$p(x)$")
    axis.grid(True, alpha=0.25)
    axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False
    )
    add_footer(
        figure,
        f"source degree {density.source_degree}, trial {density.trial}, "
        f"maximum sampled discrepancy={maximum_difference:.3e}",
    )
    figure.subplots_adjust(bottom=0.25)
    return save_figure(figure, config, "generated_polynomial_basis_equivalence")


def plot_constrained_region(density: Density, config: PlotConfig) -> list[Path]:
    grid_x, grid_y, points = make_2d_grid(config.grid_size_2d)
    values = evaluate_sparse(
        points, density.density_indices, density.density_coefficients
    ).reshape(grid_x.shape)
    lower = 0.5 - config.obstacle_half_width
    upper = 0.5 + config.obstacle_half_width
    mask = (
        (grid_x >= lower)
        & (grid_x <= upper)
        & (grid_y >= lower)
        & (grid_y <= upper)
    )
    masked = np.ma.array(values, mask=mask)
    figure, axis = plt.subplots(figsize=(config.figure_width, config.figure_height))
    image = axis.imshow(
        masked,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        aspect="equal",
    )
    axis.add_patch(
        plt.Rectangle(
            (lower, lower),
            2.0 * config.obstacle_half_width,
            2.0 * config.obstacle_half_width,
            fill=False,
            linewidth=2.0,
            linestyle="--",
        )
    )
    axis.set_title("Generated density over the square-with-obstacle feasible region")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    figure.colorbar(image, ax=axis, label=r"$p(x,y)$")
    add_footer(
        figure,
        f"source degree {density.source_degree}, trial {density.trial}, "
        f"obstacle half-width={config.obstacle_half_width:.2f}",
    )
    figure.subplots_adjust(bottom=0.10)
    return save_figure(figure, config, "generated_polynomial_constrained_region")


def print_summary(densities: Sequence[Density], grid_size: int) -> None:
    print("\nRepresentative density summaries")
    print("================================")
    for density in densities:
        if density.dimension == 1:
            points = np.linspace(0.0, 1.0, grid_size)[:, None]
        else:
            _, _, points = make_2d_grid(min(grid_size, 100))
        values = evaluate_sparse(
            points, density.density_indices, density.density_coefficients
        )
        print(
            f"dimension={density.dimension}, source_degree={density.source_degree}, "
            f"density_degree={density.density_degree}, trial={density.trial}, "
            f"seed={density.seed}, min={values.min():.6g}, max={values.max():.6g}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = PlotConfig(
        output_directory=args.output_directory,
        formats=args.formats,
        dpi=args.dpi,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        grid_size_1d=args.grid_size_1d,
        grid_size_2d=args.grid_size_2d,
        obstacle_half_width=args.obstacle_half_width,
    )
    try:
        pairs = list(zip(args.degrees, itertools.cycle(args.trials)))
        densities_1d = [
            generate_density(
                dimension=1,
                source_degree=degree,
                trial=trial,
                base_seed=args.base_seed,
                offset=args.density_offset,
                coefficient_scale=args.coefficient_scale,
            )
            for degree, trial in pairs
        ]
        densities_2d = [
            generate_density(
                dimension=2,
                source_degree=degree,
                trial=trial,
                base_seed=args.base_seed,
                offset=args.density_offset,
                coefficient_scale=args.coefficient_scale,
            )
            for degree, trial in pairs
        ]
        equivalence_density = generate_density(
            dimension=1,
            source_degree=args.equivalence_degree,
            trial=args.equivalence_trial,
            base_seed=args.base_seed,
            offset=args.density_offset,
            coefficient_scale=args.coefficient_scale,
        )
        constrained_density = generate_density(
            dimension=2,
            source_degree=args.equivalence_degree,
            trial=args.equivalence_trial,
            base_seed=args.base_seed,
            offset=args.density_offset,
            coefficient_scale=args.coefficient_scale,
        )

        print("\nGenerated-polynomial plotting configuration")
        print("==========================================")
        print(f"degrees: {args.degrees}")
        print(f"trials: {args.trials}")
        print(f"base seed: {args.base_seed}")
        print(f"density offset: {args.density_offset}")
        print(f"coefficient scale: {args.coefficient_scale}")
        print(f"output directory: {config.output_directory}")
        print(f"formats: {list(config.formats)}")

        written: list[Path] = []
        written.extend(plot_1d_examples(densities_1d, config))
        written.extend(plot_2d_heatmaps(densities_2d, config))
        written.extend(plot_2d_surfaces(densities_2d, config))
        written.extend(plot_basis_equivalence(equivalence_density, config))
        written.extend(plot_constrained_region(constrained_density, config))
        print_summary([*densities_1d, *densities_2d], config.grid_size_1d)
        print(f"\nGenerated {len(written)} figure files.")
        return 0
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
