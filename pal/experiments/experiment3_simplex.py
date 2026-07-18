from __future__ import annotations

import argparse
import csv
import itertools
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from numpy.polynomial import Chebyshev, Legendre, Polynomial
from numpy.polynomial.legendre import leggauss

from pal.experiments.common import (
    ensure_directory,
    environment_metadata,
    seed_everything,
    write_json,
)
from pal.experiments.experiment1_polynomial_evaluation import (
    evaluate_chebyshev,
    evaluate_legendre,
    evaluate_monomial,
)

BasisName = Literal["monomial", "legendre", "chebyshev"]
PrecisionName = Literal["float32", "float64"]


@dataclass(frozen=True)
class SimplexConfig:
    dimensions: tuple[int, ...] = (2, 3, 4, 5)
    degrees: tuple[int, ...] = (2, 4, 6, 8)
    scales: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    precisions: tuple[PrecisionName, ...] = ("float32", "float64")
    bases: tuple[BasisName, ...] = ("monomial", "legendre", "chebyshev")
    seeds: tuple[int, ...] = (0, 1, 2)
    quadrature_order: int = 10
    reference_dps: int = 100
    repetitions: int = 3


@dataclass(frozen=True)
class SimplexResult:
    experiment: str
    dimension: int
    degree: int
    scale: float
    basis: str
    precision: str
    seed: int
    quadrature_order: int
    reference_integral: float
    computed_integral: float
    absolute_error: float
    relative_error: float
    median_runtime_seconds: float
    status: str


def generate_coefficients(degree: int, seed: int) -> np.ndarray:
    """
    Generate one-dimensional monomial coefficients.

    The multivariate polynomial is separable:
        f(x_1,...,x_d) = product_j p(x_j).
    This keeps the same underlying function across all basis representations.
    """
    rng = np.random.default_rng(seed)
    coefficients = rng.uniform(-0.75, 0.75, size=degree + 1)

    # Keep a non-negligible constant and leading term.
    if abs(coefficients[0]) < 0.2:
        coefficients[0] = math.copysign(0.2, coefficients[0] or 1.0)
    if abs(coefficients[-1]) < 0.1:
        coefficients[-1] = math.copysign(0.1, coefficients[-1] or 1.0)

    return np.asarray(coefficients, dtype=np.float64)


def scaled_coordinate_polynomial(
    monomial_coefficients: np.ndarray,
    scale: float,
) -> Polynomial:
    """
    Return q(t)=p(scale*(t+1)/2), where t is in [-1,1].

    Legendre and Chebyshev coefficients are computed for q, so their natural
    coordinate remains in [-1,1] while the physical simplex coordinate is x.
    """
    physical = Polynomial(monomial_coefficients)
    affine_map = Polynomial([scale / 2.0, scale / 2.0])
    return physical(affine_map)


def coefficients_for_basis(
    monomial_coefficients: np.ndarray,
    scale: float,
    basis: BasisName,
) -> np.ndarray:
    if basis == "monomial":
        return np.asarray(monomial_coefficients, dtype=np.float64)

    scaled = scaled_coordinate_polynomial(monomial_coefficients, scale)
    if basis == "legendre":
        return np.asarray(scaled.convert(kind=Legendre).coef, dtype=np.float64)
    if basis == "chebyshev":
        return np.asarray(scaled.convert(kind=Chebyshev).coef, dtype=np.float64)

    raise ValueError(f"Unsupported basis: {basis}")


def evaluate_1d_basis(
    coefficients: np.ndarray,
    x_physical: np.ndarray,
    scale: float,
    basis: BasisName,
    dtype: np.dtype,
) -> np.ndarray:
    if basis == "monomial":
        return evaluate_monomial(
            coefficients,
            np.asarray(x_physical, dtype=dtype),
            dtype,
        )

    t = np.asarray((2.0 * x_physical / scale) - 1.0, dtype=dtype)

    if basis == "legendre":
        return evaluate_legendre(coefficients, t, dtype)
    if basis == "chebyshev":
        return evaluate_chebyshev(coefficients, t, dtype)

    raise ValueError(f"Unsupported basis: {basis}")


def exact_separable_simplex_integral(
    coefficients: np.ndarray,
    dimension: int,
    scale: float,
    decimal_places: int,
) -> float:
    """
    Compute the exact integral of product_j p(x_j) over

        {x_j >= 0, sum_j x_j <= scale}

    using the monomial simplex formula:
        integral x^alpha dx =
        scale^(|alpha|+d) * product(alpha_j!) / (|alpha|+d)!.
    """
    mp.mp.dps = decimal_places
    coeff_mp = [mp.mpf(str(value)) for value in coefficients]
    scale_mp = mp.mpf(str(scale))
    total = mp.mpf("0")

    exponent_range = range(len(coefficients))
    for alpha in itertools.product(exponent_range, repeat=dimension):
        coefficient_product = mp.mpf("1")
        factorial_product = mp.mpf("1")
        degree_sum = 0

        for exponent in alpha:
            coefficient_product *= coeff_mp[exponent]
            factorial_product *= mp.factorial(exponent)
            degree_sum += exponent

        term = (
            coefficient_product
            * scale_mp ** (degree_sum + dimension)
            * factorial_product
            / mp.factorial(degree_sum + dimension)
        )
        total += term

    return float(total)


def duffy_nodes_and_weights(
    dimension: int,
    scale: float,
    order: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map tensor Gauss-Legendre nodes on [0,1]^d to a scaled simplex.

    Stick-breaking/Duffy map:
        x_1 = s u_1
        x_2 = s (1-u_1) u_2
        ...
        x_d = s product_{k<d}(1-u_k) u_d

    Jacobian:
        s^d product_{k=1}^{d-1}(1-u_k)^(d-k)
    """
    nodes, weights = leggauss(order)
    unit_nodes = ((nodes + 1.0) / 2.0).astype(dtype)
    unit_weights = (weights / 2.0).astype(dtype)

    point_count = order ** dimension
    points = np.empty((point_count, dimension), dtype=dtype)
    all_weights = np.empty(point_count, dtype=dtype)

    index = 0
    for multi_index in itertools.product(range(order), repeat=dimension):
        remaining = dtype.type(1.0)
        jacobian = dtype.type(scale ** dimension)
        weight = dtype.type(1.0)

        for axis, node_index in enumerate(multi_index):
            u = unit_nodes[node_index]
            points[index, axis] = dtype.type(scale) * remaining * u
            weight *= unit_weights[node_index]

            if axis < dimension - 1:
                jacobian *= (dtype.type(1.0) - u) ** (dimension - axis - 1)

            remaining *= dtype.type(1.0) - u

        all_weights[index] = weight * jacobian
        index += 1

    return points, all_weights


def integrate_basis_over_simplex(
    coefficients: np.ndarray,
    dimension: int,
    scale: float,
    basis: BasisName,
    precision: PrecisionName,
    quadrature_order: int,
) -> float:
    dtype = np.dtype(np.float32 if precision == "float32" else np.float64)
    points, weights = duffy_nodes_and_weights(
        dimension=dimension,
        scale=scale,
        order=quadrature_order,
        dtype=dtype,
    )

    values = np.ones(points.shape[0], dtype=dtype)
    for axis in range(dimension):
        values *= evaluate_1d_basis(
            coefficients=coefficients,
            x_physical=points[:, axis],
            scale=scale,
            basis=basis,
            dtype=dtype,
        )

    # Sum in the requested precision to expose accumulation behaviour.
    return float(np.sum(values * weights, dtype=dtype))


def median_runtime(function: Callable[[], float], repetitions: int) -> float:
    timings: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def run_simplex_experiment(
    config: SimplexConfig,
    results_directory: Path,
    figures_directory: Path,
) -> list[SimplexResult]:
    ensure_directory(results_directory)
    ensure_directory(figures_directory)

    rows: list[SimplexResult] = []

    for seed in config.seeds:
        seed_everything(seed)

        for dimension in config.dimensions:
            for degree in config.degrees:
                source_coefficients = generate_coefficients(degree, seed)

                for scale in config.scales:
                    reference = exact_separable_simplex_integral(
                        source_coefficients,
                        dimension,
                        scale,
                        config.reference_dps,
                    )

                    for basis in config.bases:
                        basis_coefficients = coefficients_for_basis(
                            source_coefficients,
                            scale,
                            basis,
                        )

                        for precision in config.precisions:
                            calculation = lambda: integrate_basis_over_simplex(
                                coefficients=basis_coefficients,
                                dimension=dimension,
                                scale=scale,
                                basis=basis,
                                precision=precision,
                                quadrature_order=config.quadrature_order,
                            )

                            try:
                                computed = calculation()
                                absolute_error = abs(computed - reference)
                                relative_error = absolute_error / max(
                                    abs(reference),
                                    np.finfo(np.float64).eps,
                                )
                                runtime = median_runtime(
                                    calculation,
                                    config.repetitions,
                                )
                                status = "ok"
                            except (
                                FloatingPointError,
                                OverflowError,
                                ValueError,
                                MemoryError,
                            ) as exc:
                                computed = float("nan")
                                absolute_error = float("nan")
                                relative_error = float("nan")
                                runtime = float("nan")
                                status = (
                                    f"failed: {type(exc).__name__}: {exc}"
                                )

                            rows.append(
                                SimplexResult(
                                    experiment="experiment3_scaled_simplex",
                                    dimension=dimension,
                                    degree=degree,
                                    scale=scale,
                                    basis=basis,
                                    precision=precision,
                                    seed=seed,
                                    quadrature_order=config.quadrature_order,
                                    reference_integral=reference,
                                    computed_integral=computed,
                                    absolute_error=absolute_error,
                                    relative_error=relative_error,
                                    median_runtime_seconds=runtime,
                                    status=status,
                                )
                            )

    csv_path = results_directory / "simplex_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(rows[0]).keys()),
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    write_json(
        results_directory / "config.json",
        {
            "config": asdict(config),
            "environment": environment_metadata(),
        },
    )
    create_summary(rows, results_directory)
    create_plots(rows, figures_directory)
    return rows


def valid_rows(rows: list[SimplexResult]) -> list[SimplexResult]:
    return [
        row
        for row in rows
        if row.status == "ok"
        and np.isfinite(row.absolute_error)
        and np.isfinite(row.relative_error)
    ]


def create_summary(
    rows: list[SimplexResult],
    results_directory: Path,
) -> None:
    grouped: dict[
        tuple[int, int, float, str, str],
        list[SimplexResult],
    ] = {}

    for row in valid_rows(rows):
        key = (
            row.dimension,
            row.degree,
            row.scale,
            row.basis,
            row.precision,
        )
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []

    for key, group in sorted(grouped.items()):
        dimension, degree, scale, basis, precision = key
        summary.append(
            {
                "dimension": dimension,
                "degree": degree,
                "scale": scale,
                "basis": basis,
                "precision": precision,
                "runs": len(group),
                "median_absolute_error": float(
                    np.median([row.absolute_error for row in group])
                ),
                "median_relative_error": float(
                    np.median([row.relative_error for row in group])
                ),
                "median_runtime_seconds": float(
                    np.median(
                        [row.median_runtime_seconds for row in group]
                    )
                ),
            }
        )

    write_json(
        results_directory / "simplex_summary.json",
        summary,
    )


def median_curve(
    rows: list[SimplexResult],
    *,
    basis: str,
    precision: str,
    dimension: int,
    scale: float,
    metric: str,
) -> tuple[list[int], list[float]]:
    subset = [
        row
        for row in valid_rows(rows)
        if row.basis == basis
        and row.precision == precision
        and row.dimension == dimension
        and row.scale == scale
    ]
    degrees = sorted({row.degree for row in subset})
    values = [
        float(
            np.median(
                [
                    getattr(row, metric)
                    for row in subset
                    if row.degree == degree
                ]
            )
        )
        for degree in degrees
    ]
    return degrees, values


def create_plots(
    rows: list[SimplexResult],
    figures_directory: Path,
) -> None:
    valid = valid_rows(rows)
    dimensions = sorted({row.dimension for row in valid})
    scales = sorted({row.scale for row in valid})

    # Error versus degree, one plot per dimension/scale/precision.
    for dimension in dimensions:
        for scale in scales:
            for precision in ("float32", "float64"):
                plt.figure(figsize=(7.2, 4.8))
                for basis in ("monomial", "legendre", "chebyshev"):
                    degrees, values = median_curve(
                        valid,
                        basis=basis,
                        precision=precision,
                        dimension=dimension,
                        scale=scale,
                        metric="relative_error",
                    )
                    plt.plot(degrees, values, marker="o", label=basis)

                plt.yscale("log")
                plt.xlabel("Polynomial degree per coordinate")
                plt.ylabel("Median relative integral error")
                plt.title(
                    f"Scaled simplex: d={dimension}, scale={scale:g}, "
                    f"{precision}"
                )
                plt.grid(True, which="both", alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    figures_directory
                    / (
                        f"relative_error_vs_degree_dim{dimension}_"
                        f"scale{scale:g}_{precision}.png"
                    ),
                    dpi=220,
                )
                plt.close()

    # Error versus scale at the largest degree.
    max_degree = max(row.degree for row in valid)
    for dimension in dimensions:
        for precision in ("float32", "float64"):
            plt.figure(figsize=(7.2, 4.8))
            for basis in ("monomial", "legendre", "chebyshev"):
                subset = [
                    row
                    for row in valid
                    if row.dimension == dimension
                    and row.degree == max_degree
                    and row.precision == precision
                    and row.basis == basis
                ]
                x_scales = sorted({row.scale for row in subset})
                values = [
                    float(
                        np.median(
                            [
                                row.relative_error
                                for row in subset
                                if row.scale == scale
                            ]
                        )
                    )
                    for scale in x_scales
                ]
                plt.plot(x_scales, values, marker="o", label=basis)

            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Simplex scale")
            plt.ylabel("Median relative integral error")
            plt.title(
                f"Scale sensitivity: d={dimension}, degree={max_degree}, "
                f"{precision}"
            )
            plt.grid(True, which="both", alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                figures_directory
                / (
                    f"relative_error_vs_scale_dim{dimension}_"
                    f"degree{max_degree}_{precision}.png"
                ),
                dpi=220,
            )
            plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scaled unit-simplex experiment across dimensions, scales, "
            "bases and floating-point precisions."
        )
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced smoke-test configuration.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/experiment3"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures/experiment3"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = (
        SimplexConfig(
            dimensions=(2, 3),
            degrees=(2, 4),
            scales=(0.1, 1.0, 10.0),
            seeds=(0,),
            quadrature_order=8,
            repetitions=1,
        )
        if args.quick
        else SimplexConfig()
    )

    rows = run_simplex_experiment(
        config=config,
        results_directory=args.results_dir,
        figures_directory=args.figures_dir,
    )

    successful = sum(row.status == "ok" for row in rows)
    print(f"Completed {successful}/{len(rows)} runs.")
    print("Results:", args.results_dir / "simplex_results.csv")
    print("Summary:", args.results_dir / "simplex_summary.json")
    print("Figures:", args.figures_dir)


if __name__ == "__main__":
    main()
