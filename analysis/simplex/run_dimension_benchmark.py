#!/usr/bin/env python3
"""
Unified n-dimensional simplex benchmark.

This script is the first remaining experiment file for Chapter 5. It validates the
general simplex implementation in 1D, reproduces the current 2D experiment, and
extends the comparison to higher dimensions.

It compares monomial, Legendre, and Chebyshev product bases over the scaled simplex

    Delta_n(s) = {x in R^n : x_i >= 0 and sum_i x_i <= s}.

For each dimension-degree-trial-basis configuration it records:

- number of basis functions;
- basis-matrix shape and numerical rank;
- largest and smallest singular values;
- 2-norm condition number;
- analytical reference integral;
- computed integral in the tested basis;
- absolute and relative integration error;
- coefficient-space perturbation sensitivity;
- integration-stage runtime.

The same mathematical polynomial, point set, and perturbation direction seed are used
across the three basis representations within each trial.

The script is self-contained apart from NumPy and pandas.

Example
-------
python -m analysis.simplex.run_dimension_benchmark

Custom run
----------
python -m analysis.simplex.run_dimension_benchmark \
    --dimensions 1 2 3 4 5 \
    --trials 5 \
    --dtype float64 \
    --scale 1.0 \
    --output results/simplex/dimension_results.csv
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev, Legendre, Polynomial
from numpy.typing import NDArray


BasisName = str
MultiIndex = tuple[int, ...]
CoefficientMap = dict[MultiIndex, float]

BASES: tuple[BasisName, ...] = ("monomial", "legendre", "chebyshev")

DEFAULT_DIMENSION_DEGREE_GRID: dict[int, tuple[int, ...]] = {
    1: (0, 1, 2, 3, 5, 8, 10),
    2: (0, 1, 2, 3, 5, 8, 10),
    3: (0, 1, 2, 3, 5, 8),
    4: (0, 1, 2, 3, 5),
    5: (0, 1, 2, 3, 5),
}


@dataclass(frozen=True)
class BenchmarkRecord:
    basis: str
    dimension: int
    degree: int
    basis_count: int
    trial: int
    seed: int
    dtype: str
    scale: float
    num_points: int
    matrix_rows: int
    matrix_columns: int
    matrix_rank: int
    sigma_max: float
    sigma_min: float
    condition_number: float
    reference_integral: float
    computed_integral: float
    absolute_error: float
    relative_error: float
    perturbation_magnitude: float
    perturbed_integral: float
    perturbation_sensitivity: float
    runtime_seconds: float


def enumerate_total_degree_indices(
    dimension: int,
    degree: int,
) -> list[MultiIndex]:
    """Return all non-negative multi-indices with total degree <= degree."""
    if dimension < 1:
        raise ValueError("dimension must be at least 1")
    if degree < 0:
        raise ValueError("degree must be non-negative")

    indices = [
        tuple(alpha)
        for alpha in product(range(degree + 1), repeat=dimension)
        if sum(alpha) <= degree
    ]

    # Stable graded lexicographic ordering.
    indices.sort(key=lambda alpha: (sum(alpha), alpha))
    return indices


def expected_basis_count(dimension: int, degree: int) -> int:
    """Return C(n+d, d), the total-degree basis count."""
    return math.comb(dimension + degree, degree)


def sample_scaled_simplex_points(
    *,
    dimension: int,
    count: int,
    scale: float,
    rng: np.random.Generator,
    dtype: np.dtype,
) -> NDArray[np.floating]:
    """
    Draw points uniformly from Delta_n(scale).

    A Dirichlet vector with n+1 components is sampled; the final component is the
    slack variable. The first n components are simplex coordinates.
    """
    if count < 1:
        raise ValueError("count must be positive")
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    barycentric = rng.dirichlet(
        alpha=np.ones(dimension + 1, dtype=np.float64),
        size=count,
    )
    points = scale * barycentric[:, :dimension]
    return points.astype(dtype, copy=False)


def sample_monomial_coefficients(
    *,
    indices: Sequence[MultiIndex],
    rng: np.random.Generator,
    dtype: np.dtype,
) -> CoefficientMap:
    """
    Generate a reproducible, moderately scaled polynomial in the monomial basis.

    Coefficients are damped by total degree to avoid making every high-dimensional
    configuration dominated by the largest-degree terms.
    """
    coefficients: CoefficientMap = {}

    for alpha in indices:
        degree = sum(alpha)
        damping = 1.0 / (1.0 + degree)
        value = float(rng.normal(loc=0.0, scale=damping))
        coefficients[alpha] = float(np.asarray(value, dtype=dtype))

    # Keep the polynomial away from an identically tiny integral.
    zero_index = (0,) * len(indices[0])
    coefficients[zero_index] += 1.0
    return coefficients


def polynomial_power_coefficients_in_orthogonal_basis(
    power: int,
    basis: BasisName,
) -> NDArray[np.float64]:
    """
    Expand x**power in a 1D orthogonal basis defined on t = 2x - 1.

    Since x = (t + 1) / 2, x**power is first represented as a Polynomial in t,
    then converted to Legendre or Chebyshev coefficients.
    """
    if power < 0:
        raise ValueError("power must be non-negative")

    x_as_t = Polynomial([0.5, 0.5])
    polynomial_in_t = x_as_t ** power

    if basis == "legendre":
        return polynomial_in_t.convert(kind=Legendre).coef.astype(np.float64)
    if basis == "chebyshev":
        return polynomial_in_t.convert(kind=Chebyshev).coef.astype(np.float64)
    if basis == "monomial":
        result = np.zeros(power + 1, dtype=np.float64)
        result[power] = 1.0
        return result

    raise ValueError(f"unsupported basis: {basis}")


def convert_monomial_to_basis(
    monomial_coefficients: Mapping[MultiIndex, float],
    *,
    basis: BasisName,
) -> CoefficientMap:
    """
    Convert a multivariate monomial polynomial to a tensor-product basis.

    The basis index set remains total-degree bounded because a degree-k univariate
    monomial only produces orthogonal terms of degree at most k.
    """
    if basis == "monomial":
        return dict(monomial_coefficients)

    output: defaultdict[MultiIndex, float] = defaultdict(float)

    for alpha, coefficient in monomial_coefficients.items():
        per_axis = [
            polynomial_power_coefficients_in_orthogonal_basis(power, basis)
            for power in alpha
        ]

        index_ranges = [range(len(axis_coefficients)) for axis_coefficients in per_axis]

        for beta in product(*index_ranges):
            term_coefficient = float(coefficient)

            for axis, basis_degree in enumerate(beta):
                term_coefficient *= float(per_axis[axis][basis_degree])

            output[tuple(beta)] += term_coefficient

    return {
        index: value
        for index, value in output.items()
        if value != 0.0
    }


def compose_polynomial(
    outer: NDArray[np.float64],
    inner: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return coefficients of outer(inner(x)) in ascending power order."""
    result = np.zeros(1, dtype=np.float64)
    power = np.ones(1, dtype=np.float64)

    for coefficient in outer:
        if len(result) < len(power):
            result = np.pad(result, (0, len(power) - len(result)))
        result[: len(power)] += float(coefficient) * power
        power = np.convolve(power, inner)

    return np.trim_zeros(result, trim="b") if np.any(result) else np.zeros(1)


def orthogonal_basis_function_as_monomial(
    degree: int,
    basis: BasisName,
) -> NDArray[np.float64]:
    """
    Expand phi_degree(2x-1) as a power polynomial in x.
    """
    if basis == "monomial":
        coefficients = np.zeros(degree + 1, dtype=np.float64)
        coefficients[degree] = 1.0
        return coefficients

    if basis == "legendre":
        polynomial_in_t = Legendre.basis(degree).convert(kind=Polynomial).coef
    elif basis == "chebyshev":
        polynomial_in_t = Chebyshev.basis(degree).convert(kind=Polynomial).coef
    else:
        raise ValueError(f"unsupported basis: {basis}")

    # t = 2x - 1
    return compose_polynomial(
        np.asarray(polynomial_in_t, dtype=np.float64),
        np.asarray([-1.0, 2.0], dtype=np.float64),
    )


def convert_basis_to_monomial(
    basis_coefficients: Mapping[MultiIndex, float],
    *,
    basis: BasisName,
) -> CoefficientMap:
    """Convert a tensor-product basis representation back to monomials."""
    if basis == "monomial":
        return dict(basis_coefficients)

    output: defaultdict[MultiIndex, float] = defaultdict(float)

    for beta, coefficient in basis_coefficients.items():
        per_axis = [
            orthogonal_basis_function_as_monomial(degree, basis)
            for degree in beta
        ]
        exponent_ranges = [range(len(axis_coefficients)) for axis_coefficients in per_axis]

        for alpha in product(*exponent_ranges):
            term_coefficient = float(coefficient)

            for axis, exponent in enumerate(alpha):
                term_coefficient *= float(per_axis[axis][exponent])

            output[tuple(alpha)] += term_coefficient

    return {
        index: value
        for index, value in output.items()
        if value != 0.0
    }


def evaluate_univariate_basis(
    values: NDArray[np.floating],
    *,
    degree: int,
    basis: BasisName,
    scale: float,
) -> NDArray[np.floating]:
    """Evaluate one 1D basis function on physical coordinates."""
    if basis == "monomial":
        return values ** degree

    canonical = (2.0 * values / scale) - 1.0

    if basis == "legendre":
        coefficients = np.zeros(degree + 1, dtype=np.float64)
        coefficients[degree] = 1.0
        return np.polynomial.legendre.legval(canonical, coefficients)

    if basis == "chebyshev":
        coefficients = np.zeros(degree + 1, dtype=np.float64)
        coefficients[degree] = 1.0
        return np.polynomial.chebyshev.chebval(canonical, coefficients)

    raise ValueError(f"unsupported basis: {basis}")


def build_basis_matrix(
    points: NDArray[np.floating],
    *,
    indices: Sequence[MultiIndex],
    basis: BasisName,
    scale: float,
    dtype: np.dtype,
) -> NDArray[np.floating]:
    """Build the generalised Vandermonde matrix V_ij = phi_j(x_i)."""
    rows, dimension = points.shape
    matrix = np.ones((rows, len(indices)), dtype=dtype)

    max_degree = max((max(alpha) for alpha in indices), default=0)

    # Cache every 1D basis value used by the tensor-product terms.
    cache: dict[tuple[int, int], NDArray[np.floating]] = {}

    for axis in range(dimension):
        for degree in range(max_degree + 1):
            cache[(axis, degree)] = np.asarray(
                evaluate_univariate_basis(
                    points[:, axis],
                    degree=degree,
                    basis=basis,
                    scale=scale,
                ),
                dtype=dtype,
            )

    for column, alpha in enumerate(indices):
        values = np.ones(rows, dtype=dtype)

        for axis, degree in enumerate(alpha):
            values *= cache[(axis, degree)]

        matrix[:, column] = values

    return matrix


def matrix_diagnostics(
    matrix: NDArray[np.floating],
) -> tuple[int, float, float, float]:
    """Return numerical rank, sigma_max, sigma_min, and 2-norm condition number."""
    matrix64 = np.asarray(matrix, dtype=np.float64)

    singular_values = np.linalg.svd(
        matrix64,
        full_matrices=False,
        compute_uv=False,
    )

    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])

    tolerance = (
        max(matrix64.shape)
        * np.finfo(np.float64).eps
        * sigma_max
    )
    rank = int(np.count_nonzero(singular_values > tolerance))

    condition_number = (
        math.inf
        if sigma_min == 0.0
        else sigma_max / sigma_min
    )

    return rank, sigma_max, sigma_min, float(condition_number)


def scaled_simplex_monomial_integral(
    exponents: Sequence[int],
    *,
    scale: float,
) -> float:
    """
    Integrate prod_i x_i**alpha_i over Delta_n(scale).

    The unit-simplex value is multiplied by scale**(n + sum(alpha)).
    """
    if any(alpha < 0 for alpha in exponents):
        raise ValueError("exponents must be non-negative")
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    dimension = len(exponents)
    total_degree = sum(exponents)

    log_unit_integral = (
        sum(math.lgamma(alpha + 1.0) for alpha in exponents)
        - math.lgamma(dimension + total_degree + 1.0)
    )

    return math.exp(
        log_unit_integral
        + (dimension + total_degree) * math.log(scale)
    )


def integrate_monomial_polynomial(
    coefficients: Mapping[MultiIndex, float],
    *,
    scale: float,
) -> float:
    """Analytically integrate a monomial polynomial over Delta_n(scale)."""
    return math.fsum(
        float(coefficient)
        * scaled_simplex_monomial_integral(alpha, scale=scale)
        for alpha, coefficient in coefficients.items()
    )


def integrate_in_basis(
    coefficients: Mapping[MultiIndex, float],
    *,
    basis: BasisName,
    scale: float,
) -> float:
    """
    Integrate a basis representation by converting it to monomials and applying the
    analytical simplex formula.

    This deliberately retains basis-conversion floating-point effects while using an
    independent exact formula for each monomial term.
    """
    monomial_coefficients = convert_basis_to_monomial(
        coefficients,
        basis=basis,
    )
    return integrate_monomial_polynomial(
        monomial_coefficients,
        scale=scale,
    )


def guarded_relative_error(
    estimate: float,
    reference: float,
) -> float:
    """Compute |estimate-reference| / max(|reference|, machine epsilon)."""
    denominator = max(
        abs(reference),
        np.finfo(np.float64).eps,
    )
    return abs(estimate - reference) / denominator


def perturb_coefficients(
    coefficients: Mapping[MultiIndex, float],
    *,
    relative_magnitude: float,
    rng: np.random.Generator,
    dtype: np.dtype,
) -> CoefficientMap:
    """Apply a random relative 2-norm perturbation to a coefficient map."""
    if relative_magnitude <= 0.0:
        raise ValueError("relative_magnitude must be positive")

    indices = sorted(coefficients)
    vector = np.asarray(
        [coefficients[index] for index in indices],
        dtype=dtype,
    )

    direction = rng.normal(size=vector.shape).astype(dtype, copy=False)
    direction_norm = float(np.linalg.norm(direction.astype(np.float64)))

    if direction_norm == 0.0:
        raise FloatingPointError("sampled a zero perturbation direction")

    direction = direction / direction_norm

    coefficient_norm = float(np.linalg.norm(vector.astype(np.float64)))
    coefficient_scale = max(
        coefficient_norm,
        np.finfo(np.float64).eps,
    )

    perturbed = (
        vector
        + relative_magnitude * coefficient_scale * direction
    )

    return {
        index: float(value)
        for index, value in zip(indices, perturbed, strict=True)
    }


def perturbation_sensitivity(
    *,
    original_integral: float,
    perturbed_integral: float,
    perturbation_magnitude: float,
) -> float:
    """Return relative output change divided by relative input perturbation."""
    denominator = max(
        abs(original_integral),
        np.finfo(np.float64).eps,
    )
    relative_output_change = (
        abs(perturbed_integral - original_integral)
        / denominator
    )
    return relative_output_change / perturbation_magnitude


def parse_dtype(name: str) -> np.dtype:
    """Resolve a supported floating-point dtype."""
    if name == "float32":
        return np.dtype(np.float32)
    if name == "float64":
        return np.dtype(np.float64)
    raise ValueError("dtype must be float32 or float64")


def resolve_grid(
    dimensions: Sequence[int],
    maximum_degree: int | None,
) -> dict[int, tuple[int, ...]]:
    """Resolve the dimension-degree grid requested from the command line."""
    grid: dict[int, tuple[int, ...]] = {}

    for dimension in dimensions:
        if dimension not in DEFAULT_DIMENSION_DEGREE_GRID:
            raise ValueError(
                f"no default degree grid is defined for dimension {dimension}"
            )

        degrees = DEFAULT_DIMENSION_DEGREE_GRID[dimension]

        if maximum_degree is not None:
            degrees = tuple(
                degree
                for degree in degrees
                if degree <= maximum_degree
            )

        if not degrees:
            raise ValueError(
                f"maximum degree removed every configuration for dimension {dimension}"
            )

        grid[dimension] = degrees

    return grid


def run_benchmark(
    *,
    grid: Mapping[int, Sequence[int]],
    trials: int,
    seed: int,
    scale: float,
    dtype: np.dtype,
    oversampling_factor: float,
    perturbation_magnitude: float,
) -> list[BenchmarkRecord]:
    """Execute every requested dimension-degree-trial-basis configuration."""
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if oversampling_factor < 1.0:
        raise ValueError("oversampling_factor must be at least 1")
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    records: list[BenchmarkRecord] = []

    for dimension, degrees in grid.items():
        for degree in degrees:
            indices = enumerate_total_degree_indices(dimension, degree)
            basis_count = len(indices)
            expected_count = expected_basis_count(dimension, degree)

            if basis_count != expected_count:
                raise AssertionError(
                    f"basis count mismatch for n={dimension}, d={degree}: "
                    f"expected {expected_count}, found {basis_count}"
                )

            num_points = max(
                basis_count,
                int(math.ceil(oversampling_factor * basis_count)),
            )

            for trial in range(trials):
                trial_seed = (
                    seed
                    + 1_000_003 * dimension
                    + 10_007 * degree
                    + 101 * trial
                )
                coefficient_rng = np.random.default_rng(trial_seed)
                point_rng = np.random.default_rng(trial_seed + 1)
                perturbation_seed = trial_seed + 2

                monomial_coefficients = sample_monomial_coefficients(
                    indices=indices,
                    rng=coefficient_rng,
                    dtype=dtype,
                )

                reference_integral = integrate_monomial_polynomial(
                    monomial_coefficients,
                    scale=scale,
                )

                points = sample_scaled_simplex_points(
                    dimension=dimension,
                    count=num_points,
                    scale=scale,
                    rng=point_rng,
                    dtype=dtype,
                )

                if not np.isfinite(points).all():
                    raise FloatingPointError(
                        f"non-finite simplex points for n={dimension}, d={degree}"
                    )

                for basis_index, basis in enumerate(BASES):
                    basis_coefficients = convert_monomial_to_basis(
                        monomial_coefficients,
                        basis=basis,
                    )

                    matrix = build_basis_matrix(
                        points,
                        indices=indices,
                        basis=basis,
                        scale=scale,
                        dtype=dtype,
                    )

                    if matrix.shape[1] != expected_count:
                        raise AssertionError(
                            f"matrix column mismatch for {basis}, "
                            f"n={dimension}, d={degree}"
                        )

                    if not np.isfinite(matrix).all():
                        raise FloatingPointError(
                            f"non-finite basis matrix for {basis}, "
                            f"n={dimension}, d={degree}"
                        )

                    rank, sigma_max, sigma_min, condition_number = (
                        matrix_diagnostics(matrix)
                    )

                    start = time.perf_counter()
                    computed_integral = integrate_in_basis(
                        basis_coefficients,
                        basis=basis,
                        scale=scale,
                    )
                    runtime_seconds = time.perf_counter() - start

                    absolute_error = abs(
                        computed_integral - reference_integral
                    )
                    relative_error = guarded_relative_error(
                        computed_integral,
                        reference_integral,
                    )

                    perturbation_rng = np.random.default_rng(
                        perturbation_seed + basis_index
                    )
                    perturbed_coefficients = perturb_coefficients(
                        basis_coefficients,
                        relative_magnitude=perturbation_magnitude,
                        rng=perturbation_rng,
                        dtype=dtype,
                    )
                    perturbed_integral = integrate_in_basis(
                        perturbed_coefficients,
                        basis=basis,
                        scale=scale,
                    )
                    sensitivity = perturbation_sensitivity(
                        original_integral=computed_integral,
                        perturbed_integral=perturbed_integral,
                        perturbation_magnitude=perturbation_magnitude,
                    )

                    numeric_values = np.asarray(
                        [
                            reference_integral,
                            computed_integral,
                            absolute_error,
                            relative_error,
                            sigma_max,
                            sigma_min,
                            condition_number,
                            perturbed_integral,
                            sensitivity,
                            runtime_seconds,
                        ],
                        dtype=np.float64,
                    )

                    if not np.isfinite(numeric_values).all():
                        raise FloatingPointError(
                            f"non-finite result for {basis}, "
                            f"n={dimension}, d={degree}, trial={trial}"
                        )

                    records.append(
                        BenchmarkRecord(
                            basis=basis,
                            dimension=dimension,
                            degree=degree,
                            basis_count=basis_count,
                            trial=trial,
                            seed=trial_seed,
                            dtype=dtype.name,
                            scale=scale,
                            num_points=num_points,
                            matrix_rows=matrix.shape[0],
                            matrix_columns=matrix.shape[1],
                            matrix_rank=rank,
                            sigma_max=sigma_max,
                            sigma_min=sigma_min,
                            condition_number=condition_number,
                            reference_integral=reference_integral,
                            computed_integral=computed_integral,
                            absolute_error=absolute_error,
                            relative_error=relative_error,
                            perturbation_magnitude=perturbation_magnitude,
                            perturbed_integral=perturbed_integral,
                            perturbation_sensitivity=sensitivity,
                            runtime_seconds=runtime_seconds,
                        )
                    )

                    print(
                        f"n={dimension} d={degree:2d} trial={trial} "
                        f"basis={basis:9s} "
                        f"M={basis_count:4d} "
                        f"cond={condition_number:.3e} "
                        f"relerr={relative_error:.3e} "
                        f"sens={sensitivity:.3e} "
                        f"time={1e3 * runtime_seconds:.3f} ms"
                    )

    return records


def summarise_results(data: pd.DataFrame) -> dict[str, object]:
    """Create a compact machine-readable benchmark summary."""
    grouped = (
        data.groupby(
            ["basis", "dimension", "degree"],
            as_index=False,
        )
        .agg(
            trials=("trial", "nunique"),
            basis_count=("basis_count", "first"),
            condition_median=("condition_number", "median"),
            condition_q1=("condition_number", lambda x: x.quantile(0.25)),
            condition_q3=("condition_number", lambda x: x.quantile(0.75)),
            relative_error_median=("relative_error", "median"),
            sensitivity_median=("perturbation_sensitivity", "median"),
            runtime_median_seconds=("runtime_seconds", "median"),
            minimum_rank=("matrix_rank", "min"),
        )
    )

    return {
        "record_count": int(len(data)),
        "non_finite_count": int(
            np.size(data.select_dtypes(include=[np.number]).to_numpy())
            - np.isfinite(
                data.select_dtypes(include=[np.number]).to_numpy(
                    dtype=np.float64
                )
            ).sum()
        ),
        "bases": sorted(data["basis"].unique().tolist()),
        "dimensions": sorted(
            int(value) for value in data["dimension"].unique()
        ),
        "degrees_by_dimension": {
            str(int(dimension)): sorted(
                int(value)
                for value in subset["degree"].unique()
            )
            for dimension, subset in data.groupby("dimension")
        },
        "aggregates": grouped.to_dict(orient="records"),
    }


def write_outputs(
    records: Sequence[BenchmarkRecord],
    *,
    output_path: Path,
    summary_path: Path,
) -> None:
    """Write the detailed CSV and aggregate JSON summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.DataFrame(asdict(record) for record in records)

    expected_columns = [field.name for field in BenchmarkRecord.__dataclass_fields__.values()]
    data = data[expected_columns]

    data.to_csv(output_path, index=False)

    summary = summarise_results(data)
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Wrote {len(data)} records to {output_path}")
    print(f"Wrote aggregate summary to {summary_path}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run the unified n-dimensional simplex benchmark."
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Dimensions to evaluate. Default: 1 2 3 4 5.",
    )
    parser.add_argument(
        "--maximum-degree",
        type=int,
        default=None,
        help="Optionally remove degrees above this value.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Trials per basis-dimension-degree configuration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scaled-simplex size s. Default: 1.0.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Arithmetic precision used by sampled matrices and coefficients.",
    )
    parser.add_argument(
        "--oversampling-factor",
        type=float,
        default=2.0,
        help="Number of points divided by basis count. Must be >= 1.",
    )
    parser.add_argument(
        "--perturbation-magnitude",
        type=float,
        default=1e-8,
        help="Relative coefficient perturbation magnitude delta.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/simplex/dimension_results.csv"),
        help="Detailed CSV output path.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/simplex/dimension_summary.json"),
        help="Aggregate JSON output path.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    grid = resolve_grid(
        dimensions=args.dimensions,
        maximum_degree=args.maximum_degree,
    )

    print("Unified n-dimensional simplex benchmark")
    print(f"dimensions: {list(grid)}")
    print(f"degree grid: {grid}")
    print(f"trials: {args.trials}")
    print(f"dtype: {dtype.name}")
    print(f"scale: {args.scale}")
    print(f"oversampling factor: {args.oversampling_factor}")
    print(f"perturbation magnitude: {args.perturbation_magnitude}")
    print()

    records = run_benchmark(
        grid=grid,
        trials=args.trials,
        seed=args.seed,
        scale=args.scale,
        dtype=dtype,
        oversampling_factor=args.oversampling_factor,
        perturbation_magnitude=args.perturbation_magnitude,
    )

    write_outputs(
        records,
        output_path=args.output,
        summary_path=args.summary,
    )


if __name__ == "__main__":
    main()
