"""
Run downstream constrained-probability benchmarks.

This module evaluates whether numerical differences between polynomial basis
representations propagate into quantities used by constrained probabilistic
inference:

1. Partition functions:
       Z = integral_C p(x) dx

2. Normalised probability queries:
       Q(A) = integral_A p(x) dx / integral_C p(x) dx

3. Coefficient-perturbation sensitivity.

The benchmark uses non-negative polynomial densities of the form

    p(x) = q(x)^2 + eta,

where q is a randomly generated total-degree polynomial. The density is
constructed once in a monomial representation and converted into Legendre and
Chebyshev product bases. All three representations therefore describe the same
mathematical density, subject only to floating-point conversion error.

The primary domain is the n-dimensional unit simplex

    Delta_n = {x_i >= 0, sum_i x_i <= 1}.

Query regions are axis-scaled simplices

    A_tau = {x_i >= 0, sum_i x_i <= tau},

where 0 < tau <= 1. Analytical monomial integration provides a common reference
for both the complete domain and each query region.

Example
-------
Run the default benchmark:

    python -m analysis.downstream.run_downstream_benchmark

Run a smaller smoke test:

    python -m analysis.downstream.run_downstream_benchmark \\
        --dimensions 1 2 \\
        --degrees 1 2 3 \\
        --trials 2 \\
        --dtypes float32 float64

Outputs
-------
By default, the benchmark writes:

    results/downstream/downstream_results.csv
    results/downstream/downstream_summary.json

The CSV contains one row per

    dimension x degree x trial x dtype x basis x query region.

The JSON file contains the configuration, validation counts, aggregate metrics,
and extrema.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# Constants
# =============================================================================

SUPPORTED_BASES: tuple[str, ...] = (
    "monomial",
    "legendre",
    "chebyshev",
)

SUPPORTED_DTYPES: Mapping[str, np.dtype[Any]] = {
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}

DEFAULT_DIMENSIONS: tuple[int, ...] = (1, 2, 3)
DEFAULT_DEGREES: tuple[int, ...] = (0, 1, 2, 3, 5, 8)
DEFAULT_DTYPES: tuple[str, ...] = ("float32", "float64")
DEFAULT_QUERY_SCALES: tuple[float, ...] = (0.25, 0.50, 0.75)

DEFAULT_RESULTS_PATH = Path("results/downstream/downstream_results.csv")
DEFAULT_SUMMARY_PATH = Path("results/downstream/downstream_summary.json")

SCHEMA_VERSION = "1.0"


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass(frozen=True)
class BenchmarkConfiguration:
    """Complete configuration recorded in the summary output."""

    dimensions: tuple[int, ...]
    degrees: tuple[int, ...]
    bases: tuple[str, ...]
    dtypes: tuple[str, ...]
    trials: int
    query_scales: tuple[float, ...]
    density_offset: float
    coefficient_scale: float
    perturbation_magnitude: float
    seed: int
    results_path: str
    summary_path: str
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class BasisRepresentation:
    """Polynomial coefficients and metadata for one basis."""

    basis: str
    dimension: int
    degree: int
    multi_indices: tuple[tuple[int, ...], ...]
    coefficients_float64: np.ndarray


@dataclass(frozen=True)
class IntegralEvaluation:
    """Integral value and diagnostic timings."""

    value: float
    basis_evaluation_ms: float
    accumulation_ms: float
    total_ms: float


# =============================================================================
# General validation helpers
# =============================================================================

def require_positive_integer(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive; received {value}")


def require_non_negative_integer(value: int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative; received {value}")


def require_finite_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{name} must be finite and positive; received {value}"
        )


def require_finite_non_negative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"{name} must be finite and non-negative; received {value}"
        )


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def to_builtin(value: Any) -> Any:
    """
    Convert NumPy and dataclass values into JSON-serialisable Python values.
    """

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "__dataclass_fields__"):
        return {
            key: to_builtin(item)
            for key, item in asdict(value).items()
        }

    if isinstance(value, Mapping):
        return {
            str(key): to_builtin(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]

    if isinstance(value, list):
        return [to_builtin(item) for item in value]

    return value


# =============================================================================
# Multi-index utilities
# =============================================================================

def compositions(total: int, length: int) -> Iterator[tuple[int, ...]]:
    """
    Yield ordered non-negative integer compositions of ``total``.

    Examples
    --------
    compositions(2, 2) yields:

        (0, 2), (1, 1), (2, 0)
    """

    require_non_negative_integer(total, "total")
    require_positive_integer(length, "length")

    if length == 1:
        yield (total,)
        return

    for first in range(total + 1):
        for rest in compositions(total - first, length - 1):
            yield (first,) + rest


def total_degree_multi_indices(
    dimension: int,
    maximum_degree: int,
) -> tuple[tuple[int, ...], ...]:
    """
    Return every multi-index with total degree at most ``maximum_degree``.

    Indices are ordered first by total degree and then lexicographically
    according to the composition generator.
    """

    require_positive_integer(dimension, "dimension")
    require_non_negative_integer(maximum_degree, "maximum_degree")

    indices: list[tuple[int, ...]] = []

    for total_degree in range(maximum_degree + 1):
        indices.extend(compositions(total_degree, dimension))

    expected = math.comb(dimension + maximum_degree, maximum_degree)

    if len(indices) != expected:
        raise RuntimeError(
            "Incorrect total-degree index count: "
            f"constructed {len(indices)}, expected {expected}"
        )

    return tuple(indices)


def index_lookup(
    indices: Sequence[tuple[int, ...]],
) -> dict[tuple[int, ...], int]:
    return {
        index: position
        for position, index in enumerate(indices)
    }


# =============================================================================
# Polynomial algebra in monomial coordinates
# =============================================================================

def convolve_total_degree_coefficients(
    left_coefficients: np.ndarray,
    left_indices: Sequence[tuple[int, ...]],
    right_coefficients: np.ndarray,
    right_indices: Sequence[tuple[int, ...]],
    output_indices: Sequence[tuple[int, ...]],
) -> np.ndarray:
    """
    Multiply two multivariate polynomials represented in monomial coordinates.
    """

    left = np.asarray(left_coefficients, dtype=np.float64)
    right = np.asarray(right_coefficients, dtype=np.float64)

    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("Polynomial coefficient arrays must be one-dimensional")

    if len(left) != len(left_indices):
        raise ValueError("Left coefficient and index counts differ")

    if len(right) != len(right_indices):
        raise ValueError("Right coefficient and index counts differ")

    output = np.zeros(len(output_indices), dtype=np.float64)
    lookup = index_lookup(output_indices)

    for left_position, left_index in enumerate(left_indices):
        left_value = float(left[left_position])

        if left_value == 0.0:
            continue

        for right_position, right_index in enumerate(right_indices):
            right_value = float(right[right_position])

            if right_value == 0.0:
                continue

            combined_index = tuple(
                left_power + right_power
                for left_power, right_power in zip(
                    left_index,
                    right_index,
                    strict=True,
                )
            )

            output_position = lookup.get(combined_index)

            if output_position is None:
                raise RuntimeError(
                    "Output index set does not contain product index "
                    f"{combined_index}"
                )

            output[output_position] += left_value * right_value

    return output


def construct_non_negative_density(
    dimension: int,
    base_degree: int,
    coefficient_scale: float,
    density_offset: float,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    tuple[tuple[int, ...], ...],
    np.ndarray,
    tuple[tuple[int, ...], ...],
]:
    """
    Construct p(x) = q(x)^2 + density_offset.

    Returns
    -------
    density_coefficients:
        Monomial coefficients of p.

    density_indices:
        Total-degree multi-indices for p.

    source_coefficients:
        Monomial coefficients of q.

    source_indices:
        Total-degree multi-indices for q.
    """

    require_positive_integer(dimension, "dimension")
    require_non_negative_integer(base_degree, "base_degree")
    require_finite_positive(coefficient_scale, "coefficient_scale")
    require_finite_positive(density_offset, "density_offset")

    source_indices = total_degree_multi_indices(
        dimension=dimension,
        maximum_degree=base_degree,
    )

    source_coefficients = rng.normal(
        loc=0.0,
        scale=coefficient_scale,
        size=len(source_indices),
    ).astype(np.float64)

    # Keep the constant component away from zero so that the generated
    # polynomial is not dominated entirely by cancellation.
    source_coefficients[0] += 1.0

    density_degree = 2 * base_degree
    density_indices = total_degree_multi_indices(
        dimension=dimension,
        maximum_degree=density_degree,
    )

    density_coefficients = convolve_total_degree_coefficients(
        left_coefficients=source_coefficients,
        left_indices=source_indices,
        right_coefficients=source_coefficients,
        right_indices=source_indices,
        output_indices=density_indices,
    )

    constant_position = index_lookup(density_indices)[(0,) * dimension]
    density_coefficients[constant_position] += density_offset

    return (
        density_coefficients,
        density_indices,
        source_coefficients,
        source_indices,
    )


# =============================================================================
# One-dimensional basis transformations
# =============================================================================

def shifted_legendre_power_coefficients(
    maximum_degree: int,
) -> tuple[np.ndarray, ...]:
    """
    Return power coefficients of P_k(2x - 1), k = 0, ..., maximum_degree.

    Each returned array is in increasing monomial order:

        coefficient[j] multiplies x**j.
    """

    require_non_negative_integer(maximum_degree, "maximum_degree")

    from numpy.polynomial import Polynomial
    from numpy.polynomial.legendre import Legendre

    canonical_map = Polynomial([-1.0, 2.0])
    output: list[np.ndarray] = []

    for degree in range(maximum_degree + 1):
        canonical_polynomial = Legendre.basis(degree).convert(
            kind=Polynomial
        )
        shifted_polynomial = canonical_polynomial(canonical_map)
        coefficients = np.asarray(
            shifted_polynomial.coef,
            dtype=np.float64,
        )

        padded = np.zeros(maximum_degree + 1, dtype=np.float64)
        padded[: len(coefficients)] = coefficients
        output.append(padded)

    return tuple(output)


def shifted_chebyshev_power_coefficients(
    maximum_degree: int,
) -> tuple[np.ndarray, ...]:
    """
    Return power coefficients of T_k(2x - 1), k = 0, ..., maximum_degree.
    """

    require_non_negative_integer(maximum_degree, "maximum_degree")

    from numpy.polynomial import Polynomial
    from numpy.polynomial.chebyshev import Chebyshev

    canonical_map = Polynomial([-1.0, 2.0])
    output: list[np.ndarray] = []

    for degree in range(maximum_degree + 1):
        canonical_polynomial = Chebyshev.basis(degree).convert(
            kind=Polynomial
        )
        shifted_polynomial = canonical_polynomial(canonical_map)
        coefficients = np.asarray(
            shifted_polynomial.coef,
            dtype=np.float64,
        )

        padded = np.zeros(maximum_degree + 1, dtype=np.float64)
        padded[: len(coefficients)] = coefficients
        output.append(padded)

    return tuple(output)


def one_dimensional_basis_power_coefficients(
    basis: str,
    maximum_degree: int,
) -> tuple[np.ndarray, ...]:
    """
    Return monomial expansions of one-dimensional basis functions on [0, 1].
    """

    if basis == "monomial":
        output: list[np.ndarray] = []

        for degree in range(maximum_degree + 1):
            coefficients = np.zeros(
                maximum_degree + 1,
                dtype=np.float64,
            )
            coefficients[degree] = 1.0
            output.append(coefficients)

        return tuple(output)

    if basis == "legendre":
        return shifted_legendre_power_coefficients(maximum_degree)

    if basis == "chebyshev":
        return shifted_chebyshev_power_coefficients(maximum_degree)

    raise ValueError(
        f"Unsupported basis {basis!r}; expected one of {SUPPORTED_BASES}"
    )


# =============================================================================
# Multivariate basis transformation
# =============================================================================

def build_basis_to_monomial_matrix(
    basis: str,
    dimension: int,
    maximum_degree: int,
    indices: Sequence[tuple[int, ...]],
) -> np.ndarray:
    """
    Build the transformation from basis coefficients to monomial coefficients.

    If ``c_basis`` contains product-basis coefficients, then

        c_monomial = transformation @ c_basis.

    The matrix uses the same total-degree index set for rows and columns.
    """

    if basis not in SUPPORTED_BASES:
        raise ValueError(
            f"Unsupported basis {basis!r}; expected one of {SUPPORTED_BASES}"
        )

    require_positive_integer(dimension, "dimension")
    require_non_negative_integer(maximum_degree, "maximum_degree")

    expected_indices = total_degree_multi_indices(
        dimension=dimension,
        maximum_degree=maximum_degree,
    )

    if tuple(indices) != expected_indices:
        raise ValueError(
            "The supplied multi-index ordering does not match the runner's "
            "canonical total-degree ordering"
        )

    basis_count = len(indices)

    if basis == "monomial":
        return np.eye(basis_count, dtype=np.float64)

    one_dimensional = one_dimensional_basis_power_coefficients(
        basis=basis,
        maximum_degree=maximum_degree,
    )

    lookup = index_lookup(indices)
    transformation = np.zeros(
        (basis_count, basis_count),
        dtype=np.float64,
    )

    for column, basis_index in enumerate(indices):
        # Start with the zero-dimensional constant product.
        partial: dict[tuple[int, ...], float] = {
            tuple(): 1.0
        }

        for coordinate_degree in basis_index:
            coordinate_coefficients = one_dimensional[coordinate_degree]
            next_partial: dict[tuple[int, ...], float] = {}

            for prefix, prefix_value in partial.items():
                for power, coefficient in enumerate(
                    coordinate_coefficients
                ):
                    coefficient_value = float(coefficient)

                    if coefficient_value == 0.0:
                        continue

                    extended_index = prefix + (power,)
                    next_partial[extended_index] = (
                        next_partial.get(extended_index, 0.0)
                        + prefix_value * coefficient_value
                    )

            partial = next_partial

        for monomial_index, coefficient in partial.items():
            if sum(monomial_index) > maximum_degree:
                continue

            row = lookup.get(monomial_index)

            if row is None:
                raise RuntimeError(
                    "Missing monomial index in transformation: "
                    f"{monomial_index}"
                )

            transformation[row, column] += coefficient

    return transformation


def convert_monomial_coefficients_to_basis(
    monomial_coefficients: np.ndarray,
    transformation: np.ndarray,
) -> np.ndarray:
    """
    Solve T c_basis = c_monomial in float64.

    The transformation is square and theoretically nonsingular because each
    basis spans the same total-degree polynomial space.
    """

    coefficients = np.asarray(
        monomial_coefficients,
        dtype=np.float64,
    )
    matrix = np.asarray(
        transformation,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError("Transformation must be two-dimensional")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Transformation must be square")

    if len(coefficients) != matrix.shape[0]:
        raise ValueError(
            "Coefficient count does not match transformation size"
        )

    converted = np.linalg.solve(matrix, coefficients)

    if not np.all(np.isfinite(converted)):
        raise FloatingPointError(
            "Basis conversion produced non-finite coefficients"
        )

    return converted


def construct_basis_representations(
    monomial_coefficients: np.ndarray,
    dimension: int,
    degree: int,
    indices: tuple[tuple[int, ...], ...],
) -> dict[str, BasisRepresentation]:
    """
    Convert one monomial polynomial into every supported basis.
    """

    representations: dict[str, BasisRepresentation] = {}

    for basis in SUPPORTED_BASES:
        transformation = build_basis_to_monomial_matrix(
            basis=basis,
            dimension=dimension,
            maximum_degree=degree,
            indices=indices,
        )

        coefficients = convert_monomial_coefficients_to_basis(
            monomial_coefficients=monomial_coefficients,
            transformation=transformation,
        )

        representations[basis] = BasisRepresentation(
            basis=basis,
            dimension=dimension,
            degree=degree,
            multi_indices=indices,
            coefficients_float64=coefficients,
        )

    return representations


# =============================================================================
# Basis evaluation
# =============================================================================

def evaluate_shifted_legendre_values(
    coordinates: np.ndarray,
    maximum_degree: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """
    Evaluate shifted Legendre polynomials P_k(2x - 1).

    Returns an array with shape:

        (number_of_points, dimension, maximum_degree + 1)
    """

    points = np.asarray(coordinates, dtype=dtype)

    if points.ndim != 2:
        raise ValueError("Coordinates must have shape (N, dimension)")

    canonical = np.asarray(
        2.0 * points - 1.0,
        dtype=dtype,
    )

    values = np.empty(
        (
            points.shape[0],
            points.shape[1],
            maximum_degree + 1,
        ),
        dtype=dtype,
    )

    values[..., 0] = dtype.type(1.0)

    if maximum_degree == 0:
        return values

    values[..., 1] = canonical

    for degree in range(1, maximum_degree):
        numerator = (
            dtype.type(2 * degree + 1)
            * canonical
            * values[..., degree]
            - dtype.type(degree)
            * values[..., degree - 1]
        )
        values[..., degree + 1] = (
            numerator / dtype.type(degree + 1)
        )

    return values


def evaluate_shifted_chebyshev_values(
    coordinates: np.ndarray,
    maximum_degree: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """
    Evaluate shifted Chebyshev polynomials T_k(2x - 1).
    """

    points = np.asarray(coordinates, dtype=dtype)

    if points.ndim != 2:
        raise ValueError("Coordinates must have shape (N, dimension)")

    canonical = np.asarray(
        2.0 * points - 1.0,
        dtype=dtype,
    )

    values = np.empty(
        (
            points.shape[0],
            points.shape[1],
            maximum_degree + 1,
        ),
        dtype=dtype,
    )

    values[..., 0] = dtype.type(1.0)

    if maximum_degree == 0:
        return values

    values[..., 1] = canonical

    for degree in range(1, maximum_degree):
        values[..., degree + 1] = (
            dtype.type(2.0)
            * canonical
            * values[..., degree]
            - values[..., degree - 1]
        )

    return values


def evaluate_monomial_values(
    coordinates: np.ndarray,
    maximum_degree: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """
    Evaluate powers x^k for k = 0, ..., maximum_degree.
    """

    points = np.asarray(coordinates, dtype=dtype)

    if points.ndim != 2:
        raise ValueError("Coordinates must have shape (N, dimension)")

    values = np.empty(
        (
            points.shape[0],
            points.shape[1],
            maximum_degree + 1,
        ),
        dtype=dtype,
    )

    values[..., 0] = dtype.type(1.0)

    for degree in range(1, maximum_degree + 1):
        values[..., degree] = (
            values[..., degree - 1] * points
        )

    return values


def evaluate_one_dimensional_basis_table(
    basis: str,
    coordinates: np.ndarray,
    maximum_degree: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    if basis == "monomial":
        return evaluate_monomial_values(
            coordinates=coordinates,
            maximum_degree=maximum_degree,
            dtype=dtype,
        )

    if basis == "legendre":
        return evaluate_shifted_legendre_values(
            coordinates=coordinates,
            maximum_degree=maximum_degree,
            dtype=dtype,
        )

    if basis == "chebyshev":
        return evaluate_shifted_chebyshev_values(
            coordinates=coordinates,
            maximum_degree=maximum_degree,
            dtype=dtype,
        )

    raise ValueError(
        f"Unsupported basis {basis!r}; expected one of {SUPPORTED_BASES}"
    )


def evaluate_basis_matrix(
    basis: str,
    coordinates: np.ndarray,
    indices: Sequence[tuple[int, ...]],
    maximum_degree: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """
    Evaluate the multivariate product basis at a collection of points.
    """

    points = np.asarray(coordinates, dtype=dtype)

    if points.ndim != 2:
        raise ValueError("Coordinates must have shape (N, dimension)")

    if len(indices) == 0:
        raise ValueError("At least one basis index is required")

    dimension = len(indices[0])

    if points.shape[1] != dimension:
        raise ValueError(
            f"Point dimension {points.shape[1]} does not match "
            f"basis dimension {dimension}"
        )

    univariate_values = evaluate_one_dimensional_basis_table(
        basis=basis,
        coordinates=points,
        maximum_degree=maximum_degree,
        dtype=dtype,
    )

    matrix = np.ones(
        (points.shape[0], len(indices)),
        dtype=dtype,
    )

    for column, multi_index in enumerate(indices):
        column_values = np.ones(points.shape[0], dtype=dtype)

        for coordinate, degree in enumerate(multi_index):
            column_values *= univariate_values[
                :,
                coordinate,
                degree,
            ]

        matrix[:, column] = column_values

    return matrix


def evaluate_polynomial(
    representation: BasisRepresentation,
    coordinates: np.ndarray,
    dtype_name: str,
) -> np.ndarray:
    """
    Evaluate a basis representation in the requested arithmetic precision.
    """

    dtype = SUPPORTED_DTYPES[dtype_name]

    matrix = evaluate_basis_matrix(
        basis=representation.basis,
        coordinates=coordinates,
        indices=representation.multi_indices,
        maximum_degree=representation.degree,
        dtype=dtype,
    )

    coefficients = np.asarray(
        representation.coefficients_float64,
        dtype=dtype,
    )

    values = matrix @ coefficients

    return np.asarray(values, dtype=dtype)


# =============================================================================
# Analytical simplex integration
# =============================================================================

def unit_simplex_monomial_integral(
    exponents: Sequence[int],
) -> float:
    """
    Compute the analytical integral of one monomial over the unit simplex.

    integral_Delta x^alpha dx
        = prod_i Gamma(alpha_i + 1)
          / Gamma(n + sum_i alpha_i + 1)
    """

    if len(exponents) == 0:
        raise ValueError("At least one exponent is required")

    if any(exponent < 0 for exponent in exponents):
        raise ValueError("Exponents must be non-negative")

    dimension = len(exponents)

    log_numerator = sum(
        math.lgamma(exponent + 1.0)
        for exponent in exponents
    )

    log_denominator = math.lgamma(
        dimension + sum(exponents) + 1.0
    )

    return math.exp(log_numerator - log_denominator)


def scaled_simplex_monomial_integral(
    exponents: Sequence[int],
    scale: float,
) -> float:
    """
    Integrate a monomial over Delta_n(scale).

    Under x = scale * y, the integral acquires the factor

        scale ** (n + total_degree).
    """

    require_finite_positive(scale, "scale")

    dimension = len(exponents)
    total_degree = sum(exponents)

    return (
        scale ** (dimension + total_degree)
        * unit_simplex_monomial_integral(exponents)
    )


def analytical_monomial_polynomial_integral(
    coefficients: np.ndarray,
    indices: Sequence[tuple[int, ...]],
    simplex_scale: float,
) -> float:
    """
    Compute the analytical integral of a monomial polynomial.
    """

    coefficient_array = np.asarray(
        coefficients,
        dtype=np.float64,
    )

    if coefficient_array.ndim != 1:
        raise ValueError("Coefficients must be one-dimensional")

    if len(coefficient_array) != len(indices):
        raise ValueError("Coefficient and index counts differ")

    terms = [
        float(coefficient)
        * scaled_simplex_monomial_integral(
            exponents=index,
            scale=simplex_scale,
        )
        for coefficient, index in zip(
            coefficient_array,
            indices,
            strict=True,
        )
    ]

    return math.fsum(terms)


def basis_function_integrals_over_scaled_simplex(
    basis: str,
    dimension: int,
    degree: int,
    indices: Sequence[tuple[int, ...]],
    simplex_scale: float,
) -> np.ndarray:
    """
    Return one analytical integral for every product-basis function.

    Basis functions are first expanded in monomial coordinates. The resulting
    integration vector is then transformed back to the selected basis.
    """

    transformation = build_basis_to_monomial_matrix(
        basis=basis,
        dimension=dimension,
        maximum_degree=degree,
        indices=indices,
    )

    monomial_integrals = np.asarray(
        [
            scaled_simplex_monomial_integral(
                exponents=index,
                scale=simplex_scale,
            )
            for index in indices
        ],
        dtype=np.float64,
    )

    # If c_monomial = T c_basis, then:
    #
    # integral = monomial_integrals^T T c_basis
    #          = (T^T monomial_integrals)^T c_basis.
    basis_integrals = transformation.T @ monomial_integrals

    return np.asarray(basis_integrals, dtype=np.float64)


def integrate_basis_representation(
    representation: BasisRepresentation,
    simplex_scale: float,
    dtype_name: str,
    coefficients_override: np.ndarray | None = None,
) -> IntegralEvaluation:
    """
    Integrate a basis representation using analytical basis moments.

    The basis moments are generated from a float64 transformation, but their
    multiplication and accumulation are performed in the requested dtype. This
    exposes coefficient-rounding and accumulation effects while retaining an
    independent analytical monomial reference.
    """

    dtype = SUPPORTED_DTYPES[dtype_name]

    total_start = time.perf_counter_ns()

    basis_start = time.perf_counter_ns()

    basis_integrals_float64 = (
        basis_function_integrals_over_scaled_simplex(
            basis=representation.basis,
            dimension=representation.dimension,
            degree=representation.degree,
            indices=representation.multi_indices,
            simplex_scale=simplex_scale,
        )
    )

    basis_integrals = np.asarray(
        basis_integrals_float64,
        dtype=dtype,
    )

    basis_end = time.perf_counter_ns()

    accumulation_start = time.perf_counter_ns()

    if coefficients_override is None:
        coefficients = np.asarray(
            representation.coefficients_float64,
            dtype=dtype,
        )
    else:
        coefficients = np.asarray(
            coefficients_override,
            dtype=dtype,
        )

    if coefficients.shape != basis_integrals.shape:
        raise ValueError(
            "Coefficient and basis-integral vectors have different shapes"
        )

    products = np.asarray(
        coefficients * basis_integrals,
        dtype=dtype,
    )

    value = float(np.sum(products, dtype=dtype))

    accumulation_end = time.perf_counter_ns()
    total_end = accumulation_end

    return IntegralEvaluation(
        value=value,
        basis_evaluation_ms=(
            basis_end - basis_start
        ) / 1_000_000.0,
        accumulation_ms=(
            accumulation_end - accumulation_start
        ) / 1_000_000.0,
        total_ms=(
            total_end - total_start
        ) / 1_000_000.0,
    )


# =============================================================================
# Numerical diagnostics
# =============================================================================

def safe_relative_error(
    estimate: float,
    reference: float,
    epsilon: float,
) -> float:
    denominator = max(abs(reference), epsilon)
    return abs(estimate - reference) / denominator


def safe_absolute_error(
    estimate: float,
    reference: float,
) -> float:
    return abs(estimate - reference)


def normalised_probability(
    numerator: float,
    denominator: float,
    epsilon: float,
) -> float:
    if abs(denominator) <= epsilon:
        raise FloatingPointError(
            "Cannot normalise by a numerically zero partition function"
        )

    return numerator / denominator


def perturb_coefficients(
    coefficients: np.ndarray,
    magnitude: float,
    rng: np.random.Generator,
    dtype_name: str,
) -> tuple[np.ndarray, float, bool]:
    """
    Apply a controlled relative coefficient perturbation.

    Returns
    -------
    perturbed:
        Perturbed coefficients in the requested dtype.

    realised_relative_norm:
        ||perturbed - original|| / max(||original||, eps), measured after
        casting into the requested dtype.

    rounded_away:
        True when the requested perturbation causes no representable
        coefficient change.
    """

    require_finite_positive(magnitude, "magnitude")

    dtype = SUPPORTED_DTYPES[dtype_name]
    original = np.asarray(coefficients, dtype=dtype)

    direction_float64 = rng.normal(size=original.shape)
    direction_norm = float(np.linalg.norm(direction_float64))

    if direction_norm == 0.0:
        raise RuntimeError("Random perturbation direction has zero norm")

    direction_float64 /= direction_norm

    original_norm = float(
        np.linalg.norm(
            np.asarray(original, dtype=np.float64)
        )
    )

    scale = max(
        original_norm,
        float(np.finfo(dtype).tiny),
    )

    perturbation_float64 = (
        magnitude * scale * direction_float64
    )

    perturbed = np.asarray(
        np.asarray(original, dtype=np.float64)
        + perturbation_float64,
        dtype=dtype,
    )

    realised_difference = np.asarray(
        perturbed - original,
        dtype=np.float64,
    )

    realised_relative_norm = (
        float(np.linalg.norm(realised_difference))
        / max(original_norm, float(np.finfo(dtype).tiny))
    )

    rounded_away = bool(np.array_equal(perturbed, original))

    return (
        perturbed,
        realised_relative_norm,
        rounded_away,
    )


def perturbation_sensitivity(
    perturbed_value: float,
    original_value: float,
    requested_magnitude: float,
    epsilon: float,
) -> float:
    denominator = (
        requested_magnitude
        * max(abs(original_value), epsilon)
    )

    return abs(perturbed_value - original_value) / denominator


def estimate_representation_condition_number(
    representation: BasisRepresentation,
    sample_points: np.ndarray,
    dtype_name: str,
) -> float:
    """
    Estimate the 2-norm condition number of the sampled basis matrix.

    SVD is evaluated on float64 values obtained after constructing the matrix
    in the requested arithmetic type. This preserves float32 rounding while
    avoiding unsupported or inconsistent low-precision LAPACK paths.
    """

    dtype = SUPPORTED_DTYPES[dtype_name]

    matrix = evaluate_basis_matrix(
        basis=representation.basis,
        coordinates=sample_points,
        indices=representation.multi_indices,
        maximum_degree=representation.degree,
        dtype=dtype,
    )

    singular_values = np.linalg.svd(
        np.asarray(matrix, dtype=np.float64),
        compute_uv=False,
    )

    if len(singular_values) == 0:
        raise RuntimeError("No singular values were returned")

    smallest = float(singular_values[-1])
    largest = float(singular_values[0])

    if smallest == 0.0:
        return math.inf

    return largest / smallest


def estimate_rank_fraction(
    representation: BasisRepresentation,
    sample_points: np.ndarray,
    dtype_name: str,
) -> tuple[int, int, float]:
    """
    Estimate numerical rank using a dtype-specific tolerance.
    """

    dtype = SUPPORTED_DTYPES[dtype_name]

    matrix = evaluate_basis_matrix(
        basis=representation.basis,
        coordinates=sample_points,
        indices=representation.multi_indices,
        maximum_degree=representation.degree,
        dtype=dtype,
    )

    matrix_float64 = np.asarray(matrix, dtype=np.float64)
    singular_values = np.linalg.svd(
        matrix_float64,
        compute_uv=False,
    )

    tolerance = (
        max(matrix.shape)
        * float(np.finfo(dtype).eps)
        * float(singular_values[0])
    )

    rank = int(np.count_nonzero(singular_values > tolerance))
    columns = int(matrix.shape[1])
    fraction = rank / columns

    return rank, columns, fraction


# =============================================================================
# Point generation and validation
# =============================================================================

def sample_unit_simplex(
    number_of_points: int,
    dimension: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw approximately uniform points from the unit simplex.

    An exponential-variable construction is used. A slack coordinate is
    generated and discarded.
    """

    require_positive_integer(number_of_points, "number_of_points")
    require_positive_integer(dimension, "dimension")

    exponential_samples = rng.exponential(
        scale=1.0,
        size=(number_of_points, dimension + 1),
    )

    normalised = exponential_samples / np.sum(
        exponential_samples,
        axis=1,
        keepdims=True,
    )

    points = normalised[:, :dimension]

    if not np.all(points >= 0.0):
        raise RuntimeError("Generated simplex points contain negative values")

    if not np.all(np.sum(points, axis=1) <= 1.0 + 1e-12):
        raise RuntimeError("Generated points lie outside the unit simplex")

    return np.asarray(points, dtype=np.float64)


def deterministic_simplex_validation_points(
    dimension: int,
    number_of_points: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return sample_unit_simplex(
        number_of_points=number_of_points,
        dimension=dimension,
        rng=rng,
    )


def validate_basis_equivalence(
    monomial_representation: BasisRepresentation,
    alternative_representation: BasisRepresentation,
    validation_points: np.ndarray,
) -> float:
    """
    Return maximum float64 pointwise disagreement between two representations.
    """

    monomial_values = evaluate_polynomial(
        representation=monomial_representation,
        coordinates=validation_points,
        dtype_name="float64",
    )

    alternative_values = evaluate_polynomial(
        representation=alternative_representation,
        coordinates=validation_points,
        dtype_name="float64",
    )

    difference = np.abs(
        np.asarray(monomial_values, dtype=np.float64)
        - np.asarray(alternative_values, dtype=np.float64)
    )

    return float(np.max(difference))


def validate_density_non_negative(
    representation: BasisRepresentation,
    validation_points: np.ndarray,
) -> tuple[float, int]:
    """
    Evaluate the density on validation points and report the minimum.
    """

    values = evaluate_polynomial(
        representation=representation,
        coordinates=validation_points,
        dtype_name="float64",
    )

    minimum = float(np.min(values))
    negative_count = int(np.count_nonzero(values < -1e-10))

    return minimum, negative_count


# =============================================================================
# Benchmark execution
# =============================================================================

def configuration_seed(
    base_seed: int,
    dimension: int,
    base_degree: int,
    trial: int,
) -> int:
    """
    Derive a deterministic per-configuration seed without Python hash().
    """

    sequence = np.random.SeedSequence(
        [
            int(base_seed),
            int(dimension),
            int(base_degree),
            int(trial),
        ]
    )

    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def perturbation_seed(
    configuration_seed_value: int,
    basis_position: int,
    dtype_position: int,
    query_position: int,
) -> int:
    sequence = np.random.SeedSequence(
        [
            int(configuration_seed_value),
            int(basis_position),
            int(dtype_position),
            int(query_position),
            91_771,
        ]
    )

    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def make_base_record(
    *,
    configuration: BenchmarkConfiguration,
    dimension: int,
    requested_base_degree: int,
    density_degree: int,
    trial: int,
    trial_seed: int,
    basis: str,
    dtype_name: str,
    query_scale: float,
) -> dict[str, Any]:
    return {
        "schema_version": configuration.schema_version,
        "dimension": dimension,
        "source_polynomial_degree": requested_base_degree,
        "density_polynomial_degree": density_degree,
        "trial": trial,
        "trial_seed": trial_seed,
        "basis": basis,
        "dtype": dtype_name,
        "query_scale": query_scale,
        "density_offset": configuration.density_offset,
        "coefficient_scale": configuration.coefficient_scale,
        "requested_perturbation_magnitude": (
            configuration.perturbation_magnitude
        ),
    }


def run_single_configuration(
    *,
    configuration: BenchmarkConfiguration,
    dimension: int,
    base_degree: int,
    trial: int,
) -> list[dict[str, Any]]:
    """
    Run all bases, dtypes, and query scales for one density sample.
    """

    trial_seed = configuration_seed(
        base_seed=configuration.seed,
        dimension=dimension,
        base_degree=base_degree,
        trial=trial,
    )

    rng = np.random.default_rng(trial_seed)

    (
        density_monomial_coefficients,
        density_indices,
        source_coefficients,
        source_indices,
    ) = construct_non_negative_density(
        dimension=dimension,
        base_degree=base_degree,
        coefficient_scale=configuration.coefficient_scale,
        density_offset=configuration.density_offset,
        rng=rng,
    )

    density_degree = 2 * base_degree

    conversion_start = time.perf_counter_ns()

    representations = construct_basis_representations(
        monomial_coefficients=density_monomial_coefficients,
        dimension=dimension,
        degree=density_degree,
        indices=density_indices,
    )

    conversion_end = time.perf_counter_ns()
    total_conversion_ms = (
        conversion_end - conversion_start
    ) / 1_000_000.0

    basis_count = len(density_indices)
    source_basis_count = len(source_indices)

    validation_point_count = max(
        64,
        3 * basis_count,
    )

    validation_points = deterministic_simplex_validation_points(
        dimension=dimension,
        number_of_points=validation_point_count,
        seed=trial_seed + 11_003,
    )

    condition_point_count = max(
        basis_count + 4,
        2 * basis_count,
    )

    condition_points = deterministic_simplex_validation_points(
        dimension=dimension,
        number_of_points=condition_point_count,
        seed=trial_seed + 29_011,
    )

    reference_partition = analytical_monomial_polynomial_integral(
        coefficients=density_monomial_coefficients,
        indices=density_indices,
        simplex_scale=1.0,
    )

    if not math.isfinite(reference_partition):
        raise FloatingPointError(
            "Analytical partition function is not finite"
        )

    if reference_partition <= 0.0:
        raise FloatingPointError(
            "Constructed non-negative density has a non-positive "
            f"reference partition function: {reference_partition}"
        )

    query_references: dict[float, float] = {}

    for query_scale in configuration.query_scales:
        query_reference = analytical_monomial_polynomial_integral(
            coefficients=density_monomial_coefficients,
            indices=density_indices,
            simplex_scale=query_scale,
        )

        if not math.isfinite(query_reference):
            raise FloatingPointError(
                "Analytical query integral is not finite"
            )

        query_references[query_scale] = query_reference

    monomial_representation = representations["monomial"]

    equivalence_errors = {
        basis: validate_basis_equivalence(
            monomial_representation=monomial_representation,
            alternative_representation=representation,
            validation_points=validation_points,
        )
        for basis, representation in representations.items()
    }

    non_negativity = {
        basis: validate_density_non_negative(
            representation=representation,
            validation_points=validation_points,
        )
        for basis, representation in representations.items()
    }

    records: list[dict[str, Any]] = []

    machine_epsilons = {
        dtype_name: float(np.finfo(dtype).eps)
        for dtype_name, dtype in SUPPORTED_DTYPES.items()
    }

    for basis_position, basis in enumerate(configuration.bases):
        representation = representations[basis]

        for dtype_position, dtype_name in enumerate(
            configuration.dtypes
        ):
            epsilon = machine_epsilons[dtype_name]

            condition_number = (
                estimate_representation_condition_number(
                    representation=representation,
                    sample_points=condition_points,
                    dtype_name=dtype_name,
                )
            )

            (
                numerical_rank,
                maximum_rank,
                rank_fraction,
            ) = estimate_rank_fraction(
                representation=representation,
                sample_points=condition_points,
                dtype_name=dtype_name,
            )

            partition_evaluation = integrate_basis_representation(
                representation=representation,
                simplex_scale=1.0,
                dtype_name=dtype_name,
            )

            partition_error = safe_relative_error(
                estimate=partition_evaluation.value,
                reference=reference_partition,
                epsilon=epsilon,
            )

            for query_position, query_scale in enumerate(
                configuration.query_scales
            ):
                query_reference_integral = query_references[query_scale]

                reference_probability = normalised_probability(
                    numerator=query_reference_integral,
                    denominator=reference_partition,
                    epsilon=np.finfo(np.float64).tiny,
                )

                query_evaluation = integrate_basis_representation(
                    representation=representation,
                    simplex_scale=query_scale,
                    dtype_name=dtype_name,
                )

                estimated_probability = normalised_probability(
                    numerator=query_evaluation.value,
                    denominator=partition_evaluation.value,
                    epsilon=epsilon,
                )

                probability_absolute_error = safe_absolute_error(
                    estimate=estimated_probability,
                    reference=reference_probability,
                )

                probability_relative_error = safe_relative_error(
                    estimate=estimated_probability,
                    reference=reference_probability,
                    epsilon=epsilon,
                )

                current_perturbation_seed = perturbation_seed(
                    configuration_seed_value=trial_seed,
                    basis_position=basis_position,
                    dtype_position=dtype_position,
                    query_position=query_position,
                )

                perturbation_rng = np.random.default_rng(
                    current_perturbation_seed
                )

                (
                    perturbed_coefficients,
                    realised_relative_perturbation,
                    perturbation_rounded_away,
                ) = perturb_coefficients(
                    coefficients=representation.coefficients_float64,
                    magnitude=configuration.perturbation_magnitude,
                    rng=perturbation_rng,
                    dtype_name=dtype_name,
                )

                perturbed_partition = integrate_basis_representation(
                    representation=representation,
                    simplex_scale=1.0,
                    dtype_name=dtype_name,
                    coefficients_override=perturbed_coefficients,
                )

                perturbed_query = integrate_basis_representation(
                    representation=representation,
                    simplex_scale=query_scale,
                    dtype_name=dtype_name,
                    coefficients_override=perturbed_coefficients,
                )

                perturbed_probability = normalised_probability(
                    numerator=perturbed_query.value,
                    denominator=perturbed_partition.value,
                    epsilon=epsilon,
                )

                partition_perturbation_sensitivity = (
                    perturbation_sensitivity(
                        perturbed_value=perturbed_partition.value,
                        original_value=partition_evaluation.value,
                        requested_magnitude=(
                            configuration.perturbation_magnitude
                        ),
                        epsilon=epsilon,
                    )
                )

                query_integral_perturbation_sensitivity = (
                    perturbation_sensitivity(
                        perturbed_value=perturbed_query.value,
                        original_value=query_evaluation.value,
                        requested_magnitude=(
                            configuration.perturbation_magnitude
                        ),
                        epsilon=epsilon,
                    )
                )

                probability_perturbation_sensitivity = (
                    perturbation_sensitivity(
                        perturbed_value=perturbed_probability,
                        original_value=estimated_probability,
                        requested_magnitude=(
                            configuration.perturbation_magnitude
                        ),
                        epsilon=epsilon,
                    )
                )

                minimum_density_value, negative_density_count = (
                    non_negativity[basis]
                )

                record = make_base_record(
                    configuration=configuration,
                    dimension=dimension,
                    requested_base_degree=base_degree,
                    density_degree=density_degree,
                    trial=trial,
                    trial_seed=trial_seed,
                    basis=basis,
                    dtype_name=dtype_name,
                    query_scale=query_scale,
                )

                record.update(
                    {
                        "basis_count": basis_count,
                        "source_basis_count": source_basis_count,
                        "validation_point_count": (
                            validation_point_count
                        ),
                        "condition_point_count": condition_point_count,
                        "basis_conversion_total_ms": (
                            total_conversion_ms
                        ),
                        "maximum_basis_equivalence_error_float64": (
                            equivalence_errors[basis]
                        ),
                        "minimum_sampled_density_float64": (
                            minimum_density_value
                        ),
                        "negative_sampled_density_count": (
                            negative_density_count
                        ),
                        "condition_number": condition_number,
                        "numerical_rank": numerical_rank,
                        "maximum_rank": maximum_rank,
                        "rank_fraction": rank_fraction,
                        "reference_partition_function": (
                            reference_partition
                        ),
                        "estimated_partition_function": (
                            partition_evaluation.value
                        ),
                        "partition_function_relative_error": (
                            partition_error
                        ),
                        "reference_query_integral": (
                            query_reference_integral
                        ),
                        "estimated_query_integral": (
                            query_evaluation.value
                        ),
                        "query_integral_relative_error": (
                            safe_relative_error(
                                estimate=query_evaluation.value,
                                reference=query_reference_integral,
                                epsilon=epsilon,
                            )
                        ),
                        "reference_query_probability": (
                            reference_probability
                        ),
                        "estimated_query_probability": (
                            estimated_probability
                        ),
                        "query_probability_absolute_error": (
                            probability_absolute_error
                        ),
                        "query_probability_relative_error": (
                            probability_relative_error
                        ),
                        "perturbation_seed": (
                            current_perturbation_seed
                        ),
                        "realised_relative_coefficient_perturbation": (
                            realised_relative_perturbation
                        ),
                        "perturbation_rounded_away": (
                            perturbation_rounded_away
                        ),
                        "perturbed_partition_function": (
                            perturbed_partition.value
                        ),
                        "perturbed_query_integral": (
                            perturbed_query.value
                        ),
                        "perturbed_query_probability": (
                            perturbed_probability
                        ),
                        "partition_perturbation_sensitivity": (
                            partition_perturbation_sensitivity
                        ),
                        "query_integral_perturbation_sensitivity": (
                            query_integral_perturbation_sensitivity
                        ),
                        "query_probability_perturbation_sensitivity": (
                            probability_perturbation_sensitivity
                        ),
                        "partition_basis_moment_ms": (
                            partition_evaluation.basis_evaluation_ms
                        ),
                        "partition_accumulation_ms": (
                            partition_evaluation.accumulation_ms
                        ),
                        "partition_total_ms": (
                            partition_evaluation.total_ms
                        ),
                        "query_basis_moment_ms": (
                            query_evaluation.basis_evaluation_ms
                        ),
                        "query_accumulation_ms": (
                            query_evaluation.accumulation_ms
                        ),
                        "query_total_ms": (
                            query_evaluation.total_ms
                        ),
                        "perturbed_partition_total_ms": (
                            perturbed_partition.total_ms
                        ),
                        "perturbed_query_total_ms": (
                            perturbed_query.total_ms
                        ),
                        "total_downstream_evaluation_ms": (
                            partition_evaluation.total_ms
                            + query_evaluation.total_ms
                        ),
                        "source_coefficient_l2_norm": float(
                            np.linalg.norm(source_coefficients)
                        ),
                        "density_monomial_coefficient_l2_norm": float(
                            np.linalg.norm(
                                density_monomial_coefficients
                            )
                        ),
                        "basis_coefficient_l2_norm": float(
                            np.linalg.norm(
                                representation.coefficients_float64
                            )
                        ),
                        "finite_record": True,
                    }
                )

                numeric_values = [
                    value
                    for value in record.values()
                    if isinstance(value, (int, float, np.number))
                    and not isinstance(value, bool)
                ]

                finite_record = all(
                    math.isfinite(float(value))
                    for value in numeric_values
                    if value is not None
                )

                # Infinite condition number is scientifically meaningful if
                # the sampled matrix becomes exactly singular. It should not
                # invalidate every other metric in the row.
                if math.isinf(condition_number):
                    finite_without_condition = all(
                        math.isfinite(float(value))
                        for key, value in record.items()
                        if (
                            key != "condition_number"
                            and isinstance(
                                value,
                                (int, float, np.number),
                            )
                            and not isinstance(value, bool)
                            and value is not None
                        )
                    )
                    finite_record = finite_without_condition

                record["finite_record"] = finite_record
                records.append(record)

    return records


def run_benchmark(
    configuration: BenchmarkConfiguration,
) -> pd.DataFrame:
    """
    Execute the complete benchmark grid.
    """

    records: list[dict[str, Any]] = []

    configuration_count = (
        len(configuration.dimensions)
        * len(configuration.degrees)
        * configuration.trials
    )

    completed = 0
    benchmark_start = time.perf_counter()

    for dimension in configuration.dimensions:
        for base_degree in configuration.degrees:
            for trial in range(configuration.trials):
                completed += 1

                print(
                    "["
                    f"{completed:>4}/{configuration_count:<4}"
                    "] "
                    f"dimension={dimension}, "
                    f"source_degree={base_degree}, "
                    f"density_degree={2 * base_degree}, "
                    f"trial={trial}",
                    flush=True,
                )

                current_records = run_single_configuration(
                    configuration=configuration,
                    dimension=dimension,
                    base_degree=base_degree,
                    trial=trial,
                )

                records.extend(current_records)

    elapsed = time.perf_counter() - benchmark_start

    print(
        f"Completed {len(records)} downstream records "
        f"in {elapsed:.3f} seconds.",
        flush=True,
    )

    dataframe = pd.DataFrame.from_records(records)

    if dataframe.empty:
        raise RuntimeError("Benchmark produced no records")

    return dataframe


# =============================================================================
# Summaries
# =============================================================================

def finite_series_summary(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=np.float64))
    finite = numeric[finite_mask]

    if finite.empty:
        return {
            "count": 0,
            "median": None,
            "q1": None,
            "q3": None,
            "minimum": None,
            "maximum": None,
            "mean": None,
        }

    return {
        "count": int(finite.count()),
        "median": float(finite.median()),
        "q1": float(finite.quantile(0.25)),
        "q3": float(finite.quantile(0.75)),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
        "mean": float(finite.mean()),
    }


def grouped_metric_summary(
    dataframe: pd.DataFrame,
    metric: str,
    group_columns: Sequence[str],
) -> list[dict[str, Any]]:
    if metric not in dataframe.columns:
        raise KeyError(f"Metric column not found: {metric}")

    output: list[dict[str, Any]] = []

    grouped = dataframe.groupby(
        list(group_columns),
        dropna=False,
        sort=True,
    )

    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        record = {
            column: to_builtin(value)
            for column, value in zip(
                group_columns,
                keys,
                strict=True,
            )
        }

        record.update(
            finite_series_summary(group[metric])
        )

        output.append(record)

    return output


def row_for_maximum(
    dataframe: pd.DataFrame,
    metric: str,
    columns: Sequence[str],
) -> dict[str, Any] | None:
    numeric = pd.to_numeric(
        dataframe[metric],
        errors="coerce",
    )

    finite_mask = np.isfinite(
        numeric.to_numpy(dtype=np.float64)
    )

    if not np.any(finite_mask):
        return None

    finite_numeric = numeric[finite_mask]
    index = finite_numeric.idxmax()

    selected = dataframe.loc[index, list(columns)]

    return {
        column: to_builtin(selected[column])
        for column in columns
    }


def build_summary(
    dataframe: pd.DataFrame,
    configuration: BenchmarkConfiguration,
    elapsed_seconds: float,
) -> dict[str, Any]:
    metrics = (
        "partition_function_relative_error",
        "query_integral_relative_error",
        "query_probability_absolute_error",
        "query_probability_relative_error",
        "partition_perturbation_sensitivity",
        "query_integral_perturbation_sensitivity",
        "query_probability_perturbation_sensitivity",
        "condition_number",
        "rank_fraction",
        "total_downstream_evaluation_ms",
        "basis_conversion_total_ms",
        "maximum_basis_equivalence_error_float64",
    )

    aggregate_metrics = {
        metric: finite_series_summary(dataframe[metric])
        for metric in metrics
    }

    grouped_by_basis_and_dtype = {
        metric: grouped_metric_summary(
            dataframe=dataframe,
            metric=metric,
            group_columns=("basis", "dtype"),
        )
        for metric in metrics
    }

    grouped_by_basis_dtype_dimension = {
        metric: grouped_metric_summary(
            dataframe=dataframe,
            metric=metric,
            group_columns=("basis", "dtype", "dimension"),
        )
        for metric in (
            "partition_function_relative_error",
            "query_probability_absolute_error",
            "query_probability_relative_error",
            "rank_fraction",
            "total_downstream_evaluation_ms",
        )
    }

    rounded_away_count = int(
        dataframe["perturbation_rounded_away"].astype(bool).sum()
    )

    negative_density_records = int(
        (
            dataframe["negative_sampled_density_count"] > 0
        ).sum()
    )

    non_finite_record_count = int(
        (~dataframe["finite_record"].astype(bool)).sum()
    )

    return {
        "schema_version": configuration.schema_version,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "configuration": to_builtin(configuration),
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "execution": {
            "elapsed_seconds": elapsed_seconds,
            "record_count": int(len(dataframe)),
            "finite_record_count": int(
                dataframe["finite_record"].astype(bool).sum()
            ),
            "non_finite_record_count": non_finite_record_count,
            "unique_trials": int(
                dataframe[
                    [
                        "dimension",
                        "source_polynomial_degree",
                        "trial",
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            ),
        },
        "validation": {
            "negative_density_record_count": (
                negative_density_records
            ),
            "rounded_away_perturbation_count": (
                rounded_away_count
            ),
            "maximum_basis_equivalence_error_float64": (
                finite_series_summary(
                    dataframe[
                        "maximum_basis_equivalence_error_float64"
                    ]
                )
            ),
            "minimum_sampled_density_float64": (
                finite_series_summary(
                    dataframe["minimum_sampled_density_float64"]
                )
            ),
        },
        "aggregate_metrics": aggregate_metrics,
        "grouped_by_basis_and_dtype": (
            grouped_by_basis_and_dtype
        ),
        "grouped_by_basis_dtype_dimension": (
            grouped_by_basis_dtype_dimension
        ),
        "extreme_records": {
            "maximum_partition_function_relative_error": (
                row_for_maximum(
                    dataframe=dataframe,
                    metric=(
                        "partition_function_relative_error"
                    ),
                    columns=(
                        "basis",
                        "dtype",
                        "dimension",
                        "source_polynomial_degree",
                        "density_polynomial_degree",
                        "trial",
                        "query_scale",
                        "partition_function_relative_error",
                    ),
                )
            ),
            "maximum_query_probability_absolute_error": (
                row_for_maximum(
                    dataframe=dataframe,
                    metric=(
                        "query_probability_absolute_error"
                    ),
                    columns=(
                        "basis",
                        "dtype",
                        "dimension",
                        "source_polynomial_degree",
                        "density_polynomial_degree",
                        "trial",
                        "query_scale",
                        "query_probability_absolute_error",
                    ),
                )
            ),
            "maximum_query_probability_relative_error": (
                row_for_maximum(
                    dataframe=dataframe,
                    metric=(
                        "query_probability_relative_error"
                    ),
                    columns=(
                        "basis",
                        "dtype",
                        "dimension",
                        "source_polynomial_degree",
                        "density_polynomial_degree",
                        "trial",
                        "query_scale",
                        "query_probability_relative_error",
                    ),
                )
            ),
            "maximum_probability_perturbation_sensitivity": (
                row_for_maximum(
                    dataframe=dataframe,
                    metric=(
                        "query_probability_perturbation_sensitivity"
                    ),
                    columns=(
                        "basis",
                        "dtype",
                        "dimension",
                        "source_polynomial_degree",
                        "density_polynomial_degree",
                        "trial",
                        "query_scale",
                        "query_probability_perturbation_sensitivity",
                    ),
                )
            ),
        },
    }


def print_console_summary(dataframe: pd.DataFrame) -> None:
    print()
    print("Downstream benchmark summary")
    print("============================")
    print(f"records: {len(dataframe)}")
    print(
        "finite records: "
        f"{int(dataframe['finite_record'].astype(bool).sum())}"
    )
    print(
        "rounded-away perturbations: "
        f"{int(dataframe['perturbation_rounded_away'].astype(bool).sum())}"
    )
    print(
        "records with sampled negative density values: "
        f"{int((dataframe['negative_sampled_density_count'] > 0).sum())}"
    )

    summary_columns = [
        "basis",
        "dtype",
        "partition_function_relative_error",
        "query_probability_absolute_error",
        "query_probability_relative_error",
        "query_probability_perturbation_sensitivity",
        "total_downstream_evaluation_ms",
    ]

    grouped = (
        dataframe[summary_columns]
        .groupby(["basis", "dtype"], sort=True)
        .median(numeric_only=True)
        .reset_index()
    )

    print()
    print("Median metrics by basis and dtype")
    print(
        grouped.to_string(
            index=False,
            float_format=lambda value: f"{value:.6e}",
        )
    )

    print()
    print("Maximum query-probability error")

    maximum_index = (
        dataframe["query_probability_absolute_error"]
        .astype(float)
        .idxmax()
    )

    maximum_row = dataframe.loc[
        maximum_index,
        [
            "basis",
            "dtype",
            "dimension",
            "source_polynomial_degree",
            "density_polynomial_degree",
            "trial",
            "query_scale",
            "query_probability_absolute_error",
            "query_probability_relative_error",
        ],
    ]

    print(maximum_row.to_string())


# =============================================================================
# Output
# =============================================================================

def write_results(
    dataframe: pd.DataFrame,
    results_path: Path,
) -> None:
    ensure_parent_directory(results_path)

    dataframe.to_csv(
        results_path,
        index=False,
        float_format="%.17g",
    )

    print(f"Wrote {results_path}")


def write_summary(
    summary: Mapping[str, Any],
    summary_path: Path,
) -> None:
    ensure_parent_directory(summary_path)

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            to_builtin(summary),
            file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        file.write("\n")

    print(f"Wrote {summary_path}")


# =============================================================================
# Command-line interface
# =============================================================================

def parse_integer_sequence(
    values: Iterable[int],
    name: str,
    minimum: int,
) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)

    if len(parsed) == 0:
        raise ValueError(f"{name} must not be empty")

    if any(value < minimum for value in parsed):
        raise ValueError(
            f"Every {name} value must be at least {minimum}: {parsed}"
        )

    if len(set(parsed)) != len(parsed):
        raise ValueError(
            f"{name} contains duplicate values: {parsed}"
        )

    return parsed


def parse_query_scales(
    values: Iterable[float],
) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)

    if len(parsed) == 0:
        raise ValueError("query scales must not be empty")

    for scale in parsed:
        if not math.isfinite(scale):
            raise ValueError(
                f"Query scale must be finite: {scale}"
            )

        if scale <= 0.0 or scale > 1.0:
            raise ValueError(
                "Query scales must satisfy 0 < scale <= 1; "
                f"received {scale}"
            )

    if len(set(parsed)) != len(parsed):
        raise ValueError(
            f"query scales contain duplicates: {parsed}"
        )

    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark downstream partition functions and normalised "
            "probability queries across polynomial bases and arithmetic "
            "precisions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=list(DEFAULT_DIMENSIONS),
        help="Simplex dimensions to evaluate.",
    )

    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=list(DEFAULT_DEGREES),
        help=(
            "Degrees of the source polynomial q. The resulting density "
            "p=q^2+eta has degree twice this value."
        ),
    )

    parser.add_argument(
        "--bases",
        type=str,
        nargs="+",
        choices=SUPPORTED_BASES,
        default=list(SUPPORTED_BASES),
        help="Polynomial bases to evaluate.",
    )

    parser.add_argument(
        "--dtypes",
        type=str,
        nargs="+",
        choices=tuple(SUPPORTED_DTYPES),
        default=list(DEFAULT_DTYPES),
        help="Floating-point arithmetic types.",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of random densities per dimension-degree pair.",
    )

    parser.add_argument(
        "--query-scales",
        type=float,
        nargs="+",
        default=list(DEFAULT_QUERY_SCALES),
        help=(
            "Scaled-simplex query regions. Every scale must satisfy "
            "0 < scale <= 1."
        ),
    )

    parser.add_argument(
        "--density-offset",
        type=float,
        default=0.1,
        help=(
            "Positive eta in the non-negative density p(x)=q(x)^2+eta."
        ),
    )

    parser.add_argument(
        "--coefficient-scale",
        type=float,
        default=0.35,
        help="Standard deviation of source-polynomial coefficients.",
    )

    parser.add_argument(
        "--perturbation-magnitude",
        type=float,
        default=1e-5,
        help="Requested relative coefficient perturbation magnitude.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260803,
        help="Base random seed.",
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Output CSV path.",
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Output JSON summary path.",
    )

    parser.add_argument(
        "--fail-on-non-finite",
        action="store_true",
        help=(
            "Exit with an error if any output record contains a "
            "non-finite metric other than an infinite condition number."
        ),
    )

    parser.add_argument(
        "--fail-on-negative-density",
        action="store_true",
        help=(
            "Exit with an error if sampled validation points contain a "
            "negative density value below the validation tolerance."
        ),
    )

    return parser


def configuration_from_arguments(
    arguments: argparse.Namespace,
) -> BenchmarkConfiguration:
    dimensions = parse_integer_sequence(
        values=arguments.dimensions,
        name="dimensions",
        minimum=1,
    )

    degrees = parse_integer_sequence(
        values=arguments.degrees,
        name="degrees",
        minimum=0,
    )

    require_positive_integer(arguments.trials, "trials")
    require_finite_positive(
        arguments.density_offset,
        "density_offset",
    )
    require_finite_positive(
        arguments.coefficient_scale,
        "coefficient_scale",
    )
    require_finite_positive(
        arguments.perturbation_magnitude,
        "perturbation_magnitude",
    )

    query_scales = parse_query_scales(
        arguments.query_scales
    )

    bases = tuple(arguments.bases)
    dtypes = tuple(arguments.dtypes)

    if len(set(bases)) != len(bases):
        raise ValueError(f"bases contain duplicates: {bases}")

    if len(set(dtypes)) != len(dtypes):
        raise ValueError(f"dtypes contain duplicates: {dtypes}")

    return BenchmarkConfiguration(
        dimensions=dimensions,
        degrees=degrees,
        bases=bases,
        dtypes=dtypes,
        trials=int(arguments.trials),
        query_scales=query_scales,
        density_offset=float(arguments.density_offset),
        coefficient_scale=float(arguments.coefficient_scale),
        perturbation_magnitude=float(
            arguments.perturbation_magnitude
        ),
        seed=int(arguments.seed),
        results_path=str(arguments.results),
        summary_path=str(arguments.summary),
    )


def print_configuration(
    configuration: BenchmarkConfiguration,
) -> None:
    row_count = (
        len(configuration.dimensions)
        * len(configuration.degrees)
        * configuration.trials
        * len(configuration.bases)
        * len(configuration.dtypes)
        * len(configuration.query_scales)
    )

    print("Downstream benchmark configuration")
    print("==================================")
    print(f"dimensions: {list(configuration.dimensions)}")
    print(f"source degrees: {list(configuration.degrees)}")
    print(
        "density degrees: "
        f"{[2 * degree for degree in configuration.degrees]}"
    )
    print(f"bases: {list(configuration.bases)}")
    print(f"dtypes: {list(configuration.dtypes)}")
    print(f"trials: {configuration.trials}")
    print(f"query scales: {list(configuration.query_scales)}")
    print(f"density offset: {configuration.density_offset}")
    print(
        f"coefficient scale: {configuration.coefficient_scale}"
    )
    print(
        "perturbation magnitude: "
        f"{configuration.perturbation_magnitude}"
    )
    print(f"base seed: {configuration.seed}")
    print(f"expected records: {row_count}")
    print(f"results path: {configuration.results_path}")
    print(f"summary path: {configuration.summary_path}")
    print()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)

    try:
        configuration = configuration_from_arguments(
            arguments
        )
    except ValueError as error:
        parser.error(str(error))

    print_configuration(configuration)

    start = time.perf_counter()

    dataframe = run_benchmark(configuration)

    elapsed = time.perf_counter() - start

    print_console_summary(dataframe)

    summary = build_summary(
        dataframe=dataframe,
        configuration=configuration,
        elapsed_seconds=elapsed,
    )

    results_path = Path(configuration.results_path)
    summary_path = Path(configuration.summary_path)

    write_results(
        dataframe=dataframe,
        results_path=results_path,
    )

    write_summary(
        summary=summary,
        summary_path=summary_path,
    )

    non_finite_count = int(
        (~dataframe["finite_record"].astype(bool)).sum()
    )

    negative_density_count = int(
        (
            dataframe["negative_sampled_density_count"] > 0
        ).sum()
    )

    if arguments.fail_on_non_finite and non_finite_count > 0:
        print(
            "Error: benchmark produced "
            f"{non_finite_count} non-finite records.",
            file=sys.stderr,
        )
        return 2

    if (
        arguments.fail_on_negative_density
        and negative_density_count > 0
    ):
        print(
            "Error: benchmark produced "
            f"{negative_density_count} records with sampled negative "
            "density values.",
            file=sys.stderr,
        )
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())