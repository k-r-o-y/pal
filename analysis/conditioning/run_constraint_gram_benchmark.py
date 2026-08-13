#!/usr/bin/env python3
"""
Constraint-induced Gram-matrix conditioning benchmark.

This experiment constructs

    G_C[i, j] = ∫_C phi_i(x) phi_j(x) dx

for monomial, Legendre, and Chebyshev polynomial bases, and measures the
conditioning of G_C as polynomial degree and constraint geometry change.

The purpose is to complement the existing sampled basis-matrix experiment

    V[i, j] = phi_j(x_i)

with a matrix that is directly induced by constrained integration.

Implemented domains
-------------------
1. unit_simplex
       Delta_n = {x >= 0, sum_i x_i <= 1}

2. box_with_obstacle
       [0,1]^n minus a centred axis-aligned obstacle box.

All integrations are analytical after expanding the basis functions into
physical-coordinate monomials.

Examples
--------
Run from the repository root:

    python -m analysis.conditioning.run_constraint_gram_benchmark

or

    python -m analysis.conditioning.run_constraint_gram_benchmark \
        --dimensions 2 \
        --degrees 0 1 2 3 5 6 8 10 \
        --domains unit_simplex box_with_obstacle \
        --bases monomial legendre chebyshev

Outputs
-------
    results/conditioning/constraint_gram_results.csv
    results/conditioning/constraint_gram_summary.json

Notes
-----
The ambient physical domain for the orthogonal bases is [0,1]^n, so the
canonical coordinate is

    xi = 2*x - 1.

For the box-with-obstacle domain, the default obstacle is centred at 0.5 in
each coordinate with half-width 0.2, i.e. [0.3,0.7]^n.

The condition number is computed from singular values. Numerically singular
matrices are retained as condition_number = inf rather than being replaced
with an arbitrary finite plotting value.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial.legendre import Legendre


# =============================================================================
# Configuration
# =============================================================================

BASES = ("monomial", "legendre", "chebyshev")
DOMAINS = ("unit_simplex", "box_with_obstacle")

DEFAULT_DIMENSIONS = (2,)
DEFAULT_DEGREES = (0, 1, 2, 3, 5, 6, 8, 10)

DEFAULT_RESULTS_PATH = Path(
    "results/conditioning/constraint_gram_results.csv"
)

DEFAULT_SUMMARY_PATH = Path(
    "results/conditioning/constraint_gram_summary.json"
)

DEFAULT_OBSTACLE_HALF_WIDTH = 0.2

FLOAT64_EPS = np.finfo(np.float64).eps


# =============================================================================
# Structured result record
# =============================================================================

@dataclass
class Record:
    domain: str
    dimension: int
    degree: int
    basis: str

    basis_count: int

    condition_number: float
    log10_condition_number: float

    numerical_rank: int
    numerical_rank_fraction: float

    largest_singular_value: float
    smallest_singular_value: float

    symmetry_relative_error: float
    minimum_eigenvalue: float
    maximum_eigenvalue: float

    trace: float
    frobenius_norm: float

    obstacle_half_width: float

    construction_ms: float
    decomposition_ms: float

    finite_matrix: bool
    positive_semidefinite_within_tolerance: bool


# =============================================================================
# Multi-index utilities
# =============================================================================

def multi_indices(
    dimension: int,
    degree: int,
) -> list[tuple[int, ...]]:
    """
    Enumerate all multi-indices alpha with |alpha| <= degree.

    Ordering matches the convention used elsewhere in the project:
    first by total degree, then lexicographically.
    """
    if dimension < 1:
        raise ValueError("dimension must be positive")

    if degree < 0:
        raise ValueError("degree must be non-negative")

    return sorted(
        [
            alpha
            for alpha in product(
                range(degree + 1),
                repeat=dimension,
            )
            if sum(alpha) <= degree
        ],
        key=lambda alpha: (
            sum(alpha),
            alpha,
        ),
    )


def expected_basis_count(
    dimension: int,
    degree: int,
) -> int:
    """
    Number of total-degree basis functions:
        C(dimension + degree, degree)
    """
    return math.comb(
        dimension + degree,
        degree,
    )


# =============================================================================
# Sparse multivariate polynomial representation
# =============================================================================

SparsePolynomial = dict[tuple[int, ...], float]


def clean_sparse_polynomial(
    coefficients: Mapping[tuple[int, ...], float],
    *,
    tolerance: float = 0.0,
) -> SparsePolynomial:
    """
    Remove coefficients smaller than the requested tolerance.
    """
    return {
        alpha: float(value)
        for alpha, value in coefficients.items()
        if abs(float(value)) > tolerance
    }


def multiply_sparse_polynomials(
    first: Mapping[tuple[int, ...], float],
    second: Mapping[tuple[int, ...], float],
) -> SparsePolynomial:
    """
    Multiply two sparse multivariate polynomials represented in monomial form.
    """
    if not first or not second:
        return {}

    first_dimension = len(next(iter(first)))
    second_dimension = len(next(iter(second)))

    if first_dimension != second_dimension:
        raise ValueError(
            "polynomial dimensions do not match"
        )

    result: SparsePolynomial = {}

    for alpha, coefficient_a in first.items():
        for beta, coefficient_b in second.items():
            exponent = tuple(
                a + b
                for a, b in zip(alpha, beta)
            )

            result[exponent] = (
                result.get(exponent, 0.0)
                + float(coefficient_a)
                * float(coefficient_b)
            )

    return clean_sparse_polynomial(
        result,
        tolerance=0.0,
    )


# =============================================================================
# Univariate basis expansion in physical coordinates
# =============================================================================

def canonical_basis_polynomial(
    degree: int,
    basis: str,
) -> Polynomial:
    """
    Return the requested basis polynomial as a Polynomial in canonical xi.

    For example:
        monomial -> xi^degree
        legendre -> P_degree(xi)
        chebyshev -> T_degree(xi)

    The monomial branch is included for completeness, although physical
    monomials are handled directly elsewhere.
    """
    if degree < 0:
        raise ValueError(
            "degree must be non-negative"
        )

    if basis == "monomial":
        coefficients = np.zeros(
            degree + 1,
            dtype=np.float64,
        )
        coefficients[degree] = 1.0
        return Polynomial(coefficients)

    if basis == "legendre":
        basis_polynomial = Legendre.basis(degree)
        return basis_polynomial.convert(
            kind=Polynomial
        )

    if basis == "chebyshev":
        basis_polynomial = Chebyshev.basis(degree)
        return basis_polynomial.convert(
            kind=Polynomial
        )

    raise ValueError(
        f"unknown basis: {basis}"
    )


def compose_with_unit_interval_map(
    canonical_polynomial: Polynomial,
) -> Polynomial:
    """
    Compose a polynomial p(xi) with xi = 2*x - 1.

    The returned Polynomial is expressed directly in the physical coordinate x.
    """
    affine_map = Polynomial(
        [-1.0, 2.0]
    )

    result = Polynomial([0.0])
    power = Polynomial([1.0])

    for coefficient in canonical_polynomial.coef:
        result = (
            result
            + float(coefficient) * power
        )

        power = power * affine_map

    return result


def univariate_physical_basis_coefficients(
    degree: int,
    basis: str,
) -> np.ndarray:
    """
    Return coefficients of one univariate basis function in powers of x.

    The ambient physical coordinate is x in [0,1].

    For Legendre and Chebyshev:
        basis_k(x) = P_k(2x-1)
        basis_k(x) = T_k(2x-1)

    respectively.
    """
    if basis == "monomial":
        coefficients = np.zeros(
            degree + 1,
            dtype=np.float64,
        )
        coefficients[degree] = 1.0
        return coefficients

    canonical = canonical_basis_polynomial(
        degree,
        basis,
    )

    physical = compose_with_unit_interval_map(
        canonical
    )

    coefficients = np.asarray(
        physical.coef,
        dtype=np.float64,
    )

    return coefficients


# =============================================================================
# Multivariate basis expansion
# =============================================================================

def basis_function_as_monomials(
    alpha: Sequence[int],
    basis: str,
) -> SparsePolynomial:
    """
    Expand one multivariate basis function into physical-coordinate monomials.

    Example:
        Legendre alpha=(2,1)

    corresponds to
        P_2(2*x_1-1) * P_1(2*x_2-1),

    expanded into powers of x_1 and x_2.
    """
    alpha = tuple(
        int(value)
        for value in alpha
    )

    dimension = len(alpha)

    if basis == "monomial":
        return {
            alpha: 1.0
        }

    univariate_tables: list[np.ndarray] = []

    for degree in alpha:
        univariate_tables.append(
            univariate_physical_basis_coefficients(
                degree,
                basis,
            )
        )

    result: SparsePolynomial = {}

    exponent_ranges = [
        range(len(coefficients))
        for coefficients in univariate_tables
    ]

    for exponent_tuple in product(
        *exponent_ranges
    ):
        coefficient = 1.0

        for axis, exponent in enumerate(
            exponent_tuple
        ):
            coefficient *= float(
                univariate_tables[axis][exponent]
            )

        if coefficient != 0.0:
            result[
                tuple(exponent_tuple)
            ] = (
                result.get(
                    tuple(exponent_tuple),
                    0.0,
                )
                + coefficient
            )

    return clean_sparse_polynomial(
        result
    )


# =============================================================================
# Exact polynomial moments
# =============================================================================

def unit_simplex_monomial_integral(
    exponents: Sequence[int],
) -> float:
    """
    Analytically integrate x^alpha over the unit simplex.

        integral_Delta x^alpha dx
          = prod Gamma(alpha_i + 1)
            --------------------------------
            Gamma(n + |alpha| + 1)

    Log-Gamma evaluation avoids explicitly constructing large factorials.
    """
    exponents = tuple(
        int(alpha)
        for alpha in exponents
    )

    if any(
        alpha < 0
        for alpha in exponents
    ):
        raise ValueError(
            "exponents must be non-negative"
        )

    dimension = len(exponents)

    log_numerator = sum(
        math.lgamma(
            alpha + 1.0
        )
        for alpha in exponents
    )

    log_denominator = math.lgamma(
        dimension
        + sum(exponents)
        + 1.0
    )

    return math.exp(
        log_numerator
        - log_denominator
    )


def axis_aligned_box_monomial_integral(
    exponents: Sequence[int],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
) -> float:
    """
    Analytically integrate x^alpha over an axis-aligned box.
    """
    exponents = tuple(
        int(alpha)
        for alpha in exponents
    )

    lower_bounds = tuple(
        float(value)
        for value in lower_bounds
    )

    upper_bounds = tuple(
        float(value)
        for value in upper_bounds
    )

    if not (
        len(exponents)
        == len(lower_bounds)
        == len(upper_bounds)
    ):
        raise ValueError(
            "dimension mismatch in box integral"
        )

    result = 1.0

    for alpha, lower, upper in zip(
        exponents,
        lower_bounds,
        upper_bounds,
    ):
        if lower > upper:
            raise ValueError(
                "lower box bound exceeds upper bound"
            )

        power = alpha + 1

        result *= (
            upper ** power
            - lower ** power
        ) / power

    return float(result)


def integrate_sparse_over_unit_simplex(
    polynomial: Mapping[
        tuple[int, ...],
        float,
    ],
) -> float:
    """
    Integrate a sparse monomial polynomial over the unit simplex.
    """
    total = 0.0

    for alpha, coefficient in polynomial.items():
        total += (
            float(coefficient)
            * unit_simplex_monomial_integral(
                alpha
            )
        )

    return float(total)


def integrate_sparse_over_box(
    polynomial: Mapping[
        tuple[int, ...],
        float,
    ],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
) -> float:
    """
    Integrate a sparse monomial polynomial over an axis-aligned box.
    """
    total = 0.0

    for alpha, coefficient in polynomial.items():
        total += (
            float(coefficient)
            * axis_aligned_box_monomial_integral(
                alpha,
                lower_bounds,
                upper_bounds,
            )
        )

    return float(total)


def integrate_sparse_over_box_with_obstacle(
    polynomial: Mapping[
        tuple[int, ...],
        float,
    ],
    dimension: int,
    obstacle_half_width: float,
) -> float:
    """
    Integrate over

        [0,1]^n minus centred obstacle box.

    The obstacle is centred at 0.5 on every axis.
    """
    if not (
        0.0
        <= obstacle_half_width
        < 0.5
    ):
        raise ValueError(
            "obstacle_half_width must satisfy "
            "0 <= width < 0.5"
        )

    outer_lower = np.zeros(
        dimension,
        dtype=np.float64,
    )

    outer_upper = np.ones(
        dimension,
        dtype=np.float64,
    )

    centre = 0.5

    obstacle_lower = np.full(
        dimension,
        centre - obstacle_half_width,
        dtype=np.float64,
    )

    obstacle_upper = np.full(
        dimension,
        centre + obstacle_half_width,
        dtype=np.float64,
    )

    outer_integral = integrate_sparse_over_box(
        polynomial,
        outer_lower,
        outer_upper,
    )

    obstacle_integral = integrate_sparse_over_box(
        polynomial,
        obstacle_lower,
        obstacle_upper,
    )

    return float(
        outer_integral
        - obstacle_integral
    )


def integrate_sparse_polynomial(
    polynomial: Mapping[
        tuple[int, ...],
        float,
    ],
    *,
    domain: str,
    dimension: int,
    obstacle_half_width: float,
) -> float:
    """
    Dispatch exact polynomial integration to the selected domain.
    """
    if domain == "unit_simplex":
        return integrate_sparse_over_unit_simplex(
            polynomial
        )

    if domain == "box_with_obstacle":
        return integrate_sparse_over_box_with_obstacle(
            polynomial,
            dimension,
            obstacle_half_width,
        )

    raise ValueError(
        f"unknown domain: {domain}"
    )


# =============================================================================
# Gram matrix construction
# =============================================================================

def precompute_basis_expansions(
    indices: Sequence[tuple[int, ...]],
    basis: str,
) -> list[SparsePolynomial]:
    """
    Expand every selected basis function into physical monomials once.
    """
    return [
        basis_function_as_monomials(
            alpha,
            basis,
        )
        for alpha in indices
    ]


def constraint_gram_matrix(
    *,
    dimension: int,
    degree: int,
    basis: str,
    domain: str,
    obstacle_half_width: float,
) -> np.ndarray:
    """
    Construct the constraint-induced Gram matrix

        G_C[i,j] = integral_C phi_i(x) phi_j(x) dx.

    Symmetry is exploited: only the upper triangle is explicitly integrated.
    """
    indices = multi_indices(
        dimension,
        degree,
    )

    expected = expected_basis_count(
        dimension,
        degree,
    )

    if len(indices) != expected:
        raise RuntimeError(
            "internal basis-count mismatch: "
            f"expected {expected}, got {len(indices)}"
        )

    expansions = precompute_basis_expansions(
        indices,
        basis,
    )

    count = len(indices)

    matrix = np.empty(
        (count, count),
        dtype=np.float64,
    )

    for row in range(count):
        first = expansions[row]

        for column in range(
            row,
            count,
        ):
            second = expansions[column]

            product_polynomial = (
                multiply_sparse_polynomials(
                    first,
                    second,
                )
            )

            value = integrate_sparse_polynomial(
                product_polynomial,
                domain=domain,
                dimension=dimension,
                obstacle_half_width=(
                    obstacle_half_width
                ),
            )

            matrix[
                row,
                column,
            ] = value

            matrix[
                column,
                row,
            ] = value

    return matrix


# =============================================================================
# Matrix diagnostics
# =============================================================================

def symmetry_relative_error(
    matrix: np.ndarray,
) -> float:
    """
    ||G - G^T||_F / max(||G||_F, eps)
    """
    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    numerator = np.linalg.norm(
        matrix - matrix.T,
        ord="fro",
    )

    denominator = max(
        np.linalg.norm(
            matrix,
            ord="fro",
        ),
        FLOAT64_EPS,
    )

    return float(
        numerator / denominator
    )


def singular_value_diagnostics(
    matrix: np.ndarray,
) -> tuple[
    np.ndarray,
    int,
    float,
]:
    """
    Return singular values, numerical rank, and condition number.

    NumPy's conventional matrix-rank tolerance is used:

        tol = sigma_max * max(m,n) * eps
    """
    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    singular_values = np.linalg.svd(
        matrix,
        compute_uv=False,
    )

    if singular_values.size == 0:
        return (
            singular_values,
            0,
            math.inf,
        )

    largest = float(
        singular_values[0]
    )

    smallest = float(
        singular_values[-1]
    )

    tolerance = (
        largest
        * max(matrix.shape)
        * np.finfo(
            matrix.dtype
        ).eps
    )

    numerical_rank = int(
        np.sum(
            singular_values
            > tolerance
        )
    )

    if (
        smallest <= tolerance
        or smallest == 0.0
        or not math.isfinite(smallest)
    ):
        condition_number = math.inf
    else:
        condition_number = float(
            largest / smallest
        )

    return (
        singular_values,
        numerical_rank,
        condition_number,
    )


def eigenvalue_diagnostics(
    matrix: np.ndarray,
) -> tuple[
    float,
    float,
    bool,
]:
    """
    Return minimum eigenvalue, maximum eigenvalue, and whether the matrix is
    positive semidefinite within a scale-aware floating-point tolerance.
    """
    symmetric = 0.5 * (
        matrix + matrix.T
    )

    eigenvalues = np.linalg.eigvalsh(
        symmetric
    )

    minimum = float(
        eigenvalues[0]
    )

    maximum = float(
        eigenvalues[-1]
    )

    tolerance = (
        100.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            abs(maximum),
            np.linalg.norm(
                symmetric,
                ord=2,
            ),
        )
    )

    psd = bool(
        minimum >= -tolerance
    )

    return (
        minimum,
        maximum,
        psd,
    )


def safe_log10_condition_number(
    condition_number: float,
) -> float:
    """
    Preserve infinite condition numbers as +inf.
    """
    if math.isinf(
        condition_number
    ):
        return math.inf

    if condition_number <= 0.0:
        return math.nan

    return float(
        math.log10(
            condition_number
        )
    )


# =============================================================================
# Validation
# =============================================================================

def validate_known_interval_case() -> None:
    """
    Validate the classic degree-1 monomial Gram matrix on [0,1]:

        [[1,   1/2],
         [1/2, 1/3]]

    This is equivalent to the one-dimensional unit simplex.
    """
    matrix = constraint_gram_matrix(
        dimension=1,
        degree=1,
        basis="monomial",
        domain="unit_simplex",
        obstacle_half_width=(
            DEFAULT_OBSTACLE_HALF_WIDTH
        ),
    )

    expected = np.asarray(
        [
            [1.0, 0.5],
            [0.5, 1.0 / 3.0],
        ],
        dtype=np.float64,
    )

    if not np.allclose(
        matrix,
        expected,
        rtol=1e-13,
        atol=1e-15,
    ):
        raise RuntimeError(
            "known 1D monomial Gram-matrix "
            "validation failed.\n"
            f"computed:\n{matrix}\n"
            f"expected:\n{expected}"
        )


def validate_constant_basis_case(
    *,
    dimension: int,
    domain: str,
    obstacle_half_width: float,
) -> None:
    """
    At degree zero, every basis consists only of the constant function 1.

    Therefore G_C is the 1x1 matrix containing the volume of C and must agree
    across all three bases.
    """
    matrices = {}

    for basis in BASES:
        matrices[basis] = (
            constraint_gram_matrix(
                dimension=dimension,
                degree=0,
                basis=basis,
                domain=domain,
                obstacle_half_width=(
                    obstacle_half_width
                ),
            )
        )

    reference = matrices["monomial"]

    for basis in (
        "legendre",
        "chebyshev",
    ):
        if not np.allclose(
            matrices[basis],
            reference,
            rtol=1e-13,
            atol=1e-15,
        ):
            raise RuntimeError(
                "constant-basis validation failed "
                f"for {domain}, dimension "
                f"{dimension}, basis {basis}"
            )


def run_validation_suite(
    *,
    dimensions: Iterable[int],
    domains: Iterable[str],
    obstacle_half_width: float,
) -> None:
    """
    Run inexpensive analytical checks before the benchmark.
    """
    validate_known_interval_case()

    for dimension in dimensions:
        for domain in domains:
            validate_constant_basis_case(
                dimension=dimension,
                domain=domain,
                obstacle_half_width=(
                    obstacle_half_width
                ),
            )


# =============================================================================
# Benchmark execution
# =============================================================================

def evaluate_configuration(
    *,
    domain: str,
    dimension: int,
    degree: int,
    basis: str,
    obstacle_half_width: float,
) -> Record:
    """
    Construct G_C and compute all requested diagnostics for one configuration.
    """
    start = time.perf_counter()

    matrix = constraint_gram_matrix(
        dimension=dimension,
        degree=degree,
        basis=basis,
        domain=domain,
        obstacle_half_width=(
            obstacle_half_width
        ),
    )

    after_construction = (
        time.perf_counter()
    )

    finite_matrix = bool(
        np.all(
            np.isfinite(matrix)
        )
    )

    if not finite_matrix:
        raise FloatingPointError(
            "non-finite Gram matrix encountered for "
            f"domain={domain}, "
            f"dimension={dimension}, "
            f"degree={degree}, "
            f"basis={basis}"
        )

    symmetry_error = (
        symmetry_relative_error(
            matrix
        )
    )

    (
        singular_values,
        numerical_rank,
        condition_number,
    ) = singular_value_diagnostics(
        matrix
    )

    (
        minimum_eigenvalue,
        maximum_eigenvalue,
        psd,
    ) = eigenvalue_diagnostics(
        matrix
    )

    after_decomposition = (
        time.perf_counter()
    )

    basis_count = matrix.shape[0]

    largest_singular_value = (
        float(
            singular_values[0]
        )
        if singular_values.size
        else math.nan
    )

    smallest_singular_value = (
        float(
            singular_values[-1]
        )
        if singular_values.size
        else math.nan
    )

    return Record(
        domain=domain,
        dimension=dimension,
        degree=degree,
        basis=basis,

        basis_count=basis_count,

        condition_number=(
            condition_number
        ),
        log10_condition_number=(
            safe_log10_condition_number(
                condition_number
            )
        ),

        numerical_rank=(
            numerical_rank
        ),
        numerical_rank_fraction=(
            numerical_rank / basis_count
            if basis_count
            else math.nan
        ),

        largest_singular_value=(
            largest_singular_value
        ),
        smallest_singular_value=(
            smallest_singular_value
        ),

        symmetry_relative_error=(
            symmetry_error
        ),
        minimum_eigenvalue=(
            minimum_eigenvalue
        ),
        maximum_eigenvalue=(
            maximum_eigenvalue
        ),

        trace=float(
            np.trace(matrix)
        ),
        frobenius_norm=float(
            np.linalg.norm(
                matrix,
                ord="fro",
            )
        ),

        obstacle_half_width=(
            obstacle_half_width
            if domain
            == "box_with_obstacle"
            else 0.0
        ),

        construction_ms=(
            (
                after_construction
                - start
            )
            * 1000.0
        ),
        decomposition_ms=(
            (
                after_decomposition
                - after_construction
            )
            * 1000.0
        ),

        finite_matrix=(
            finite_matrix
        ),
        positive_semidefinite_within_tolerance=(
            psd
        ),
    )


# =============================================================================
# Output
# =============================================================================

def ensure_parent_directory(
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def write_csv(
    records: Sequence[Record],
    path: Path,
) -> None:
    ensure_parent_directory(
        path
    )

    if not records:
        raise ValueError(
            "cannot write empty results"
        )

    fieldnames = list(
        asdict(records[0]).keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                asdict(record)
            )


def json_safe_number(
    value: float,
):
    """
    JSON does not have a portable representation for NaN/Infinity.

    Convert them to strings while preserving ordinary finite numbers.
    """
    value = float(value)

    if math.isnan(value):
        return "nan"

    if math.isinf(value):
        if value > 0:
            return "inf"
        return "-inf"

    return value


def build_summary(
    *,
    records: Sequence[Record],
    dimensions: Sequence[int],
    degrees: Sequence[int],
    bases: Sequence[str],
    domains: Sequence[str],
    obstacle_half_width: float,
) -> dict:
    """
    Construct a compact JSON summary of the completed run.
    """
    groups = []

    for domain in domains:
        for dimension in dimensions:
            selected = [
                record
                for record in records
                if (
                    record.domain
                    == domain
                    and record.dimension
                    == dimension
                )
            ]

            if not selected:
                continue

            finite_conditions = [
                record.condition_number
                for record in selected
                if math.isfinite(
                    record.condition_number
                )
            ]

            infinite_conditions = sum(
                1
                for record in selected
                if math.isinf(
                    record.condition_number
                )
            )

            minimum_rank_fraction = min(
                record.numerical_rank_fraction
                for record in selected
            )

            maximum_symmetry_error = max(
                record.symmetry_relative_error
                for record in selected
            )

            minimum_eigenvalue = min(
                record.minimum_eigenvalue
                for record in selected
            )

            groups.append(
                {
                    "domain": domain,
                    "dimension": dimension,
                    "records": len(
                        selected
                    ),
                    "finite_condition_numbers": len(
                        finite_conditions
                    ),
                    "infinite_condition_numbers": (
                        infinite_conditions
                    ),
                    "maximum_finite_condition_number": (
                        json_safe_number(
                            max(
                                finite_conditions
                            )
                        )
                        if finite_conditions
                        else None
                    ),
                    "minimum_rank_fraction": (
                        json_safe_number(
                            minimum_rank_fraction
                        )
                    ),
                    "maximum_symmetry_relative_error": (
                        json_safe_number(
                            maximum_symmetry_error
                        )
                    ),
                    "minimum_eigenvalue": (
                        json_safe_number(
                            minimum_eigenvalue
                        )
                    ),
                    "all_psd_within_tolerance": all(
                        record.positive_semidefinite_within_tolerance
                        for record in selected
                    ),
                }
            )

    return {
        "experiment": (
            "constraint_induced_gram_conditioning"
        ),
        "definition": (
            "G_C[i,j] = integral_C "
            "phi_i(x) phi_j(x) dx"
        ),
        "condition_metric": (
            "spectral condition number "
            "sigma_max / sigma_min"
        ),
        "dimensions": list(
            dimensions
        ),
        "degrees": list(
            degrees
        ),
        "bases": list(
            bases
        ),
        "domains": list(
            domains
        ),
        "ambient_domain": (
            "[0,1]^dimension"
        ),
        "orthogonal_coordinate_map": (
            "xi = 2*x - 1"
        ),
        "obstacle_half_width": (
            obstacle_half_width
        ),
        "record_count": len(
            records
        ),
        "groups": groups,
    }


def write_json(
    summary: dict,
    path: Path,
) -> None:
    ensure_parent_directory(
        path
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            allow_nan=False,
        )

        handle.write("\n")


# =============================================================================
# Console output
# =============================================================================

def format_condition(
    value: float,
) -> str:
    if math.isinf(value):
        return "inf"

    if math.isnan(value):
        return "nan"

    return f"{value:.6e}"


def print_record(
    record: Record,
) -> None:
    print(
        f"domain={record.domain:17s} "
        f"n={record.dimension:d} "
        f"degree={record.degree:2d} "
        f"basis={record.basis:9s} "
        f"M={record.basis_count:4d} "
        f"kappa={format_condition(record.condition_number):>13s} "
        f"rank={record.numerical_rank:4d}/"
        f"{record.basis_count:<4d} "
        f"rank_frac={record.numerical_rank_fraction:.3f} "
        f"lambda_min={record.minimum_eigenvalue:.3e} "
        f"build={record.construction_ms:.2f} ms"
    )


# =============================================================================
# Command-line interface
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure conditioning of the "
            "constraint-induced Gram matrix G_C "
            "for monomial, Legendre, and "
            "Chebyshev polynomial bases."
        )
    )

    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=list(
            DEFAULT_DIMENSIONS
        ),
        help=(
            "Dimensions to evaluate. "
            "Default: 2"
        ),
    )

    parser.add_argument(
        "--degrees",
        nargs="+",
        type=int,
        default=list(
            DEFAULT_DEGREES
        ),
        help=(
            "Maximum total polynomial degrees."
        ),
    )

    parser.add_argument(
        "--bases",
        nargs="+",
        choices=BASES,
        default=list(
            BASES
        ),
        help=(
            "Polynomial bases to evaluate."
        ),
    )

    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(
            DOMAINS
        ),
        help=(
            "Constraint geometries to evaluate."
        ),
    )

    parser.add_argument(
        "--obstacle-half-width",
        type=float,
        default=(
            DEFAULT_OBSTACLE_HALF_WIDTH
        ),
        help=(
            "Half-width of the centred obstacle "
            "used by box_with_obstacle. "
            "Default: 0.2"
        ),
    )

    parser.add_argument(
        "--results-path",
        type=Path,
        default=(
            DEFAULT_RESULTS_PATH
        ),
    )

    parser.add_argument(
        "--summary-path",
        type=Path,
        default=(
            DEFAULT_SUMMARY_PATH
        ),
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help=(
            "Skip inexpensive analytical "
            "validation checks."
        ),
    )

    return parser


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    if any(
        dimension < 1
        for dimension in args.dimensions
    ):
        raise ValueError(
            "all dimensions must be positive"
        )

    if any(
        degree < 0
        for degree in args.degrees
    ):
        raise ValueError(
            "all degrees must be non-negative"
        )

    if not (
        0.0
        <= args.obstacle_half_width
        < 0.5
    ):
        raise ValueError(
            "--obstacle-half-width must "
            "satisfy 0 <= width < 0.5"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = build_parser()

    args = parser.parse_args()

    validate_arguments(
        args
    )

    dimensions = sorted(
        set(args.dimensions)
    )

    degrees = sorted(
        set(args.degrees)
    )

    bases = list(
        dict.fromkeys(
            args.bases
        )
    )

    domains = list(
        dict.fromkeys(
            args.domains
        )
    )

    print(
        "Constraint-induced Gram conditioning benchmark"
    )
    print(
        "=============================================="
    )

    print(
        f"dimensions: {dimensions}"
    )

    print(
        f"degrees: {degrees}"
    )

    print(
        f"bases: {bases}"
    )

    print(
        f"domains: {domains}"
    )

    print(
        "ambient domain: [0,1]^n"
    )

    print(
        "orthogonal coordinate map: xi = 2*x - 1"
    )

    print(
        "obstacle half-width: "
        f"{args.obstacle_half_width}"
    )

    expected_records = (
        len(dimensions)
        * len(degrees)
        * len(bases)
        * len(domains)
    )

    print(
        f"expected records: {expected_records}"
    )

    print()

    if not args.skip_validation:
        print(
            "Running validation checks..."
        )

        run_validation_suite(
            dimensions=dimensions,
            domains=domains,
            obstacle_half_width=(
                args.obstacle_half_width
            ),
        )

        print(
            "Validation passed."
        )

        print()

    records: list[Record] = []

    total_start = time.perf_counter()

    for domain in domains:
        for dimension in dimensions:
            for degree in degrees:
                for basis in bases:
                    record = (
                        evaluate_configuration(
                            domain=domain,
                            dimension=dimension,
                            degree=degree,
                            basis=basis,
                            obstacle_half_width=(
                                args.obstacle_half_width
                            ),
                        )
                    )

                    records.append(
                        record
                    )

                    print_record(
                        record
                    )

    elapsed = (
        time.perf_counter()
        - total_start
    )

    print()

    if len(records) != expected_records:
        raise RuntimeError(
            "unexpected record count: "
            f"expected {expected_records}, "
            f"got {len(records)}"
        )

    write_csv(
        records,
        args.results_path,
    )

    summary = build_summary(
        records=records,
        dimensions=dimensions,
        degrees=degrees,
        bases=bases,
        domains=domains,
        obstacle_half_width=(
            args.obstacle_half_width
        ),
    )

    write_json(
        summary,
        args.summary_path,
    )

    print(
        f"Wrote {args.results_path}"
    )

    print(
        f"Wrote {args.summary_path}"
    )

    print(
        f"Completed {len(records)} records "
        f"in {elapsed:.3f} seconds."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )