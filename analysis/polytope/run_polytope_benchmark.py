"""
Static and dynamic constrained-polytope benchmark.

This benchmark compares monomial, Legendre, and Chebyshev polynomial
representations over convex boxes and boxes containing axis-aligned obstacles.

The benchmark records:

1. Relative integration error.
2. Coefficient-perturbation sensitivity.
3. Basis evaluation-matrix condition number.
4. Retained numerical-rank fraction.
5. Function-value noise amplification during coefficient recovery.
6. Relative coefficient-recovery error.
7. Relative integral error after noisy coefficient recovery.
8. End-to-end evaluation runtime.

The mathematical density is constructed as

    p(x) = q(x)^2 + eta,

which guarantees non-negativity.

The same mathematical density is represented in all three bases. Analytical
integration over axis-aligned boxes is used as the common reference. A
box-with-obstacle integral is computed as the outer-box integral minus the
obstacle-box integral.

Run with:

python -m analysis.polytope.run_polytope_benchmark \
    --dimensions 2 3 \
    --degrees 0 1 2 3 5 \
    --trials 10 \
    --dtypes float32 float64 \
    --schedules shrinking expanding oscillating pulsed random_walk \
    --trajectory-steps 12 \
    --perturbation-magnitude 1e-5 \
    --value-noise-magnitude 1e-5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


BASIS_NAMES = ("monomial", "legendre", "chebyshev")
SUPPORTED_DTYPES = {
    "float32": np.float32,
    "float64": np.float64,
}
SUPPORTED_SCHEDULES = (
    "shrinking",
    "expanding",
    "oscillating",
    "pulsed",
    "random_walk",
)

DEFAULT_RESULTS_PATH = Path("results/polytope/polytope_results.csv")
DEFAULT_SUMMARY_PATH = Path("results/polytope/polytope_summary.json")

EPSILON_FLOAT64 = np.finfo(np.float64).eps


@dataclass(frozen=True)
class AxisAlignedBox:
    """Closed axis-aligned box [lower, upper]."""

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)

        if lower.ndim != 1 or upper.ndim != 1:
            raise ValueError("Box bounds must be one-dimensional arrays.")
        if lower.shape != upper.shape:
            raise ValueError("Lower and upper bounds must have equal shapes.")
        if np.any(upper <= lower):
            raise ValueError("Every upper bound must exceed its lower bound.")

        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def dimension(self) -> int:
        return int(self.lower.size)

    @property
    def volume(self) -> float:
        return float(np.prod(self.upper - self.lower))


@dataclass
class BenchmarkRecord:
    scenario: str
    schedule: str
    trajectory_step: int
    trajectory_steps: int

    dimension: int
    source_polynomial_degree: int
    density_polynomial_degree: int
    trial: int
    basis: str
    dtype: str

    coefficient_seed: int
    perturbation_seed: int
    point_seed: int
    value_noise_seed: int

    density_offset: float
    coefficient_scale: float
    perturbation_magnitude: float
    value_noise_magnitude: float

    obstacle_half_width: float

    basis_count: int
    recovery_point_count: int

    reference_integral: float
    computed_integral: float
    perturbed_integral: float
    recovered_integral: float

    relative_integration_error: float
    perturbation_sensitivity: float

    basis_condition_number: float
    numerical_rank: int
    numerical_rank_fraction: float

    function_value_noise_relative_norm: float
    coefficient_recovery_relative_error: float
    coefficient_noise_amplification: float
    recovered_integral_relative_error: float

    perturbation_rounded_away: bool
    sampled_negative_density: bool

    integration_ms: float
    conditioning_ms: float
    recovery_ms: float
    total_evaluation_ms: float


# =============================================================================
# Multi-index utilities
# =============================================================================


def total_degree_indices(dimension: int, degree: int) -> list[tuple[int, ...]]:
    """Enumerate all non-negative multi-indices with total degree <= degree."""

    if dimension < 1:
        raise ValueError("dimension must be positive")
    if degree < 0:
        raise ValueError("degree must be non-negative")

    indices: list[tuple[int, ...]] = []

    def recurse(prefix: tuple[int, ...], remaining_dim: int, remaining: int) -> None:
        if remaining_dim == 1:
            for value in range(remaining + 1):
                indices.append(prefix + (value,))
            return

        for value in range(remaining + 1):
            recurse(prefix + (value,), remaining_dim - 1, remaining - value)

    recurse((), dimension, degree)
    indices.sort(key=lambda alpha: (sum(alpha), alpha))
    return indices


def add_multi_indices(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


# =============================================================================
# Polynomial construction
# =============================================================================


def generate_source_polynomial(
    dimension: int,
    degree: int,
    coefficient_scale: float,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Generate coefficients of q(x) in the monomial basis."""

    indices = total_degree_indices(dimension, degree)
    coefficients = rng.normal(
        loc=0.0,
        scale=coefficient_scale,
        size=len(indices),
    ).astype(np.float64)

    # Avoid a completely tiny source polynomial.
    if np.linalg.norm(coefficients) < 1e-12:
        coefficients[0] = coefficient_scale

    return indices, coefficients


def square_monomial_polynomial(
    source_indices: Sequence[tuple[int, ...]],
    source_coefficients: np.ndarray,
    density_offset: float,
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """
    Construct p(x) = q(x)^2 + density_offset in the monomial basis.
    """

    if density_offset <= 0.0:
        raise ValueError("density_offset must be positive")

    accumulator: dict[tuple[int, ...], float] = {}

    for i, alpha in enumerate(source_indices):
        for j, beta in enumerate(source_indices):
            gamma = add_multi_indices(alpha, beta)
            accumulator[gamma] = (
                accumulator.get(gamma, 0.0)
                + float(source_coefficients[i] * source_coefficients[j])
            )

    zero_index = tuple(0 for _ in source_indices[0])
    accumulator[zero_index] = accumulator.get(zero_index, 0.0) + density_offset

    density_indices = sorted(
        accumulator,
        key=lambda alpha: (sum(alpha), alpha),
    )
    density_coefficients = np.asarray(
        [accumulator[alpha] for alpha in density_indices],
        dtype=np.float64,
    )

    return density_indices, density_coefficients


# =============================================================================
# One-dimensional basis conversion
# =============================================================================


def affine_canonical_polynomial(lower: float, upper: float) -> np.polynomial.Polynomial:
    """
    Return the physical-coordinate polynomial for

        t(x) = 2(x - lower)/(upper - lower) - 1.
    """

    width = upper - lower
    if width <= 0.0:
        raise ValueError("upper must exceed lower")

    return np.polynomial.Polynomial(
        [
            -(upper + lower) / width,
            2.0 / width,
        ]
    )


def canonical_basis_polynomial_in_physical_coordinate(
    basis: str,
    degree: int,
    lower: float,
    upper: float,
) -> np.polynomial.Polynomial:
    """
    Expand one Legendre/Chebyshev basis function into physical monomials.
    """

    canonical_map = affine_canonical_polynomial(lower, upper)

    if basis == "monomial":
        coefficients = np.zeros(degree + 1, dtype=np.float64)
        coefficients[degree] = 1.0
        return np.polynomial.Polynomial(coefficients)

    if basis == "legendre":
        basis_coefficients = np.zeros(degree + 1, dtype=np.float64)
        basis_coefficients[degree] = 1.0
        canonical_polynomial = np.polynomial.Legendre(
            basis_coefficients
        ).convert(kind=np.polynomial.Polynomial)
        return canonical_polynomial(canonical_map)

    if basis == "chebyshev":
        basis_coefficients = np.zeros(degree + 1, dtype=np.float64)
        basis_coefficients[degree] = 1.0
        canonical_polynomial = np.polynomial.Chebyshev(
            basis_coefficients
        ).convert(kind=np.polynomial.Polynomial)
        return canonical_polynomial(canonical_map)

    raise ValueError(f"Unsupported basis: {basis}")


def build_univariate_basis_to_monomial_matrix(
    basis: str,
    maximum_degree: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    """
    Matrix C satisfying

        monomial_coefficients = C @ basis_coefficients.
    """

    matrix = np.zeros(
        (maximum_degree + 1, maximum_degree + 1),
        dtype=np.float64,
    )

    for basis_degree in range(maximum_degree + 1):
        polynomial = canonical_basis_polynomial_in_physical_coordinate(
            basis=basis,
            degree=basis_degree,
            lower=lower,
            upper=upper,
        )
        coefficients = np.asarray(polynomial.coef, dtype=np.float64)
        matrix[: coefficients.size, basis_degree] = coefficients

    return matrix


# =============================================================================
# Multivariate basis conversion
# =============================================================================


def convert_monomial_to_basis(
    monomial_indices: Sequence[tuple[int, ...]],
    monomial_coefficients: np.ndarray,
    target_indices: Sequence[tuple[int, ...]],
    basis: str,
    domain: AxisAlignedBox,
) -> np.ndarray:
    """
    Convert a multivariate physical-monomial polynomial to a product basis.

    The target basis uses canonical Legendre/Chebyshev factors on each physical
    coordinate interval.
    """

    if basis == "monomial":
        coefficient_map = {
            alpha: float(value)
            for alpha, value in zip(
                monomial_indices,
                monomial_coefficients,
                strict=True,
            )
        }
        return np.asarray(
            [coefficient_map.get(alpha, 0.0) for alpha in target_indices],
            dtype=np.float64,
        )

    dimension = domain.dimension
    maximum_degree = max(
        max(alpha) for alpha in target_indices
    )

    univariate_matrices = [
        build_univariate_basis_to_monomial_matrix(
            basis=basis,
            maximum_degree=maximum_degree,
            lower=float(domain.lower[axis]),
            upper=float(domain.upper[axis]),
        )
        for axis in range(dimension)
    ]

    row_lookup = {
        alpha: row
        for row, alpha in enumerate(monomial_indices)
    }

    transformation = np.zeros(
        (len(monomial_indices), len(target_indices)),
        dtype=np.float64,
    )

    for column, beta in enumerate(target_indices):
        per_axis_terms: list[list[tuple[int, float]]] = []

        for axis, basis_degree in enumerate(beta):
            column_values = univariate_matrices[axis][:, basis_degree]
            terms = [
                (power, float(value))
                for power, value in enumerate(column_values)
                if value != 0.0
            ]
            per_axis_terms.append(terms)

        for term_combination in product(*per_axis_terms):
            monomial_alpha = tuple(
                term[0] for term in term_combination
            )
            row = row_lookup.get(monomial_alpha)
            if row is None:
                continue

            coefficient = math.prod(
                term[1] for term in term_combination
            )
            transformation[row, column] += coefficient

    target_coefficients, *_ = np.linalg.lstsq(
        transformation,
        np.asarray(monomial_coefficients, dtype=np.float64),
        rcond=None,
    )

    residual = (
        transformation @ target_coefficients
        - np.asarray(monomial_coefficients, dtype=np.float64)
    )
    residual_scale = max(
        np.linalg.norm(monomial_coefficients),
        EPSILON_FLOAT64,
    )

    if np.linalg.norm(residual) / residual_scale > 1e-9:
        warnings.warn(
            f"Basis conversion residual for {basis} is "
            f"{np.linalg.norm(residual) / residual_scale:.3e}.",
            RuntimeWarning,
            stacklevel=2,
        )

    return np.asarray(target_coefficients, dtype=np.float64)


# =============================================================================
# Basis evaluation
# =============================================================================


def map_to_canonical(
    points: np.ndarray,
    domain: AxisAlignedBox,
    dtype: np.dtype,
) -> np.ndarray:
    points = np.asarray(points, dtype=dtype)
    lower = np.asarray(domain.lower, dtype=dtype)
    upper = np.asarray(domain.upper, dtype=dtype)
    return (
        dtype.type(2.0) * (points - lower) / (upper - lower)
        - dtype.type(1.0)
    )


def univariate_basis_table(
    values: np.ndarray,
    maximum_degree: int,
    basis: str,
    dtype: np.dtype,
) -> np.ndarray:
    """Evaluate degrees 0,...,maximum_degree at a vector of values."""

    values = np.asarray(values, dtype=dtype)
    table = np.empty(
        (values.size, maximum_degree + 1),
        dtype=dtype,
    )
    table[:, 0] = dtype.type(1.0)

    if maximum_degree == 0:
        return table

    table[:, 1] = values

    if basis == "monomial":
        for degree in range(2, maximum_degree + 1):
            table[:, degree] = table[:, degree - 1] * values
        return table

    if basis == "legendre":
        for degree in range(2, maximum_degree + 1):
            n = degree - 1
            table[:, degree] = (
                dtype.type(2 * n + 1)
                * values
                * table[:, degree - 1]
                - dtype.type(n) * table[:, degree - 2]
            ) / dtype.type(n + 1)
        return table

    if basis == "chebyshev":
        for degree in range(2, maximum_degree + 1):
            table[:, degree] = (
                dtype.type(2.0)
                * values
                * table[:, degree - 1]
                - table[:, degree - 2]
            )
        return table

    raise ValueError(f"Unsupported basis: {basis}")


def build_basis_matrix(
    points: np.ndarray,
    indices: Sequence[tuple[int, ...]],
    basis: str,
    domain: AxisAlignedBox,
    dtype_name: str,
) -> np.ndarray:
    """Construct the product-basis evaluation matrix."""

    dtype = np.dtype(SUPPORTED_DTYPES[dtype_name])
    points = np.asarray(points, dtype=dtype)

    if points.ndim != 2:
        raise ValueError("points must have shape (N, dimension)")
    if points.shape[1] != domain.dimension:
        raise ValueError("point dimension does not match domain")

    maximum_degree = max(max(alpha) for alpha in indices)

    if basis == "monomial":
        evaluation_points = points
    else:
        evaluation_points = map_to_canonical(points, domain, dtype)

    per_axis_tables = [
        univariate_basis_table(
            evaluation_points[:, axis],
            maximum_degree,
            basis,
            dtype,
        )
        for axis in range(domain.dimension)
    ]

    matrix = np.ones(
        (points.shape[0], len(indices)),
        dtype=dtype,
    )

    for column, alpha in enumerate(indices):
        for axis, degree in enumerate(alpha):
            matrix[:, column] *= per_axis_tables[axis][:, degree]

    return matrix


def evaluate_polynomial(
    points: np.ndarray,
    indices: Sequence[tuple[int, ...]],
    coefficients: np.ndarray,
    basis: str,
    domain: AxisAlignedBox,
    dtype_name: str,
) -> np.ndarray:
    matrix = build_basis_matrix(
        points=points,
        indices=indices,
        basis=basis,
        domain=domain,
        dtype_name=dtype_name,
    )
    dtype = SUPPORTED_DTYPES[dtype_name]
    coefficients_typed = np.asarray(coefficients, dtype=dtype)
    return matrix @ coefficients_typed


# =============================================================================
# Analytical integration
# =============================================================================


def monomial_box_integral(
    alpha: tuple[int, ...],
    box: AxisAlignedBox,
) -> float:
    factors = []

    for exponent, lower, upper in zip(
        alpha,
        box.lower,
        box.upper,
        strict=True,
    ):
        factors.append(
            (
                float(upper) ** (exponent + 1)
                - float(lower) ** (exponent + 1)
            )
            / float(exponent + 1)
        )

    return float(math.prod(factors))


def basis_function_box_integral(
    alpha: tuple[int, ...],
    basis: str,
    representation_domain: AxisAlignedBox,
    integration_box: AxisAlignedBox,
) -> float:
    """
    Integrate one product-basis function over an axis-aligned box.
    """

    factors: list[float] = []

    for axis, degree in enumerate(alpha):
        polynomial = canonical_basis_polynomial_in_physical_coordinate(
            basis=basis,
            degree=degree,
            lower=float(representation_domain.lower[axis]),
            upper=float(representation_domain.upper[axis]),
        )
        antiderivative = polynomial.integ()

        upper_value = antiderivative(float(integration_box.upper[axis]))
        lower_value = antiderivative(float(integration_box.lower[axis]))
        factors.append(float(upper_value - lower_value))

    return float(math.prod(factors))


def integrate_basis_polynomial_over_box(
    indices: Sequence[tuple[int, ...]],
    coefficients: np.ndarray,
    basis: str,
    representation_domain: AxisAlignedBox,
    integration_box: AxisAlignedBox,
    dtype_name: str,
) -> float:
    """
    Integrate a represented polynomial while accumulating in the requested dtype.
    """

    dtype = SUPPORTED_DTYPES[dtype_name]
    accumulator = dtype(0.0)
    typed_coefficients = np.asarray(coefficients, dtype=dtype)

    for alpha, coefficient in zip(
        indices,
        typed_coefficients,
        strict=True,
    ):
        integral = basis_function_box_integral(
            alpha=alpha,
            basis=basis,
            representation_domain=representation_domain,
            integration_box=integration_box,
        )
        accumulator = dtype(
            accumulator + dtype(coefficient) * dtype(integral)
        )

    return float(accumulator)


def integrate_region(
    indices: Sequence[tuple[int, ...]],
    coefficients: np.ndarray,
    basis: str,
    representation_domain: AxisAlignedBox,
    outer_box: AxisAlignedBox,
    obstacle_box: AxisAlignedBox | None,
    dtype_name: str,
) -> float:
    outer_integral = integrate_basis_polynomial_over_box(
        indices=indices,
        coefficients=coefficients,
        basis=basis,
        representation_domain=representation_domain,
        integration_box=outer_box,
        dtype_name=dtype_name,
    )

    if obstacle_box is None:
        return outer_integral

    obstacle_integral = integrate_basis_polynomial_over_box(
        indices=indices,
        coefficients=coefficients,
        basis=basis,
        representation_domain=representation_domain,
        integration_box=obstacle_box,
        dtype_name=dtype_name,
    )

    dtype = SUPPORTED_DTYPES[dtype_name]
    return float(dtype(dtype(outer_integral) - dtype(obstacle_integral)))


def reference_region_integral(
    monomial_indices: Sequence[tuple[int, ...]],
    monomial_coefficients: np.ndarray,
    outer_box: AxisAlignedBox,
    obstacle_box: AxisAlignedBox | None,
) -> float:
    outer_value = math.fsum(
        float(coefficient) * monomial_box_integral(alpha, outer_box)
        for alpha, coefficient in zip(
            monomial_indices,
            monomial_coefficients,
            strict=True,
        )
    )

    if obstacle_box is None:
        return float(outer_value)

    obstacle_value = math.fsum(
        float(coefficient) * monomial_box_integral(alpha, obstacle_box)
        for alpha, coefficient in zip(
            monomial_indices,
            monomial_coefficients,
            strict=True,
        )
    )

    return float(outer_value - obstacle_value)


# =============================================================================
# Metrics
# =============================================================================


def safe_relative_error(
    computed: float,
    reference: float,
) -> float:
    denominator = max(abs(reference), EPSILON_FLOAT64)
    return abs(computed - reference) / denominator


def perturb_coefficients(
    coefficients: np.ndarray,
    magnitude: float,
    rng: np.random.Generator,
    dtype_name: str,
) -> tuple[np.ndarray, bool]:
    dtype = SUPPORTED_DTYPES[dtype_name]
    coefficients_typed = np.asarray(coefficients, dtype=dtype)

    direction = rng.normal(size=coefficients_typed.shape).astype(np.float64)
    direction_norm = np.linalg.norm(direction)

    if direction_norm == 0.0:
        direction.flat[0] = 1.0
        direction_norm = 1.0

    direction /= direction_norm

    coefficient_norm = max(
        np.linalg.norm(coefficients_typed.astype(np.float64)),
        EPSILON_FLOAT64,
    )

    perturbation = magnitude * coefficient_norm * direction
    perturbed = np.asarray(
        coefficients_typed.astype(np.float64) + perturbation,
        dtype=dtype,
    )

    rounded_away = np.array_equal(
        perturbed,
        coefficients_typed,
    )

    return perturbed, rounded_away


def coefficient_perturbation_sensitivity(
    original_integral: float,
    perturbed_integral: float,
    perturbation_magnitude: float,
) -> float:
    denominator = (
        perturbation_magnitude
        * max(abs(original_integral), EPSILON_FLOAT64)
    )
    return abs(perturbed_integral - original_integral) / denominator


def condition_and_rank(
    matrix: np.ndarray,
    dtype_name: str,
) -> tuple[float, int, float]:
    """
    Compute condition number and numerical rank using a dtype-aware tolerance.
    """

    matrix64 = np.asarray(matrix, dtype=np.float64)
    singular_values = np.linalg.svd(matrix64, compute_uv=False)

    if singular_values.size == 0:
        return float("nan"), 0, float("nan")

    largest = float(singular_values[0])
    smallest = float(singular_values[-1])

    if smallest == 0.0:
        condition_number = float("inf")
    else:
        condition_number = largest / smallest

    dtype = SUPPORTED_DTYPES[dtype_name]
    tolerance = (
        max(matrix.shape)
        * np.finfo(dtype).eps
        * largest
    )

    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    rank_fraction = numerical_rank / matrix.shape[1]

    return condition_number, numerical_rank, rank_fraction


def add_relative_value_noise(
    values: np.ndarray,
    magnitude: float,
    rng: np.random.Generator,
    dtype_name: str,
) -> tuple[np.ndarray, float]:
    """
    Add a controlled random perturbation to sampled function values.

    The requested perturbation has Euclidean norm

        magnitude * max(||values||_2, eps).
    """

    dtype = SUPPORTED_DTYPES[dtype_name]
    values_typed = np.asarray(values, dtype=dtype)

    direction = rng.normal(size=values_typed.shape).astype(np.float64)
    norm = np.linalg.norm(direction)

    if norm == 0.0:
        direction.flat[0] = 1.0
        norm = 1.0

    direction /= norm

    value_norm = max(
        np.linalg.norm(values_typed.astype(np.float64)),
        EPSILON_FLOAT64,
    )
    noise = magnitude * value_norm * direction

    noisy_values = np.asarray(
        values_typed.astype(np.float64) + noise,
        dtype=dtype,
    )

    realised_noise = (
        noisy_values.astype(np.float64)
        - values_typed.astype(np.float64)
    )
    relative_noise_norm = (
        np.linalg.norm(realised_noise)
        / value_norm
    )

    return noisy_values, float(relative_noise_norm)


def recover_coefficients_from_values(
    matrix: np.ndarray,
    noisy_values: np.ndarray,
    dtype_name: str,
) -> np.ndarray:
    """
    Recover coefficients by least squares.

    NumPy's linear algebra backend may internally operate in float64 on some
    platforms. The input matrix and observations are first rounded to the
    requested arithmetic precision, so the measured effect still includes the
    information lost in that representation.
    """

    dtype = SUPPORTED_DTYPES[dtype_name]

    matrix_typed = np.asarray(matrix, dtype=dtype)
    values_typed = np.asarray(noisy_values, dtype=dtype)

    recovered, *_ = np.linalg.lstsq(
        matrix_typed.astype(np.float64),
        values_typed.astype(np.float64),
        rcond=None,
    )

    return np.asarray(recovered, dtype=dtype)


def recovery_metrics(
    original_coefficients: np.ndarray,
    recovered_coefficients: np.ndarray,
    relative_value_noise: float,
) -> tuple[float, float]:
    original = np.asarray(original_coefficients, dtype=np.float64)
    recovered = np.asarray(recovered_coefficients, dtype=np.float64)

    coefficient_error = (
        np.linalg.norm(recovered - original)
        / max(np.linalg.norm(original), EPSILON_FLOAT64)
    )

    amplification = coefficient_error / max(
        relative_value_noise,
        EPSILON_FLOAT64,
    )

    return float(coefficient_error), float(amplification)


# =============================================================================
# Point construction
# =============================================================================


def sample_uniform_box_points(
    box: AxisAlignedBox,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count < 1:
        raise ValueError("count must be positive")

    unit_points = rng.random((count, box.dimension))
    return (
        box.lower[None, :]
        + unit_points * (box.upper - box.lower)[None, :]
    )


def construct_recovery_points(
    outer_box: AxisAlignedBox,
    obstacle_box: AxisAlignedBox | None,
    basis_count: int,
    oversampling_factor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a common point set inside the feasible region.

    Rejection sampling is used for obstacle scenarios.
    """

    target_count = max(
        basis_count,
        int(math.ceil(oversampling_factor * basis_count)),
    )

    accepted: list[np.ndarray] = []
    accepted_count = 0

    while accepted_count < target_count:
        remaining = target_count - accepted_count
        candidate_count = max(remaining * 2, 32)

        candidates = sample_uniform_box_points(
            box=outer_box,
            count=candidate_count,
            rng=rng,
        )

        if obstacle_box is not None:
            inside_obstacle = np.all(
                (candidates >= obstacle_box.lower[None, :])
                & (candidates <= obstacle_box.upper[None, :]),
                axis=1,
            )
            candidates = candidates[~inside_obstacle]

        if candidates.size == 0:
            continue

        accepted.append(candidates)
        accepted_count += candidates.shape[0]

    return np.concatenate(accepted, axis=0)[:target_count]


def sampled_density_has_negative_values(
    monomial_indices: Sequence[tuple[int, ...]],
    monomial_coefficients: np.ndarray,
    domain: AxisAlignedBox,
    seed: int,
    sample_count: int = 256,
) -> bool:
    rng = np.random.default_rng(seed)
    points = sample_uniform_box_points(domain, sample_count, rng)
    values = evaluate_polynomial(
        points=points,
        indices=monomial_indices,
        coefficients=monomial_coefficients,
        basis="monomial",
        domain=domain,
        dtype_name="float64",
    )
    return bool(np.any(values < -1e-12))


# =============================================================================
# Scenario construction
# =============================================================================


def centred_obstacle(
    dimension: int,
    half_width: float,
) -> AxisAlignedBox:
    if not 0.0 < half_width < 0.5:
        raise ValueError("obstacle half-width must lie in (0, 0.5)")

    centre = np.full(dimension, 0.5, dtype=np.float64)
    return AxisAlignedBox(
        lower=centre - half_width,
        upper=centre + half_width,
    )


def schedule_half_widths(
    schedule: str,
    trajectory_steps: int,
    minimum: float,
    maximum: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if trajectory_steps < 1:
        raise ValueError("trajectory_steps must be positive")
    if not 0.0 < minimum <= maximum < 0.5:
        raise ValueError("Invalid obstacle half-width range")

    if trajectory_steps == 1:
        return np.asarray([(minimum + maximum) / 2.0])

    if schedule == "shrinking":
        return np.linspace(maximum, minimum, trajectory_steps)

    if schedule == "expanding":
        return np.linspace(minimum, maximum, trajectory_steps)

    if schedule == "oscillating":
        phase = np.linspace(
            0.0,
            2.0 * np.pi,
            trajectory_steps,
            endpoint=False,
        )
        midpoint = 0.5 * (minimum + maximum)
        amplitude = 0.5 * (maximum - minimum)
        return midpoint + amplitude * np.sin(phase)

    if schedule == "pulsed":
        values = np.empty(trajectory_steps, dtype=np.float64)
        for step in range(trajectory_steps):
            values[step] = maximum if (step // 2) % 2 else minimum
        return values

    if schedule == "random_walk":
        values = np.empty(trajectory_steps, dtype=np.float64)
        values[0] = 0.5 * (minimum + maximum)
        step_scale = 0.15 * (maximum - minimum)

        for step in range(1, trajectory_steps):
            values[step] = np.clip(
                values[step - 1] + rng.normal(scale=step_scale),
                minimum,
                maximum,
            )

        return values

    raise ValueError(f"Unsupported schedule: {schedule}")


# =============================================================================
# Single-configuration evaluation
# =============================================================================


def evaluate_configuration(
    *,
    scenario: str,
    schedule: str,
    trajectory_step: int,
    trajectory_steps: int,
    outer_box: AxisAlignedBox,
    obstacle_box: AxisAlignedBox | None,
    obstacle_half_width: float,
    monomial_indices: Sequence[tuple[int, ...]],
    monomial_coefficients: np.ndarray,
    basis_indices: Sequence[tuple[int, ...]],
    basis_coefficients: np.ndarray,
    basis: str,
    dtype_name: str,
    dimension: int,
    source_degree: int,
    density_degree: int,
    trial: int,
    coefficient_seed: int,
    perturbation_seed: int,
    point_seed: int,
    value_noise_seed: int,
    density_offset: float,
    coefficient_scale: float,
    perturbation_magnitude: float,
    value_noise_magnitude: float,
    recovery_oversampling_factor: float,
    sampled_negative_density: bool,
) -> BenchmarkRecord:
    total_start = time.perf_counter()

    reference_integral = reference_region_integral(
        monomial_indices=monomial_indices,
        monomial_coefficients=monomial_coefficients,
        outer_box=outer_box,
        obstacle_box=obstacle_box,
    )

    integration_start = time.perf_counter()

    computed_integral = integrate_region(
        indices=basis_indices,
        coefficients=basis_coefficients,
        basis=basis,
        representation_domain=outer_box,
        outer_box=outer_box,
        obstacle_box=obstacle_box,
        dtype_name=dtype_name,
    )

    perturbation_rng = np.random.default_rng(perturbation_seed)
    perturbed_coefficients, perturbation_rounded_away = perturb_coefficients(
        coefficients=basis_coefficients,
        magnitude=perturbation_magnitude,
        rng=perturbation_rng,
        dtype_name=dtype_name,
    )

    perturbed_integral = integrate_region(
        indices=basis_indices,
        coefficients=perturbed_coefficients,
        basis=basis,
        representation_domain=outer_box,
        outer_box=outer_box,
        obstacle_box=obstacle_box,
        dtype_name=dtype_name,
    )

    integration_ms = 1000.0 * (
        time.perf_counter() - integration_start
    )

    relative_integration_error = safe_relative_error(
        computed=computed_integral,
        reference=reference_integral,
    )
    perturbation_sensitivity = coefficient_perturbation_sensitivity(
        original_integral=computed_integral,
        perturbed_integral=perturbed_integral,
        perturbation_magnitude=perturbation_magnitude,
    )

    point_rng = np.random.default_rng(point_seed)
    recovery_points = construct_recovery_points(
        outer_box=outer_box,
        obstacle_box=obstacle_box,
        basis_count=len(basis_indices),
        oversampling_factor=recovery_oversampling_factor,
        rng=point_rng,
    )

    conditioning_start = time.perf_counter()

    basis_matrix = build_basis_matrix(
        points=recovery_points,
        indices=basis_indices,
        basis=basis,
        domain=outer_box,
        dtype_name=dtype_name,
    )

    condition_number, numerical_rank, numerical_rank_fraction = (
        condition_and_rank(
            matrix=basis_matrix,
            dtype_name=dtype_name,
        )
    )

    conditioning_ms = 1000.0 * (
        time.perf_counter() - conditioning_start
    )

    recovery_start = time.perf_counter()

    clean_values = (
        basis_matrix
        @ np.asarray(
            basis_coefficients,
            dtype=SUPPORTED_DTYPES[dtype_name],
        )
    )

    value_noise_rng = np.random.default_rng(value_noise_seed)
    noisy_values, realised_relative_value_noise = add_relative_value_noise(
        values=clean_values,
        magnitude=value_noise_magnitude,
        rng=value_noise_rng,
        dtype_name=dtype_name,
    )

    recovered_coefficients = recover_coefficients_from_values(
        matrix=basis_matrix,
        noisy_values=noisy_values,
        dtype_name=dtype_name,
    )

    coefficient_recovery_error, noise_amplification = recovery_metrics(
        original_coefficients=np.asarray(
            basis_coefficients,
            dtype=SUPPORTED_DTYPES[dtype_name],
        ),
        recovered_coefficients=recovered_coefficients,
        relative_value_noise=realised_relative_value_noise,
    )

    recovered_integral = integrate_region(
        indices=basis_indices,
        coefficients=recovered_coefficients,
        basis=basis,
        representation_domain=outer_box,
        outer_box=outer_box,
        obstacle_box=obstacle_box,
        dtype_name=dtype_name,
    )

    recovered_integral_relative_error = safe_relative_error(
        computed=recovered_integral,
        reference=reference_integral,
    )

    recovery_ms = 1000.0 * (
        time.perf_counter() - recovery_start
    )

    total_evaluation_ms = 1000.0 * (
        time.perf_counter() - total_start
    )

    return BenchmarkRecord(
        scenario=scenario,
        schedule=schedule,
        trajectory_step=trajectory_step,
        trajectory_steps=trajectory_steps,
        dimension=dimension,
        source_polynomial_degree=source_degree,
        density_polynomial_degree=density_degree,
        trial=trial,
        basis=basis,
        dtype=dtype_name,
        coefficient_seed=coefficient_seed,
        perturbation_seed=perturbation_seed,
        point_seed=point_seed,
        value_noise_seed=value_noise_seed,
        density_offset=density_offset,
        coefficient_scale=coefficient_scale,
        perturbation_magnitude=perturbation_magnitude,
        value_noise_magnitude=value_noise_magnitude,
        obstacle_half_width=obstacle_half_width,
        basis_count=len(basis_indices),
        recovery_point_count=recovery_points.shape[0],
        reference_integral=reference_integral,
        computed_integral=computed_integral,
        perturbed_integral=perturbed_integral,
        recovered_integral=recovered_integral,
        relative_integration_error=relative_integration_error,
        perturbation_sensitivity=perturbation_sensitivity,
        basis_condition_number=condition_number,
        numerical_rank=numerical_rank,
        numerical_rank_fraction=numerical_rank_fraction,
        function_value_noise_relative_norm=realised_relative_value_noise,
        coefficient_recovery_relative_error=coefficient_recovery_error,
        coefficient_noise_amplification=noise_amplification,
        recovered_integral_relative_error=recovered_integral_relative_error,
        perturbation_rounded_away=perturbation_rounded_away,
        sampled_negative_density=sampled_negative_density,
        integration_ms=integration_ms,
        conditioning_ms=conditioning_ms,
        recovery_ms=recovery_ms,
        total_evaluation_ms=total_evaluation_ms,
    )


# =============================================================================
# Result output
# =============================================================================


def write_records(
    records: Sequence[BenchmarkRecord],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        raise ValueError("No benchmark records were produced.")

    fieldnames = list(asdict(records[0]).keys())

    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def finite_record(record: BenchmarkRecord) -> bool:
    """
    Check scientific outputs for finite values.

    Infinite condition numbers are allowed because they legitimately indicate a
    numerically singular sampled basis matrix.
    """

    required_finite = (
        record.reference_integral,
        record.computed_integral,
        record.perturbed_integral,
        record.recovered_integral,
        record.relative_integration_error,
        record.perturbation_sensitivity,
        record.numerical_rank_fraction,
        record.function_value_noise_relative_norm,
        record.coefficient_recovery_relative_error,
        record.coefficient_noise_amplification,
        record.recovered_integral_relative_error,
        record.integration_ms,
        record.conditioning_ms,
        record.recovery_ms,
        record.total_evaluation_ms,
    )

    return bool(np.all(np.isfinite(required_finite)))


def median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.median(array))


def grouped_summary(
    records: Sequence[BenchmarkRecord],
) -> list[dict[str, object]]:
    groups: dict[
        tuple[str, str, str],
        list[BenchmarkRecord],
    ] = {}

    for record in records:
        key = (record.scenario, record.basis, record.dtype)
        groups.setdefault(key, []).append(record)

    rows: list[dict[str, object]] = []

    for key in sorted(groups):
        scenario, basis, dtype_name = key
        group = groups[key]

        finite_conditions = [
            record.basis_condition_number
            for record in group
            if np.isfinite(record.basis_condition_number)
        ]

        rows.append(
            {
                "scenario": scenario,
                "basis": basis,
                "dtype": dtype_name,
                "records": len(group),
                "relative_integration_error": median(
                    r.relative_integration_error for r in group
                ),
                "perturbation_sensitivity": median(
                    r.perturbation_sensitivity for r in group
                ),
                "basis_condition_number": median(finite_conditions),
                "numerical_rank_fraction": median(
                    r.numerical_rank_fraction for r in group
                ),
                "coefficient_recovery_relative_error": median(
                    r.coefficient_recovery_relative_error for r in group
                ),
                "coefficient_noise_amplification": median(
                    r.coefficient_noise_amplification for r in group
                ),
                "recovered_integral_relative_error": median(
                    r.recovered_integral_relative_error for r in group
                ),
                "total_evaluation_ms": median(
                    r.total_evaluation_ms for r in group
                ),
            }
        )

    return rows


def maximum_record(
    records: Sequence[BenchmarkRecord],
    attribute: str,
) -> BenchmarkRecord:
    return max(records, key=lambda record: float(getattr(record, attribute)))


def build_summary(
    records: Sequence[BenchmarkRecord],
    configuration: dict[str, object],
    elapsed_seconds: float,
) -> dict[str, object]:
    finite_count = sum(finite_record(record) for record in records)
    infinite_condition_count = sum(
        not np.isfinite(record.basis_condition_number)
        for record in records
    )

    maximum_integration_error = maximum_record(
        records,
        "relative_integration_error",
    )
    maximum_noise_amplification = maximum_record(
        records,
        "coefficient_noise_amplification",
    )
    maximum_recovered_integral_error = maximum_record(
        records,
        "recovered_integral_relative_error",
    )

    return {
        "configuration": configuration,
        "elapsed_seconds": elapsed_seconds,
        "records": len(records),
        "finite_records": finite_count,
        "infinite_condition_numbers": infinite_condition_count,
        "rounded_away_perturbations": sum(
            record.perturbation_rounded_away
            for record in records
        ),
        "records_with_sampled_negative_density_values": sum(
            record.sampled_negative_density
            for record in records
        ),
        "median_metrics_by_scenario_basis_dtype": grouped_summary(records),
        "maximum_relative_integration_error": asdict(
            maximum_integration_error
        ),
        "maximum_coefficient_noise_amplification": asdict(
            maximum_noise_amplification
        ),
        "maximum_recovered_integral_relative_error": asdict(
            maximum_recovered_integral_error
        ),
    }


def write_summary(
    summary: dict[str, object],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, allow_nan=True)


def print_grouped_summary(rows: Sequence[dict[str, object]]) -> None:
    headers = (
        "scenario",
        "basis",
        "dtype",
        "relative_integration_error",
        "perturbation_sensitivity",
        "basis_condition_number",
        "numerical_rank_fraction",
        "coefficient_noise_amplification",
        "recovered_integral_relative_error",
        "total_evaluation_ms",
    )

    widths = {
        header: max(
            len(header),
            max(
                (
                    len(
                        f"{row[header]:.6e}"
                        if isinstance(row[header], float)
                        else str(row[header])
                    )
                    for row in rows
                ),
                default=0,
            ),
        )
        for header in headers
    }

    print(" ".join(header.rjust(widths[header]) for header in headers))

    for row in rows:
        cells = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                text = f"{value:.6e}"
            else:
                text = str(value)
            cells.append(text.rjust(widths[header]))
        print(" ".join(cells))


# =============================================================================
# Main benchmark loop
# =============================================================================


def derive_seed(
    base_seed: int,
    dimension: int,
    degree: int,
    trial: int,
    stream: int,
    trajectory_step: int = 0,
) -> int:
    seed_sequence = np.random.SeedSequence(
        [
            base_seed,
            dimension,
            degree,
            trial,
            stream,
            trajectory_step,
        ]
    )
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def run_benchmark(args: argparse.Namespace) -> list[BenchmarkRecord]:
    dimensions = sorted(set(args.dimensions))
    source_degrees = sorted(set(args.degrees))
    dtypes = list(dict.fromkeys(args.dtypes))
    schedules = list(dict.fromkeys(args.schedules))

    invalid_dtypes = set(dtypes) - set(SUPPORTED_DTYPES)
    if invalid_dtypes:
        raise ValueError(f"Unsupported dtypes: {sorted(invalid_dtypes)}")

    invalid_schedules = set(schedules) - set(SUPPORTED_SCHEDULES)
    if invalid_schedules:
        raise ValueError(
            f"Unsupported schedules: {sorted(invalid_schedules)}"
        )

    outer_boxes = {
        dimension: AxisAlignedBox(
            lower=np.zeros(dimension, dtype=np.float64),
            upper=np.ones(dimension, dtype=np.float64),
        )
        for dimension in dimensions
    }

    expected_records_per_trial = (
        2 * len(BASIS_NAMES) * len(dtypes)
        + len(schedules)
        * args.trajectory_steps
        * len(BASIS_NAMES)
        * len(dtypes)
    )
    expected_records = (
        len(dimensions)
        * len(source_degrees)
        * args.trials
        * expected_records_per_trial
    )

    base_configurations = (
        len(dimensions)
        * len(source_degrees)
        * args.trials
    )

    print()
    print("Polytope benchmark configuration")
    print("================================")
    print(f"dimensions: {dimensions}")
    print(f"source degrees: {source_degrees}")
    print(
        "density degrees: "
        f"{[2 * degree for degree in source_degrees]}"
    )
    print(f"bases: {list(BASIS_NAMES)}")
    print(f"dtypes: {dtypes}")
    print(f"trials: {args.trials}")
    print("static scenarios: ['convex_box', 'box_with_obstacle']")
    print(f"dynamic schedules: {schedules}")
    print(f"trajectory steps: {args.trajectory_steps}")
    print(f"density offset: {args.density_offset}")
    print(f"coefficient scale: {args.coefficient_scale}")
    print(f"perturbation magnitude: {args.perturbation_magnitude}")
    print(f"value-noise magnitude: {args.value_noise_magnitude}")
    print(
        "recovery oversampling factor: "
        f"{args.recovery_oversampling_factor}"
    )
    print("outer domain: [0.0, 1.0]^n")
    print(
        "static obstacle half-width: "
        f"{args.static_obstacle_half_width}"
    )
    print(
        "dynamic obstacle half-width range: "
        f"[{args.dynamic_min_half_width}, "
        f"{args.dynamic_max_half_width}]"
    )
    print(f"base seed: {args.base_seed}")
    print(f"expected records: {expected_records}")
    print(f"results path: {args.results_path}")
    print(f"summary path: {args.summary_path}")
    print()

    records: list[BenchmarkRecord] = []
    progress_index = 0

    for dimension in dimensions:
        outer_box = outer_boxes[dimension]

        for source_degree in source_degrees:
            density_degree = 2 * source_degree
            density_indices = total_degree_indices(
                dimension,
                density_degree,
            )

            for trial in range(args.trials):
                progress_index += 1
                print(
                    f"[{progress_index:4d}/{base_configurations:<4d}] "
                    f"dimension={dimension}, "
                    f"source_degree={source_degree}, "
                    f"density_degree={density_degree}, "
                    f"trial={trial}"
                )

                coefficient_seed = derive_seed(
                    args.base_seed,
                    dimension,
                    source_degree,
                    trial,
                    stream=1,
                )
                coefficient_rng = np.random.default_rng(coefficient_seed)

                source_indices, source_coefficients = (
                    generate_source_polynomial(
                        dimension=dimension,
                        degree=source_degree,
                        coefficient_scale=args.coefficient_scale,
                        rng=coefficient_rng,
                    )
                )

                monomial_indices, monomial_coefficients = (
                    square_monomial_polynomial(
                        source_indices=source_indices,
                        source_coefficients=source_coefficients,
                        density_offset=args.density_offset,
                    )
                )

                sampled_negative_density = (
                    sampled_density_has_negative_values(
                        monomial_indices=monomial_indices,
                        monomial_coefficients=monomial_coefficients,
                        domain=outer_box,
                        seed=derive_seed(
                            args.base_seed,
                            dimension,
                            source_degree,
                            trial,
                            stream=2,
                        ),
                    )
                )

                basis_coefficients_by_basis = {
                    basis: convert_monomial_to_basis(
                        monomial_indices=monomial_indices,
                        monomial_coefficients=monomial_coefficients,
                        target_indices=density_indices,
                        basis=basis,
                        domain=outer_box,
                    )
                    for basis in BASIS_NAMES
                }

                static_obstacle = centred_obstacle(
                    dimension=dimension,
                    half_width=args.static_obstacle_half_width,
                )

                static_scenarios = (
                    (
                        "convex_box",
                        "static",
                        None,
                        float("nan"),
                    ),
                    (
                        "box_with_obstacle",
                        "static",
                        static_obstacle,
                        args.static_obstacle_half_width,
                    ),
                )

                for scenario, schedule, obstacle_box, half_width in static_scenarios:
                    for basis in BASIS_NAMES:
                        for dtype_name in dtypes:
                            records.append(
                                evaluate_configuration(
                                    scenario=scenario,
                                    schedule=schedule,
                                    trajectory_step=0,
                                    trajectory_steps=1,
                                    outer_box=outer_box,
                                    obstacle_box=obstacle_box,
                                    obstacle_half_width=half_width,
                                    monomial_indices=monomial_indices,
                                    monomial_coefficients=monomial_coefficients,
                                    basis_indices=density_indices,
                                    basis_coefficients=(
                                        basis_coefficients_by_basis[basis]
                                    ),
                                    basis=basis,
                                    dtype_name=dtype_name,
                                    dimension=dimension,
                                    source_degree=source_degree,
                                    density_degree=density_degree,
                                    trial=trial,
                                    coefficient_seed=coefficient_seed,
                                    perturbation_seed=derive_seed(
                                        args.base_seed,
                                        dimension,
                                        source_degree,
                                        trial,
                                        stream=10
                                        + BASIS_NAMES.index(basis) * 2
                                        + dtypes.index(dtype_name),
                                    ),
                                    point_seed=derive_seed(
                                        args.base_seed,
                                        dimension,
                                        source_degree,
                                        trial,
                                        stream=30
                                        + BASIS_NAMES.index(basis) * 2
                                        + dtypes.index(dtype_name),
                                    ),
                                    value_noise_seed=derive_seed(
                                        args.base_seed,
                                        dimension,
                                        source_degree,
                                        trial,
                                        stream=50
                                        + BASIS_NAMES.index(basis) * 2
                                        + dtypes.index(dtype_name),
                                    ),
                                    density_offset=args.density_offset,
                                    coefficient_scale=args.coefficient_scale,
                                    perturbation_magnitude=(
                                        args.perturbation_magnitude
                                    ),
                                    value_noise_magnitude=(
                                        args.value_noise_magnitude
                                    ),
                                    recovery_oversampling_factor=(
                                        args.recovery_oversampling_factor
                                    ),
                                    sampled_negative_density=(
                                        sampled_negative_density
                                    ),
                                )
                            )

                for schedule_index, schedule in enumerate(schedules):
                    schedule_seed = derive_seed(
                        args.base_seed,
                        dimension,
                        source_degree,
                        trial,
                        stream=100 + schedule_index,
                    )
                    schedule_rng = np.random.default_rng(schedule_seed)

                    half_widths = schedule_half_widths(
                        schedule=schedule,
                        trajectory_steps=args.trajectory_steps,
                        minimum=args.dynamic_min_half_width,
                        maximum=args.dynamic_max_half_width,
                        rng=schedule_rng,
                    )

                    for trajectory_step, half_width in enumerate(half_widths):
                        obstacle_box = centred_obstacle(
                            dimension=dimension,
                            half_width=float(half_width),
                        )

                        for basis in BASIS_NAMES:
                            for dtype_name in dtypes:
                                basis_index = BASIS_NAMES.index(basis)
                                dtype_index = dtypes.index(dtype_name)

                                records.append(
                                    evaluate_configuration(
                                        scenario=(
                                            "dynamic_box_with_obstacle"
                                        ),
                                        schedule=schedule,
                                        trajectory_step=trajectory_step,
                                        trajectory_steps=args.trajectory_steps,
                                        outer_box=outer_box,
                                        obstacle_box=obstacle_box,
                                        obstacle_half_width=float(half_width),
                                        monomial_indices=monomial_indices,
                                        monomial_coefficients=(
                                            monomial_coefficients
                                        ),
                                        basis_indices=density_indices,
                                        basis_coefficients=(
                                            basis_coefficients_by_basis[basis]
                                        ),
                                        basis=basis,
                                        dtype_name=dtype_name,
                                        dimension=dimension,
                                        source_degree=source_degree,
                                        density_degree=density_degree,
                                        trial=trial,
                                        coefficient_seed=coefficient_seed,
                                        perturbation_seed=derive_seed(
                                            args.base_seed,
                                            dimension,
                                            source_degree,
                                            trial,
                                            stream=200
                                            + schedule_index * 20
                                            + basis_index * 2
                                            + dtype_index,
                                            trajectory_step=trajectory_step,
                                        ),
                                        point_seed=derive_seed(
                                            args.base_seed,
                                            dimension,
                                            source_degree,
                                            trial,
                                            stream=400
                                            + schedule_index * 20
                                            + basis_index * 2
                                            + dtype_index,
                                            trajectory_step=trajectory_step,
                                        ),
                                        value_noise_seed=derive_seed(
                                            args.base_seed,
                                            dimension,
                                            source_degree,
                                            trial,
                                            stream=600
                                            + schedule_index * 20
                                            + basis_index * 2
                                            + dtype_index,
                                            trajectory_step=trajectory_step,
                                        ),
                                        density_offset=args.density_offset,
                                        coefficient_scale=(
                                            args.coefficient_scale
                                        ),
                                        perturbation_magnitude=(
                                            args.perturbation_magnitude
                                        ),
                                        value_noise_magnitude=(
                                            args.value_noise_magnitude
                                        ),
                                        recovery_oversampling_factor=(
                                            args.recovery_oversampling_factor
                                        ),
                                        sampled_negative_density=(
                                            sampled_negative_density
                                        ),
                                    )
                                )

    if len(records) != expected_records:
        raise RuntimeError(
            f"Expected {expected_records} records but produced "
            f"{len(records)}."
        )

    return records


# =============================================================================
# Command-line interface
# =============================================================================


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark polynomial basis stability over static and dynamic "
            "axis-aligned constrained polytopes."
        )
    )

    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=positive_integer,
        default=[2, 3],
    )
    parser.add_argument(
        "--degrees",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 5],
    )
    parser.add_argument(
        "--trials",
        type=positive_integer,
        default=10,
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=sorted(SUPPORTED_DTYPES),
        default=["float32", "float64"],
    )
    parser.add_argument(
        "--schedules",
        nargs="+",
        choices=SUPPORTED_SCHEDULES,
        default=list(SUPPORTED_SCHEDULES),
    )
    parser.add_argument(
        "--trajectory-steps",
        type=positive_integer,
        default=12,
    )
    parser.add_argument(
        "--density-offset",
        type=positive_float,
        default=0.1,
    )
    parser.add_argument(
        "--coefficient-scale",
        type=positive_float,
        default=0.35,
    )
    parser.add_argument(
        "--perturbation-magnitude",
        type=positive_float,
        default=1e-5,
    )
    parser.add_argument(
        "--value-noise-magnitude",
        type=positive_float,
        default=1e-5,
        help=(
            "Relative Euclidean-norm perturbation applied to sampled "
            "function values before coefficient recovery."
        ),
    )
    parser.add_argument(
        "--recovery-oversampling-factor",
        type=positive_float,
        default=2.0,
        help=(
            "Number of coefficient-recovery points relative to the basis "
            "count."
        ),
    )
    parser.add_argument(
        "--static-obstacle-half-width",
        type=positive_float,
        default=0.15,
    )
    parser.add_argument(
        "--dynamic-min-half-width",
        type=positive_float,
        default=0.05,
    )
    parser.add_argument(
        "--dynamic-max-half-width",
        type=positive_float,
        default=0.22,
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260804,
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
    )

    arguments = parser.parse_args()

    if any(degree < 0 for degree in arguments.degrees):
        parser.error("All degrees must be non-negative.")

    if arguments.static_obstacle_half_width >= 0.5:
        parser.error(
            "--static-obstacle-half-width must be less than 0.5."
        )

    if not (
        0.0
        < arguments.dynamic_min_half_width
        <= arguments.dynamic_max_half_width
        < 0.5
    ):
        parser.error(
            "Dynamic obstacle half-widths must satisfy "
            "0 < minimum <= maximum < 0.5."
        )

    return arguments


def main() -> None:
    args = parse_arguments()

    start = time.perf_counter()
    records = run_benchmark(args)
    elapsed_seconds = time.perf_counter() - start

    write_records(records, args.results_path)

    configuration = {
        "dimensions": sorted(set(args.dimensions)),
        "source_degrees": sorted(set(args.degrees)),
        "density_degrees": [
            2 * degree for degree in sorted(set(args.degrees))
        ],
        "bases": list(BASIS_NAMES),
        "dtypes": list(dict.fromkeys(args.dtypes)),
        "trials": args.trials,
        "static_scenarios": [
            "convex_box",
            "box_with_obstacle",
        ],
        "dynamic_schedules": list(dict.fromkeys(args.schedules)),
        "trajectory_steps": args.trajectory_steps,
        "density_offset": args.density_offset,
        "coefficient_scale": args.coefficient_scale,
        "perturbation_magnitude": args.perturbation_magnitude,
        "value_noise_magnitude": args.value_noise_magnitude,
        "recovery_oversampling_factor": (
            args.recovery_oversampling_factor
        ),
        "static_obstacle_half_width": (
            args.static_obstacle_half_width
        ),
        "dynamic_min_half_width": (
            args.dynamic_min_half_width
        ),
        "dynamic_max_half_width": (
            args.dynamic_max_half_width
        ),
        "base_seed": args.base_seed,
        "results_path": str(args.results_path),
        "summary_path": str(args.summary_path),
    }

    summary = build_summary(
        records=records,
        configuration=configuration,
        elapsed_seconds=elapsed_seconds,
    )
    write_summary(summary, args.summary_path)

    print()
    print(
        f"Completed {len(records)} polytope records in "
        f"{elapsed_seconds:.3f} seconds."
    )
    print()
    print("Polytope benchmark summary")
    print("==========================")
    print(f"records: {summary['records']}")
    print(f"finite records: {summary['finite_records']}")
    print(
        "infinite condition numbers: "
        f"{summary['infinite_condition_numbers']}"
    )
    print(
        "rounded-away perturbations: "
        f"{summary['rounded_away_perturbations']}"
    )
    print(
        "records with sampled negative density values: "
        f"{summary['records_with_sampled_negative_density_values']}"
    )

    print()
    print("Median metrics by scenario, basis, and dtype")
    print_grouped_summary(
        summary["median_metrics_by_scenario_basis_dtype"]
    )

    maximum_error = summary["maximum_relative_integration_error"]
    print()
    print("Maximum relative integration error")
    print("==================================")
    for key in (
        "scenario",
        "schedule",
        "trajectory_step",
        "dimension",
        "source_polynomial_degree",
        "density_polynomial_degree",
        "trial",
        "basis",
        "dtype",
        "obstacle_half_width",
        "reference_integral",
        "computed_integral",
        "relative_integration_error",
        "basis_condition_number",
        "numerical_rank_fraction",
        "total_evaluation_ms",
    ):
        print(f"{key}: {maximum_error[key]}")

    maximum_amplification = summary[
        "maximum_coefficient_noise_amplification"
    ]
    print()
    print("Maximum function-value noise amplification")
    print("==========================================")
    for key in (
        "scenario",
        "schedule",
        "trajectory_step",
        "dimension",
        "source_polynomial_degree",
        "density_polynomial_degree",
        "trial",
        "basis",
        "dtype",
        "basis_condition_number",
        "numerical_rank_fraction",
        "function_value_noise_relative_norm",
        "coefficient_recovery_relative_error",
        "coefficient_noise_amplification",
        "recovered_integral_relative_error",
    ):
        print(f"{key}: {maximum_amplification[key]}")

    maximum_recovered_error = summary[
        "maximum_recovered_integral_relative_error"
    ]
    print()
    print("Maximum recovered-integral relative error")
    print("=========================================")
    for key in (
        "scenario",
        "schedule",
        "trajectory_step",
        "dimension",
        "source_polynomial_degree",
        "density_polynomial_degree",
        "trial",
        "basis",
        "dtype",
        "basis_condition_number",
        "numerical_rank_fraction",
        "coefficient_noise_amplification",
        "recovered_integral_relative_error",
    ):
        print(f"{key}: {maximum_recovered_error[key]}")

    print(f"Wrote {args.results_path}")
    print(f"Wrote {args.summary_path}")


if __name__ == "__main__":
    main()