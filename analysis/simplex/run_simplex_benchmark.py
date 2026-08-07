#!/usr/bin/env python3
"""
Run the controlled unit-simplex benchmark for Chapter 5.

The benchmark compares monomial, Legendre, and Chebyshev representations of
the same multivariate polynomial over the scaled unit simplex

    Δ_n(s) = {x in R^n : x_i >= 0 and sum_i x_i <= s}.

For every configuration, the script records:

- polynomial degree;
- simplex dimension;
- geometric scale;
- floating-point precision;
- basis;
- analytical reference integral;
- quadrature integral;
- absolute and relative integration error;
- basis-matrix condition number;
- coefficient-perturbation sensitivity;
- conversion, evaluation, integration, and total runtime.

The exact reference uses the monomial identity

    ∫_{Δ_n(s)} x^α dx
      = s^(n + |α|) * prod_i Γ(α_i + 1)
        / Γ(n + |α| + 1).

Numerical integration uses a tensor-product Gauss-Legendre rule combined with
a Duffy transformation from [0, 1]^n to the simplex. This keeps the geometry,
quadrature rule, and accumulation procedure fixed across basis choices.

Outputs
-------
results/simplex/unit_simplex_results.csv
results/simplex/unit_simplex_summary.json

Example
-------
Run from the repository root:

    python -m analysis.simplex.run_simplex_benchmark

A smaller smoke test:

    python -m analysis.simplex.run_simplex_benchmark \
        --degrees 0 1 2 3 \
        --dimensions 1 2 \
        --scales 1 \
        --dtypes float64 \
        --trials 2 \
        --quadrature-order 12

Notes
-----
1. Coefficient tensors use one axis per variable.
2. Only total-degree multi-indices are included in reported basis vectors.
3. Orthogonal bases are tensor-product Legendre or Chebyshev bases, with every
   physical coordinate x_i in [0, scale] mapped to [-1, 1].
4. Condition numbers are computed from sampled basis-evaluation matrices, not
   from raw coefficient magnitudes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.polynomial import Chebyshev, Legendre, Polynomial
from numpy.typing import NDArray


# =============================================================================
# Repository setup
# =============================================================================

THIS_FILE = Path(__file__).resolve()
REPOSITORY_ROOT = THIS_FILE.parents[2]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


# =============================================================================
# Constants
# =============================================================================

SUPPORTED_BASES = ("monomial", "legendre", "chebyshev")
SUPPORTED_DTYPES = {
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class BenchmarkConfiguration:
    """One shared benchmark configuration."""

    trial: int
    seed: int
    degree: int
    dimension: int
    scale: float
    dtype: str
    coefficient_scale: float
    perturbation_magnitude: float
    quadrature_order: int


@dataclass
class SimplexBenchmarkRecord:
    """One result for one basis and one benchmark configuration."""

    trial: int
    seed: int
    degree: int
    dimension: int
    scale: float
    dtype: str
    basis: str
    coefficient_scale: float
    perturbation_magnitude: float
    quadrature_order: int
    coefficient_count: int
    quadrature_point_count: int

    reference_integral: float
    computed_integral: float
    integration_absolute_error: float
    integration_relative_error: float

    condition_number: float
    log10_condition_number: float

    coefficient_norm: float
    perturbation_norm: float
    perturbed_integral: float
    perturbation_absolute_change: float
    perturbation_relative_change: float
    perturbation_sensitivity: float
    relative_perturbation_sensitivity: float

    conversion_runtime_seconds: float
    condition_runtime_seconds: float
    evaluation_runtime_seconds: float
    integration_runtime_seconds: float
    perturbation_runtime_seconds: float
    total_runtime_seconds: float

    finite: bool


@dataclass
class BenchmarkSummary:
    """Aggregate benchmark summary."""

    total_records: int
    finite_records: int
    non_finite_records: int
    maximum_integration_absolute_error: float
    maximum_integration_relative_error: float
    maximum_condition_number: float
    maximum_perturbation_sensitivity: float
    maximum_total_runtime_seconds: float


# =============================================================================
# Validation and parsing helpers
# =============================================================================

def validate_positive_finite(value: float, name: str) -> None:
    """Require a finite value greater than zero."""
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive; received {value}.")


def parse_dtype(value: str) -> np.dtype:
    """Parse a supported NumPy floating-point dtype."""
    try:
        return SUPPORTED_DTYPES[value]
    except KeyError as exc:
        allowed = ", ".join(SUPPORTED_DTYPES)
        raise argparse.ArgumentTypeError(
            f"dtype must be one of: {allowed}."
        ) from exc


def parse_basis(value: str) -> str:
    """Parse a supported basis name."""
    normalised = value.lower()

    if normalised not in SUPPORTED_BASES:
        allowed = ", ".join(SUPPORTED_BASES)
        raise argparse.ArgumentTypeError(
            f"basis must be one of: {allowed}."
        )

    return normalised


# =============================================================================
# Multi-index and coefficient generation
# =============================================================================

def total_degree_multiindices(
    dimension: int,
    degree: int,
) -> list[tuple[int, ...]]:
    """
    Return all non-negative multi-indices α with |α| <= degree.

    The ordering is deterministic: increasing total degree followed by
    lexicographic order.
    """
    if dimension <= 0:
        raise ValueError("dimension must be positive.")

    if degree < 0:
        raise ValueError("degree must be non-negative.")

    indices = [
        index
        for index in product(range(degree + 1), repeat=dimension)
        if sum(index) <= degree
    ]

    indices.sort(key=lambda index: (sum(index), index))
    return indices


def generate_monomial_tensor(
    *,
    rng: np.random.Generator,
    dimension: int,
    degree: int,
    coefficient_scale: float,
    dtype: np.dtype,
) -> tuple[NDArray[np.floating], list[tuple[int, ...]]]:
    """
    Generate a random total-degree polynomial in the monomial basis.

    Coefficients decay with total degree to prevent every high-degree case from
    being dominated solely by coefficient growth. Geometric scaling remains an
    explicit independent benchmark variable.
    """
    indices = total_degree_multiindices(dimension, degree)
    shape = (degree + 1,) * dimension
    coefficients = np.zeros(shape, dtype=dtype)

    for index in indices:
        total_degree = sum(index)
        decay = float(total_degree + 1)
        value = rng.normal(0.0, coefficient_scale) / decay
        coefficients[index] = dtype.type(value)

    # Keep the polynomial away from the identically-zero and nearly-zero cases.
    zero_index = (0,) * dimension
    coefficients[zero_index] = dtype.type(
        float(coefficients[zero_index]) + 1.0
    )

    return coefficients, indices


def extract_coefficients(
    coefficient_tensor: NDArray[np.floating],
    indices: Sequence[tuple[int, ...]],
    dtype: np.dtype,
) -> NDArray[np.floating]:
    """Extract tensor entries into a deterministic coefficient vector."""
    return np.asarray(
        [coefficient_tensor[index] for index in indices],
        dtype=dtype,
    )


def insert_coefficients(
    vector: NDArray[np.floating],
    indices: Sequence[tuple[int, ...]],
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> NDArray[np.floating]:
    """Insert a coefficient vector into a zero-filled coefficient tensor."""
    tensor = np.zeros(shape, dtype=dtype)

    for value, index in zip(vector, indices, strict=True):
        tensor[index] = value

    return tensor


# =============================================================================
# Basis conversion
# =============================================================================

def univariate_conversion_matrix(
    *,
    degree: int,
    basis: str,
    lower: float,
    upper: float,
    dtype: np.dtype,
) -> NDArray[np.floating]:
    """
    Build a matrix mapping physical-x monomial coefficients to a target basis.

    Matrix column j contains the target-basis coefficients representing x**j
    over the physical interval [lower, upper].
    """
    if basis == "monomial":
        return np.eye(degree + 1, dtype=dtype)

    if basis == "legendre":
        target_kind = Legendre
    elif basis == "chebyshev":
        target_kind = Chebyshev
    else:
        raise ValueError(f"Unsupported basis: {basis}")

    matrix = np.zeros((degree + 1, degree + 1), dtype=np.float64)

    for power in range(degree + 1):
        source = Polynomial.basis(power)
        converted = source.convert(
            kind=target_kind,
            domain=[lower, upper],
            window=[-1.0, 1.0],
        )

        converted_coefficients = np.asarray(
            converted.coef,
            dtype=np.float64,
        )

        matrix[: converted_coefficients.size, power] = (
            converted_coefficients
        )

    return np.asarray(matrix, dtype=dtype)


def apply_axis_conversion(
    coefficient_tensor: NDArray[np.floating],
    conversion_matrix: NDArray[np.floating],
    axis: int,
    dtype: np.dtype,
) -> NDArray[np.floating]:
    """
    Apply one univariate coefficient transformation along one tensor axis.

    If T maps source coefficients to target coefficients, the transformed axis
    satisfies

        output[..., i, ...] = sum_j T[i, j] input[..., j, ...].
    """
    transformed = np.tensordot(
        conversion_matrix,
        coefficient_tensor,
        axes=(1, axis),
    )

    transformed = np.moveaxis(transformed, 0, axis)
    return np.asarray(transformed, dtype=dtype)


def convert_monomial_tensor(
    *,
    monomial_tensor: NDArray[np.floating],
    basis: str,
    degree: int,
    dimension: int,
    scale: float,
    dtype: np.dtype,
) -> NDArray[np.floating]:
    """
    Convert a multivariate physical-coordinate monomial tensor to a basis.

    The multivariate transformation is separable because the target basis is a
    tensor product of one-dimensional bases.
    """
    if basis == "monomial":
        return np.asarray(monomial_tensor, dtype=dtype).copy()

    conversion_matrix = univariate_conversion_matrix(
        degree=degree,
        basis=basis,
        lower=0.0,
        upper=scale,
        dtype=dtype,
    )

    converted = np.asarray(monomial_tensor, dtype=dtype).copy()

    for axis in range(dimension):
        converted = apply_axis_conversion(
            coefficient_tensor=converted,
            conversion_matrix=conversion_matrix,
            axis=axis,
            dtype=dtype,
        )

    return converted


# =============================================================================
# Simplex reference integral
# =============================================================================

def monomial_simplex_integral(
    multiindex: Sequence[int],
    scale: float,
) -> np.longdouble:
    """
    Integrate x**α exactly over the scaled unit simplex.

    Uses logarithms of Gamma functions for range robustness.
    """
    dimension = len(multiindex)
    total_degree = int(sum(multiindex))

    log_value = (
        np.longdouble(dimension + total_degree) * np.log(
            np.longdouble(scale)
        )
    )

    for exponent in multiindex:
        log_value += np.longdouble(math.lgamma(exponent + 1.0))

    log_value -= np.longdouble(
        math.lgamma(dimension + total_degree + 1.0)
    )

    return np.exp(log_value, dtype=np.longdouble)


def exact_simplex_integral(
    monomial_tensor: NDArray[np.floating],
    indices: Sequence[tuple[int, ...]],
    scale: float,
) -> float:
    """Compute a high-precision analytical integral of a monomial tensor."""
    total = np.longdouble(0.0)

    for index in indices:
        coefficient = np.longdouble(monomial_tensor[index])
        total += coefficient * monomial_simplex_integral(index, scale)

    return float(total)


# =============================================================================
# Duffy-transformed Gauss-Legendre quadrature
# =============================================================================

def gauss_legendre_rule_01(
    order: int,
    dtype: np.dtype,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return Gauss-Legendre nodes and weights on [0, 1]."""
    if order <= 0:
        raise ValueError("quadrature order must be positive.")

    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    return (
        np.asarray(nodes, dtype=dtype),
        np.asarray(weights, dtype=dtype),
    )


def simplex_quadrature_rule(
    *,
    dimension: int,
    scale: float,
    order: int,
    dtype: np.dtype,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Construct a deterministic quadrature rule for the scaled unit simplex.

    Duffy transformation:
        x_1 = s u_1
        x_2 = s (1-u_1) u_2
        ...
        x_n = s prod_{j<n}(1-u_j) u_n

    Jacobian:
        s^n prod_{j=1}^{n-1} (1-u_j)^(n-j)
    """
    if dimension <= 0:
        raise ValueError("dimension must be positive.")

    validate_positive_finite(scale, "scale")

    nodes_1d, weights_1d = gauss_legendre_rule_01(order, dtype)

    grid_indices = np.asarray(
        list(product(range(order), repeat=dimension)),
        dtype=np.int64,
    )

    unit_coordinates = nodes_1d[grid_indices]
    tensor_weights = np.prod(
        weights_1d[grid_indices],
        axis=1,
        dtype=np.float64,
    ).astype(dtype)

    point_count = unit_coordinates.shape[0]
    points = np.empty((point_count, dimension), dtype=dtype)

    remaining = np.ones(point_count, dtype=dtype)

    for axis in range(dimension):
        points[:, axis] = (
            dtype.type(scale)
            * remaining
            * unit_coordinates[:, axis]
        )
        remaining = remaining * (dtype.type(1.0) - unit_coordinates[:, axis])

    jacobian = np.full(
        point_count,
        dtype.type(scale ** dimension),
        dtype=dtype,
    )

    for axis in range(dimension - 1):
        exponent = dimension - axis - 1
        jacobian *= (
            dtype.type(1.0) - unit_coordinates[:, axis]
        ) ** exponent

    weights = tensor_weights * jacobian
    return points, weights


# =============================================================================
# Basis evaluation
# =============================================================================

def scaled_coordinate(
    x: NDArray[np.floating],
    scale: float,
    dtype: np.dtype,
) -> NDArray[np.floating]:
    """Map x from [0, scale] to [-1, 1]."""
    return np.asarray(
        dtype.type(2.0) * x / dtype.type(scale) - dtype.type(1.0),
        dtype=dtype,
    )


def univariate_vandermonde(
    *,
    basis: str,
    x: NDArray[np.floating],
    degree: int,
    scale: float,
    dtype: np.dtype,
) -> NDArray[np.floating]:
    """Evaluate all one-dimensional basis functions through the given degree."""
    if basis == "monomial":
        values = np.polynomial.polynomial.polyvander(x, degree)
    elif basis == "legendre":
        values = np.polynomial.legendre.legvander(
            scaled_coordinate(x, scale, dtype),
            degree,
        )
    elif basis == "chebyshev":
        values = np.polynomial.chebyshev.chebvander(
            scaled_coordinate(x, scale, dtype),
            degree,
        )
    else:
        raise ValueError(f"Unsupported basis: {basis}")

    return np.asarray(values, dtype=dtype)


def multivariate_design_matrix(
    *,
    basis: str,
    points: NDArray[np.floating],
    indices: Sequence[tuple[int, ...]],
    degree: int,
    scale: float,
    dtype: np.dtype,
) -> NDArray[np.floating]:
    """
    Build the total-degree basis-evaluation matrix.

    Row r and column k contain the basis function indexed by indices[k]
    evaluated at points[r].
    """
    point_count, dimension = points.shape
    column_count = len(indices)

    axis_values = [
        univariate_vandermonde(
            basis=basis,
            x=points[:, axis],
            degree=degree,
            scale=scale,
            dtype=dtype,
        )
        for axis in range(dimension)
    ]

    matrix = np.ones((point_count, column_count), dtype=dtype)

    for column, index in enumerate(indices):
        values = np.ones(point_count, dtype=dtype)

        for axis, exponent in enumerate(index):
            values *= axis_values[axis][:, exponent]

        matrix[:, column] = values

    return matrix


def evaluate_from_design_matrix(
    design_matrix: NDArray[np.floating],
    coefficients: NDArray[np.floating],
    dtype: np.dtype,
) -> NDArray[np.floating]:
    """Evaluate a coefficient vector using a precomputed design matrix."""
    return np.asarray(design_matrix @ coefficients, dtype=dtype)


# =============================================================================
# Condition-number sampling
# =============================================================================

def sample_uniform_simplex(
    *,
    rng: np.random.Generator,
    point_count: int,
    dimension: int,
    scale: float,
    dtype: np.dtype,
) -> NDArray[np.floating]:
    """
    Draw points uniformly from the scaled unit simplex.

    Exponential spacings provide a Dirichlet(1,...,1) sample. The final
    barycentric coordinate is dropped.
    """
    exponential = rng.exponential(
        scale=1.0,
        size=(point_count, dimension + 1),
    )

    barycentric = exponential / np.sum(
        exponential,
        axis=1,
        keepdims=True,
    )

    return np.asarray(
        scale * barycentric[:, :dimension],
        dtype=dtype,
    )


def estimate_condition_number(
    *,
    basis: str,
    indices: Sequence[tuple[int, ...]],
    degree: int,
    dimension: int,
    scale: float,
    dtype: np.dtype,
    rng: np.random.Generator,
    sample_multiplier: int,
    maximum_samples: int,
) -> float:
    """
    Estimate the 2-norm condition number of a sampled basis matrix.

    Columns are normalised before the SVD so the metric focuses on near-linear
    dependence rather than trivial differences in column magnitude.
    """
    coefficient_count = len(indices)
    sample_count = max(
        coefficient_count,
        min(maximum_samples, sample_multiplier * coefficient_count),
    )

    points = sample_uniform_simplex(
        rng=rng,
        point_count=sample_count,
        dimension=dimension,
        scale=scale,
        dtype=dtype,
    )

    matrix = multivariate_design_matrix(
        basis=basis,
        points=points,
        indices=indices,
        degree=degree,
        scale=scale,
        dtype=dtype,
    )

    matrix64 = np.asarray(matrix, dtype=np.float64)
    column_norms = np.linalg.norm(matrix64, axis=0)
    safe_norms = np.where(column_norms > 0.0, column_norms, 1.0)
    normalised = matrix64 / safe_norms

    singular_values = np.linalg.svd(
        normalised,
        compute_uv=False,
        full_matrices=False,
    )

    if singular_values.size == 0 or singular_values[-1] == 0.0:
        return float("inf")

    return float(singular_values[0] / singular_values[-1])


# =============================================================================
# Numerical metrics
# =============================================================================

def safe_relative_error(
    estimate: float,
    reference: float,
    floor: float = 1.0e-30,
) -> float:
    """Compute absolute relative error with a safe denominator."""
    denominator = max(abs(reference), floor)
    return abs(estimate - reference) / denominator


def integrate_values(
    values: NDArray[np.floating],
    weights: NDArray[np.floating],
    dtype: np.dtype,
) -> float:
    """
    Accumulate a quadrature integral in the requested precision.

    np.sum is used deliberately so accumulation remains part of the measured
    finite-precision pipeline.
    """
    products = np.asarray(values * weights, dtype=dtype)
    return float(np.sum(products, dtype=dtype))


def perturb_coefficients(
    *,
    coefficients: NDArray[np.floating],
    magnitude: float,
    rng: np.random.Generator,
    dtype: np.dtype,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Add a random coefficient perturbation with controlled relative 2-norm.

    The perturbation satisfies approximately

        ||δ||_2 = magnitude * max(||c||_2, 1).
    """
    direction = rng.normal(size=coefficients.size)
    direction_norm = np.linalg.norm(direction)

    if direction_norm == 0.0:
        direction[0] = 1.0
        direction_norm = 1.0

    direction = direction / direction_norm

    coefficient_norm = np.linalg.norm(
        np.asarray(coefficients, dtype=np.float64)
    )

    target_norm = magnitude * max(coefficient_norm, 1.0)
    perturbation = np.asarray(target_norm * direction, dtype=dtype)
    perturbed = np.asarray(coefficients + perturbation, dtype=dtype)

    return perturbed, perturbation


# =============================================================================
# One benchmark case
# =============================================================================

def run_basis_case(
    *,
    configuration: BenchmarkConfiguration,
    basis: str,
    monomial_tensor: NDArray[np.floating],
    indices: Sequence[tuple[int, ...]],
    reference_integral: float,
    quadrature_points: NDArray[np.floating],
    quadrature_weights: NDArray[np.floating],
    condition_rng: np.random.Generator,
    perturbation_rng: np.random.Generator,
    condition_sample_multiplier: int,
    maximum_condition_samples: int,
) -> SimplexBenchmarkRecord:
    """Run one basis for one shared benchmark configuration."""
    dtype = SUPPORTED_DTYPES[configuration.dtype]
    start_total = time.perf_counter()

    start = time.perf_counter()
    basis_tensor = convert_monomial_tensor(
        monomial_tensor=monomial_tensor,
        basis=basis,
        degree=configuration.degree,
        dimension=configuration.dimension,
        scale=configuration.scale,
        dtype=dtype,
    )
    conversion_runtime = time.perf_counter() - start

    coefficients = extract_coefficients(
        basis_tensor,
        indices,
        dtype,
    )

    start = time.perf_counter()
    condition_number = estimate_condition_number(
        basis=basis,
        indices=indices,
        degree=configuration.degree,
        dimension=configuration.dimension,
        scale=configuration.scale,
        dtype=dtype,
        rng=condition_rng,
        sample_multiplier=condition_sample_multiplier,
        maximum_samples=maximum_condition_samples,
    )
    condition_runtime = time.perf_counter() - start

    start = time.perf_counter()
    quadrature_matrix = multivariate_design_matrix(
        basis=basis,
        points=quadrature_points,
        indices=indices,
        degree=configuration.degree,
        scale=configuration.scale,
        dtype=dtype,
    )
    values = evaluate_from_design_matrix(
        quadrature_matrix,
        coefficients,
        dtype,
    )
    evaluation_runtime = time.perf_counter() - start

    start = time.perf_counter()
    computed_integral = integrate_values(
        values,
        quadrature_weights,
        dtype,
    )
    integration_runtime = time.perf_counter() - start

    integration_absolute_error = abs(
        computed_integral - reference_integral
    )
    integration_relative_error = safe_relative_error(
        computed_integral,
        reference_integral,
    )

    start = time.perf_counter()
    perturbed_coefficients, perturbation = perturb_coefficients(
        coefficients=coefficients,
        magnitude=configuration.perturbation_magnitude,
        rng=perturbation_rng,
        dtype=dtype,
    )

    perturbed_values = evaluate_from_design_matrix(
        quadrature_matrix,
        perturbed_coefficients,
        dtype,
    )

    perturbed_integral = integrate_values(
        perturbed_values,
        quadrature_weights,
        dtype,
    )
    perturbation_runtime = time.perf_counter() - start

    coefficient_norm = float(
        np.linalg.norm(np.asarray(coefficients, dtype=np.float64))
    )
    perturbation_norm = float(
        np.linalg.norm(np.asarray(perturbation, dtype=np.float64))
    )

    perturbation_absolute_change = abs(
        perturbed_integral - computed_integral
    )

    perturbation_relative_change = (
        perturbation_absolute_change
        / max(abs(computed_integral), 1.0e-30)
    )

    perturbation_sensitivity = (
        perturbation_absolute_change
        / max(perturbation_norm, 1.0e-30)
    )

    relative_input_change = (
        perturbation_norm
        / max(coefficient_norm, 1.0e-30)
    )

    relative_perturbation_sensitivity = (
        perturbation_relative_change
        / max(relative_input_change, 1.0e-30)
    )

    total_runtime = time.perf_counter() - start_total

    finite_values = (
        reference_integral,
        computed_integral,
        integration_absolute_error,
        integration_relative_error,
        condition_number,
        coefficient_norm,
        perturbation_norm,
        perturbed_integral,
        perturbation_absolute_change,
        perturbation_relative_change,
        perturbation_sensitivity,
        relative_perturbation_sensitivity,
        conversion_runtime,
        condition_runtime,
        evaluation_runtime,
        integration_runtime,
        perturbation_runtime,
        total_runtime,
    )

    finite = all(np.isfinite(value) for value in finite_values)

    return SimplexBenchmarkRecord(
        trial=configuration.trial,
        seed=configuration.seed,
        degree=configuration.degree,
        dimension=configuration.dimension,
        scale=configuration.scale,
        dtype=configuration.dtype,
        basis=basis,
        coefficient_scale=configuration.coefficient_scale,
        perturbation_magnitude=configuration.perturbation_magnitude,
        quadrature_order=configuration.quadrature_order,
        coefficient_count=len(indices),
        quadrature_point_count=quadrature_points.shape[0],
        reference_integral=reference_integral,
        computed_integral=computed_integral,
        integration_absolute_error=integration_absolute_error,
        integration_relative_error=integration_relative_error,
        condition_number=condition_number,
        log10_condition_number=(
            math.log10(condition_number)
            if condition_number > 0.0 and np.isfinite(condition_number)
            else float("inf")
        ),
        coefficient_norm=coefficient_norm,
        perturbation_norm=perturbation_norm,
        perturbed_integral=perturbed_integral,
        perturbation_absolute_change=perturbation_absolute_change,
        perturbation_relative_change=perturbation_relative_change,
        perturbation_sensitivity=perturbation_sensitivity,
        relative_perturbation_sensitivity=(
            relative_perturbation_sensitivity
        ),
        conversion_runtime_seconds=conversion_runtime,
        condition_runtime_seconds=condition_runtime,
        evaluation_runtime_seconds=evaluation_runtime,
        integration_runtime_seconds=integration_runtime,
        perturbation_runtime_seconds=perturbation_runtime,
        total_runtime_seconds=total_runtime,
        finite=finite,
    )


# =============================================================================
# Complete experiment
# =============================================================================

def run_benchmark(
    *,
    degrees: Sequence[int],
    dimensions: Sequence[int],
    scales: Sequence[float],
    dtypes: Sequence[np.dtype],
    bases: Sequence[str],
    trials: int,
    seed: int,
    coefficient_scale: float,
    perturbation_magnitude: float,
    quadrature_order: int,
    condition_sample_multiplier: int,
    maximum_condition_samples: int,
) -> list[SimplexBenchmarkRecord]:
    """Run all requested unit-simplex benchmark configurations."""
    if trials <= 0:
        raise ValueError("trials must be positive.")

    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive.")

    if condition_sample_multiplier <= 0:
        raise ValueError("condition_sample_multiplier must be positive.")

    if maximum_condition_samples <= 0:
        raise ValueError("maximum_condition_samples must be positive.")

    validate_positive_finite(coefficient_scale, "coefficient_scale")
    validate_positive_finite(
        perturbation_magnitude,
        "perturbation_magnitude",
    )

    for degree in degrees:
        if degree < 0:
            raise ValueError("degrees must be non-negative.")

    for dimension in dimensions:
        if dimension <= 0:
            raise ValueError("dimensions must be positive.")

    for scale in scales:
        validate_positive_finite(scale, "scale")

    records: list[SimplexBenchmarkRecord] = []

    for dtype_index, dtype_value in enumerate(dtypes):
        dtype = np.dtype(dtype_value)
        dtype_name = dtype.name

        for dimension_index, dimension in enumerate(dimensions):
            for scale_index, scale in enumerate(scales):
                quadrature_points, quadrature_weights = (
                    simplex_quadrature_rule(
                        dimension=dimension,
                        scale=scale,
                        order=quadrature_order,
                        dtype=dtype,
                    )
                )

                for degree in degrees:
                    indices = total_degree_multiindices(
                        dimension,
                        degree,
                    )

                    for trial in range(trials):
                        trial_seed = (
                            seed
                            + 10_000_000 * dtype_index
                            + 1_000_000 * dimension_index
                            + 100_000 * scale_index
                            + 1_000 * degree
                            + trial
                        )

                        coefficient_rng = np.random.default_rng(
                            trial_seed
                        )

                        monomial_tensor, generated_indices = (
                            generate_monomial_tensor(
                                rng=coefficient_rng,
                                dimension=dimension,
                                degree=degree,
                                coefficient_scale=coefficient_scale,
                                dtype=dtype,
                            )
                        )

                        if generated_indices != indices:
                            raise RuntimeError(
                                "Internal multi-index ordering mismatch."
                            )

                        reference_integral = exact_simplex_integral(
                            monomial_tensor,
                            indices,
                            scale,
                        )

                        configuration = BenchmarkConfiguration(
                            trial=trial,
                            seed=trial_seed,
                            degree=degree,
                            dimension=dimension,
                            scale=scale,
                            dtype=dtype_name,
                            coefficient_scale=coefficient_scale,
                            perturbation_magnitude=(
                                perturbation_magnitude
                            ),
                            quadrature_order=quadrature_order,
                        )

                        for basis_index, basis in enumerate(bases):
                            condition_rng = np.random.default_rng(
                                trial_seed
                                + 100_000_000
                                + 10_000 * basis_index
                            )

                            perturbation_rng = np.random.default_rng(
                                trial_seed
                                + 200_000_000
                                + 10_000 * basis_index
                            )

                            record = run_basis_case(
                                configuration=configuration,
                                basis=basis,
                                monomial_tensor=monomial_tensor,
                                indices=indices,
                                reference_integral=reference_integral,
                                quadrature_points=quadrature_points,
                                quadrature_weights=quadrature_weights,
                                condition_rng=condition_rng,
                                perturbation_rng=perturbation_rng,
                                condition_sample_multiplier=(
                                    condition_sample_multiplier
                                ),
                                maximum_condition_samples=(
                                    maximum_condition_samples
                                ),
                            )

                            records.append(record)

    return records


# =============================================================================
# Reporting
# =============================================================================

def create_summary(
    records: Sequence[SimplexBenchmarkRecord],
) -> BenchmarkSummary:
    """Create an aggregate benchmark summary."""
    if not records:
        return BenchmarkSummary(
            total_records=0,
            finite_records=0,
            non_finite_records=0,
            maximum_integration_absolute_error=0.0,
            maximum_integration_relative_error=0.0,
            maximum_condition_number=0.0,
            maximum_perturbation_sensitivity=0.0,
            maximum_total_runtime_seconds=0.0,
        )

    finite_records = sum(record.finite for record in records)

    return BenchmarkSummary(
        total_records=len(records),
        finite_records=finite_records,
        non_finite_records=len(records) - finite_records,
        maximum_integration_absolute_error=max(
            record.integration_absolute_error
            for record in records
        ),
        maximum_integration_relative_error=max(
            record.integration_relative_error
            for record in records
        ),
        maximum_condition_number=max(
            record.condition_number
            for record in records
        ),
        maximum_perturbation_sensitivity=max(
            record.perturbation_sensitivity
            for record in records
        ),
        maximum_total_runtime_seconds=max(
            record.total_runtime_seconds
            for record in records
        ),
    )


def write_csv(
    records: Sequence[SimplexBenchmarkRecord],
    output_path: Path,
) -> None:
    """Write detailed records to CSV."""
    if not records:
        raise ValueError("Cannot write an empty benchmark result set.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))


def environment_information() -> dict[str, str | int | None]:
    """Collect reproducibility metadata without requiring extra packages."""
    return {
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "cpu_count": os.cpu_count(),
    }


def write_json(
    *,
    summary: BenchmarkSummary,
    configuration: dict,
    output_path: Path,
) -> None:
    """Write summary, configuration, and environment metadata to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": asdict(summary),
        "configuration": configuration,
        "environment": environment_information(),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def grouped_records(
    records: Iterable[SimplexBenchmarkRecord],
) -> dict[
    tuple[str, int, float, str],
    list[SimplexBenchmarkRecord],
]:
    """Group records for compact terminal output."""
    groups: dict[
        tuple[str, int, float, str],
        list[SimplexBenchmarkRecord],
    ] = {}

    for record in records:
        key = (
            record.dtype,
            record.dimension,
            record.scale,
            record.basis,
        )
        groups.setdefault(key, []).append(record)

    return groups


def print_table(
    records: Sequence[SimplexBenchmarkRecord],
) -> None:
    """Print a compact aggregate table."""
    groups = grouped_records(records)

    header = (
        f"{'dtype':<9}"
        f"{'dim':>5}"
        f"{'scale':>10}"
        f"{'basis':>12}"
        f"{'max rel err':>16}"
        f"{'max cond':>16}"
        f"{'max pert sens':>16}"
        f"{'mean time(s)':>16}"
    )

    print()
    print(header)
    print("-" * len(header))

    for key in sorted(groups):
        dtype_name, dimension, scale, basis = key
        group = groups[key]

        max_relative_error = max(
            record.integration_relative_error
            for record in group
        )
        max_condition = max(
            record.condition_number
            for record in group
        )
        max_sensitivity = max(
            record.perturbation_sensitivity
            for record in group
        )
        mean_runtime = float(
            np.mean(
                [
                    record.total_runtime_seconds
                    for record in group
                ]
            )
        )

        print(
            f"{dtype_name:<9}"
            f"{dimension:>5d}"
            f"{scale:>10.3g}"
            f"{basis:>12}"
            f"{max_relative_error:>16.6e}"
            f"{max_condition:>16.6e}"
            f"{max_sensitivity:>16.6e}"
            f"{mean_runtime:>16.6e}"
        )


def print_non_finite_records(
    records: Sequence[SimplexBenchmarkRecord],
    maximum_to_show: int = 20,
) -> None:
    """Print configurations that produced a non-finite quantity."""
    failures = [record for record in records if not record.finite]

    if not failures:
        return

    print("\nNon-finite benchmark records:")

    for record in failures[:maximum_to_show]:
        print(
            "  "
            f"basis={record.basis}, "
            f"dtype={record.dtype}, "
            f"dimension={record.dimension}, "
            f"degree={record.degree}, "
            f"scale={record.scale:g}, "
            f"trial={record.trial}, "
            f"integral={record.computed_integral:.6e}, "
            f"condition={record.condition_number:.6e}"
        )

    if len(failures) > maximum_to_show:
        print(
            f"  ... and {len(failures) - maximum_to_show} "
            "additional non-finite records."
        )


# =============================================================================
# CLI
# =============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark monomial, Legendre, and Chebyshev representations "
            "over scaled unit simplices."
        )
    )

    parser.add_argument(
        "--degrees",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 5, 8, 10],
        help="Total polynomial degrees to benchmark.",
    )

    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Simplex dimensions to benchmark.",
    )

    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 10.0],
        help="Geometric simplex scales.",
    )

    parser.add_argument(
        "--dtypes",
        nargs="+",
        type=parse_dtype,
        default=[
            np.dtype(np.float64),
            np.dtype(np.float32),
        ],
        help="Floating-point types: float32 and/or float64.",
    )

    parser.add_argument(
        "--bases",
        nargs="+",
        type=parse_basis,
        default=list(SUPPORTED_BASES),
        help="Bases: monomial, legendre, chebyshev.",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Random coefficient trials per shared configuration.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )

    parser.add_argument(
        "--coefficient-scale",
        type=float,
        default=1.0,
        help="Standard deviation used for random coefficients.",
    )

    parser.add_argument(
        "--perturbation-magnitude",
        type=float,
        default=1.0e-7,
        help="Relative 2-norm of coefficient perturbations.",
    )

    parser.add_argument(
        "--quadrature-order",
        type=int,
        default=16,
        help=(
            "One-dimensional Gauss-Legendre order used in each Duffy "
            "coordinate."
        ),
    )

    parser.add_argument(
        "--condition-sample-multiplier",
        type=int,
        default=2,
        help=(
            "Condition matrix rows as a multiple of coefficient count, "
            "subject to --maximum-condition-samples."
        ),
    )

    parser.add_argument(
        "--maximum-condition-samples",
        type=int,
        default=2000,
        help="Maximum random simplex samples used per condition estimate.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/simplex"),
        help="Directory for structured benchmark outputs.",
    )

    parser.add_argument(
        "--allow-non-finite",
        action="store_true",
        help="Return exit code zero even if non-finite results occur.",
    )

    return parser


def main() -> int:
    """Run the complete unit-simplex benchmark."""
    parser = build_argument_parser()
    arguments = parser.parse_args()

    dtype_names = [
        np.dtype(dtype).name
        for dtype in arguments.dtypes
    ]

    print("Unit-simplex benchmark")
    print("----------------------")
    print(f"Degrees:              {arguments.degrees}")
    print(f"Dimensions:           {arguments.dimensions}")
    print(f"Scales:               {arguments.scales}")
    print(f"Dtypes:               {dtype_names}")
    print(f"Bases:                {arguments.bases}")
    print(f"Trials:               {arguments.trials}")
    print(f"Quadrature order:     {arguments.quadrature_order}")
    print(
        "Perturbation magnitude: "
        f"{arguments.perturbation_magnitude:.3e}"
    )

    records = run_benchmark(
        degrees=arguments.degrees,
        dimensions=arguments.dimensions,
        scales=arguments.scales,
        dtypes=arguments.dtypes,
        bases=arguments.bases,
        trials=arguments.trials,
        seed=arguments.seed,
        coefficient_scale=arguments.coefficient_scale,
        perturbation_magnitude=arguments.perturbation_magnitude,
        quadrature_order=arguments.quadrature_order,
        condition_sample_multiplier=(
            arguments.condition_sample_multiplier
        ),
        maximum_condition_samples=(
            arguments.maximum_condition_samples
        ),
    )

    summary = create_summary(records)

    output_directory = arguments.output_directory.resolve()
    csv_path = output_directory / "unit_simplex_results.csv"
    json_path = output_directory / "unit_simplex_summary.json"

    configuration = {
        "degrees": arguments.degrees,
        "dimensions": arguments.dimensions,
        "scales": arguments.scales,
        "dtypes": dtype_names,
        "bases": arguments.bases,
        "trials": arguments.trials,
        "seed": arguments.seed,
        "coefficient_scale": arguments.coefficient_scale,
        "perturbation_magnitude": arguments.perturbation_magnitude,
        "quadrature_order": arguments.quadrature_order,
        "condition_sample_multiplier": (
            arguments.condition_sample_multiplier
        ),
        "maximum_condition_samples": (
            arguments.maximum_condition_samples
        ),
        "simplex_definition": (
            "x_i >= 0 and sum_i x_i <= scale"
        ),
        "reference_method": (
            "analytical monomial integral using Gamma identities"
        ),
        "integration_method": (
            "tensor Gauss-Legendre quadrature under Duffy transformation"
        ),
        "runtime_method": "time.perf_counter",
    }

    write_csv(records, csv_path)
    write_json(
        summary=summary,
        configuration=configuration,
        output_path=json_path,
    )

    print_table(records)
    print_non_finite_records(records)

    print("\nBenchmark summary")
    print("-----------------")
    print(f"Total records:             {summary.total_records}")
    print(f"Finite records:            {summary.finite_records}")
    print(f"Non-finite records:        {summary.non_finite_records}")
    print(
        "Maximum relative error:    "
        f"{summary.maximum_integration_relative_error:.6e}"
    )
    print(
        "Maximum condition number:  "
        f"{summary.maximum_condition_number:.6e}"
    )
    print(
        "Maximum perturbation sens: "
        f"{summary.maximum_perturbation_sensitivity:.6e}"
    )
    print(
        "Maximum runtime (seconds): "
        f"{summary.maximum_total_runtime_seconds:.6e}"
    )
    print(f"\nDetailed CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")

    if summary.non_finite_records == 0:
        print("\nRESULT: UNIT-SIMPLEX BENCHMARK COMPLETED")
        return 0

    print("\nRESULT: BENCHMARK PRODUCED NON-FINITE VALUES")

    if arguments.allow_non_finite:
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
