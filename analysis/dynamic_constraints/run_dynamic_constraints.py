#!/usr/bin/env python3
"""
Chapter 5 dynamic-constraint benchmark.

This experiment measures how monomial, Legendre, and Chebyshev polynomial
representations behave when the feasible integration region changes for every
sample or time step.

The changing region is a scaled unit simplex

    Delta_d(c_t) = {x in R^d : x_i >= 0 and sum_i x_i <= c_t},

where c_t follows a deterministic schedule. The underlying polynomial remains
fixed throughout a trajectory, so only the constraint changes. This models the
per-sample/per-trajectory constraint setting discussed in the dissertation.

The script reuses the validated numerical infrastructure from

    analysis.simplex.run_simplex_benchmark

and therefore expects that file to exist in the repository.

Measured quantities
-------------------
For every trial, degree, dimension, floating-point type, basis, schedule, and
time step, the benchmark records:

* exact analytical reference integral;
* numerical quadrature integral;
* absolute and relative integration error;
* sampled basis-matrix condition number;
* coefficient-perturbation sensitivity;
* constraint-update time;
* quadrature construction time;
* condition-estimation time;
* evaluation time;
* integration time;
* total per-step runtime;
* one-time basis conversion time;
* cumulative trajectory runtime.

The same polynomial, dynamic constraint sequence, quadrature rule, random
condition points, and perturbation direction are shared across bases wherever
possible.

Outputs
-------
results/dynamic_constraints/dynamic_constraints_results.csv
results/dynamic_constraints/dynamic_constraints_summary.json

Example smoke test
------------------
python -m analysis.dynamic_constraints.run_dynamic_constraints \
    --degrees 0 1 2 3 \
    --dimensions 1 2 \
    --schedules shrink oscillate \
    --steps 7 \
    --dtypes float64 \
    --trials 2 \
    --quadrature-order 12

Full default experiment
-----------------------
python -m analysis.dynamic_constraints.run_dynamic_constraints
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
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Repository imports
# =============================================================================

THIS_FILE = Path(__file__).resolve()
REPOSITORY_ROOT = THIS_FILE.parents[2]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from analysis.simplex.run_simplex_benchmark import (
        SUPPORTED_BASES,
        SUPPORTED_DTYPES,
        convert_monomial_tensor,
        exact_simplex_integral,
        extract_coefficients,
        generate_monomial_tensor,
        integrate_values,
        multivariate_design_matrix,
        safe_relative_error,
        simplex_quadrature_rule,
        total_degree_multiindices,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import analysis.simplex.run_simplex_benchmark. "
        "Place run_simplex_benchmark.py in analysis/simplex/ and ensure "
        "analysis/simplex/__init__.py exists."
    ) from exc


SUPPORTED_SCHEDULES = (
    "shrink",
    "expand",
    "oscillate",
    "pulse",
    "random_walk",
)


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class TrajectoryConfiguration:
    """Configuration shared by every step of one dynamic trajectory."""

    trial: int
    seed: int
    degree: int
    dimension: int
    dtype: str
    schedule: str
    steps: int
    minimum_constraint_scale: float
    maximum_constraint_scale: float
    basis_domain_scale: float
    coefficient_scale: float
    perturbation_magnitude: float
    quadrature_order: int


@dataclass
class DynamicConstraintRecord:
    """One result for one basis at one trajectory step."""

    trial: int
    seed: int
    degree: int
    dimension: int
    dtype: str
    basis: str
    schedule: str
    step: int
    steps: int
    normalised_time: float

    previous_constraint_scale: float
    constraint_scale: float
    constraint_delta: float
    relative_constraint_delta: float
    simplex_volume: float
    relative_simplex_volume: float
    basis_domain_scale: float

    coefficient_scale: float
    perturbation_magnitude: float
    quadrature_order: int
    coefficient_count: int
    quadrature_point_count: int
    condition_sample_count: int

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

    basis_conversion_runtime_seconds: float
    constraint_update_runtime_seconds: float
    quadrature_runtime_seconds: float
    condition_runtime_seconds: float
    evaluation_runtime_seconds: float
    integration_runtime_seconds: float
    perturbation_runtime_seconds: float
    step_runtime_seconds: float
    cumulative_trajectory_runtime_seconds: float

    finite: bool


@dataclass
class DynamicConstraintSummary:
    """Aggregate summary for one benchmark run."""

    total_records: int
    finite_records: int
    non_finite_records: int
    trajectory_count: int
    maximum_integration_absolute_error: float
    maximum_integration_relative_error: float
    maximum_condition_number: float
    maximum_perturbation_sensitivity: float
    maximum_step_runtime_seconds: float
    maximum_cumulative_trajectory_runtime_seconds: float


# =============================================================================
# CLI parsers and validation
# =============================================================================

def parse_dtype(value: str) -> np.dtype:
    """Parse a supported NumPy floating-point type."""
    try:
        return SUPPORTED_DTYPES[value]
    except KeyError as exc:
        allowed = ", ".join(SUPPORTED_DTYPES)
        raise argparse.ArgumentTypeError(
            f"dtype must be one of: {allowed}"
        ) from exc


def parse_basis(value: str) -> str:
    """Parse a supported polynomial basis."""
    normalised = value.lower()

    if normalised not in SUPPORTED_BASES:
        allowed = ", ".join(SUPPORTED_BASES)
        raise argparse.ArgumentTypeError(
            f"basis must be one of: {allowed}"
        )

    return normalised


def parse_schedule(value: str) -> str:
    """Parse a supported dynamic-constraint schedule."""
    normalised = value.lower()

    if normalised not in SUPPORTED_SCHEDULES:
        allowed = ", ".join(SUPPORTED_SCHEDULES)
        raise argparse.ArgumentTypeError(
            f"schedule must be one of: {allowed}"
        )

    return normalised


def require_positive(value: float, name: str) -> None:
    """Require a finite positive value."""
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{name} must be finite and positive; received {value}."
        )


# =============================================================================
# Constraint schedules
# =============================================================================

def linear_schedule(
    start: float,
    end: float,
    steps: int,
) -> NDArray[np.float64]:
    """Return an inclusive linear schedule."""
    return np.linspace(start, end, steps, dtype=np.float64)


def oscillating_schedule(
    minimum: float,
    maximum: float,
    steps: int,
) -> NDArray[np.float64]:
    """Start at the maximum, reach the minimum, then return."""
    if steps == 1:
        return np.asarray([maximum], dtype=np.float64)

    phase = np.linspace(0.0, 2.0 * np.pi, steps, dtype=np.float64)
    midpoint = 0.5 * (minimum + maximum)
    amplitude = 0.5 * (maximum - minimum)
    values = midpoint + amplitude * np.cos(phase)
    values[0] = maximum
    values[-1] = maximum
    return values


def pulse_schedule(
    minimum: float,
    maximum: float,
    steps: int,
) -> NDArray[np.float64]:
    """Alternate smoothly between large and small feasible regions."""
    if steps == 1:
        return np.asarray([maximum], dtype=np.float64)

    values = np.empty(steps, dtype=np.float64)
    block = max(1, steps // 4)

    for step in range(steps):
        phase = (step // block) % 2
        values[step] = maximum if phase == 0 else minimum

    values[0] = maximum
    return values


def random_walk_schedule(
    *,
    minimum: float,
    maximum: float,
    steps: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Produce a reproducible bounded random walk.

    Reflection is used at the boundaries so the schedule does not accumulate
    artificial clipping plateaus.
    """
    if steps == 1:
        return np.asarray([maximum], dtype=np.float64)

    values = np.empty(steps, dtype=np.float64)
    values[0] = maximum
    standard_step = 0.12 * (maximum - minimum)

    for step in range(1, steps):
        proposal = values[step - 1] + rng.normal(0.0, standard_step)

        while proposal < minimum or proposal > maximum:
            if proposal < minimum:
                proposal = minimum + (minimum - proposal)
            if proposal > maximum:
                proposal = maximum - (proposal - maximum)

        values[step] = proposal

    return values


def generate_constraint_schedule(
    *,
    schedule: str,
    minimum: float,
    maximum: float,
    steps: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate a complete deterministic or seeded constraint trajectory."""
    if steps <= 0:
        raise ValueError("steps must be positive.")

    require_positive(minimum, "minimum constraint scale")
    require_positive(maximum, "maximum constraint scale")

    if minimum > maximum:
        raise ValueError(
            "minimum constraint scale cannot exceed maximum constraint scale."
        )

    if schedule == "shrink":
        values = linear_schedule(maximum, minimum, steps)
    elif schedule == "expand":
        values = linear_schedule(minimum, maximum, steps)
    elif schedule == "oscillate":
        values = oscillating_schedule(minimum, maximum, steps)
    elif schedule == "pulse":
        values = pulse_schedule(minimum, maximum, steps)
    elif schedule == "random_walk":
        values = random_walk_schedule(
            minimum=minimum,
            maximum=maximum,
            steps=steps,
            rng=rng,
        )
    else:
        raise ValueError(f"Unsupported schedule: {schedule}")

    if not np.all(np.isfinite(values)):
        raise RuntimeError("Constraint schedule contains non-finite values.")

    if np.any(values < minimum) or np.any(values > maximum):
        raise RuntimeError("Constraint schedule escaped its requested bounds.")

    return values


# =============================================================================
# Geometry and sampling
# =============================================================================

def simplex_volume(scale: float, dimension: int) -> float:
    """Return the volume of {x >= 0, sum(x) <= scale}."""
    return float(scale ** dimension / math.factorial(dimension))


def sample_uniform_simplex_from_exponentials(
    *,
    exponential_samples: NDArray[np.float64],
    scale: float,
    dtype: np.dtype,
) -> NDArray[np.floating]:
    """
    Map shared exponential samples to uniform points in the current simplex.

    Reusing the same base samples across bases ensures identical condition
    matrices apart from basis evaluation.
    """
    barycentric = exponential_samples / np.sum(
        exponential_samples,
        axis=1,
        keepdims=True,
    )

    dimension = exponential_samples.shape[1] - 1
    points = scale * barycentric[:, :dimension]
    return np.asarray(points, dtype=dtype)


def condition_sample_count(
    coefficient_count: int,
    multiplier: int,
    maximum_samples: int,
) -> int:
    """Choose the number of rows used in a sampled condition matrix."""
    return max(
        coefficient_count,
        min(maximum_samples, multiplier * coefficient_count),
    )


def condition_number_from_points(
    *,
    basis: str,
    points: NDArray[np.floating],
    indices: Sequence[tuple[int, ...]],
    degree: int,
    basis_domain_scale: float,
    dtype: np.dtype,
    normalise_columns: bool,
) -> float:
    """
    Compute the sampled 2-norm condition number for a basis matrix.

    Column normalisation is enabled by default to focus on near-linear
    dependence rather than merely different basis-function magnitudes.
    """
    matrix = multivariate_design_matrix(
        basis=basis,
        points=points,
        indices=indices,
        degree=degree,
        scale=basis_domain_scale,
        dtype=dtype,
    )

    matrix64 = np.asarray(matrix, dtype=np.float64)

    if normalise_columns:
        norms = np.linalg.norm(matrix64, axis=0)
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        matrix64 = matrix64 / safe_norms

    singular_values = np.linalg.svd(
        matrix64,
        compute_uv=False,
        full_matrices=False,
    )

    if singular_values.size == 0 or singular_values[-1] == 0.0:
        return float("inf")

    return float(singular_values[0] / singular_values[-1])


# =============================================================================
# Dynamic benchmark mechanics
# =============================================================================

def deterministic_perturbation(
    *,
    coefficients: NDArray[np.floating],
    unit_direction: NDArray[np.float64],
    magnitude: float,
    dtype: np.dtype,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Apply the same perturbation direction to every time step of a trajectory.
    """
    coefficient_norm = np.linalg.norm(
        np.asarray(coefficients, dtype=np.float64)
    )
    target_norm = magnitude * max(coefficient_norm, 1.0)
    perturbation = np.asarray(
        target_norm * unit_direction,
        dtype=dtype,
    )
    perturbed = np.asarray(coefficients + perturbation, dtype=dtype)
    return perturbed, perturbation


def make_unit_direction(
    *,
    size: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate a deterministic random unit vector."""
    direction = rng.normal(size=size)
    norm = np.linalg.norm(direction)

    if norm == 0.0:
        direction[0] = 1.0
        norm = 1.0

    return np.asarray(direction / norm, dtype=np.float64)


def run_basis_trajectory(
    *,
    configuration: TrajectoryConfiguration,
    basis: str,
    schedule_values: NDArray[np.float64],
    monomial_tensor: NDArray[np.floating],
    indices: Sequence[tuple[int, ...]],
    condition_base_samples: Sequence[NDArray[np.float64]],
    perturbation_direction: NDArray[np.float64],
    condition_column_normalisation: bool,
) -> list[DynamicConstraintRecord]:
    """Run one basis through an entire changing-constraint trajectory."""
    dtype = SUPPORTED_DTYPES[configuration.dtype]

    conversion_start = time.perf_counter()
    basis_tensor = convert_monomial_tensor(
        monomial_tensor=monomial_tensor,
        basis=basis,
        degree=configuration.degree,
        dimension=configuration.dimension,
        scale=configuration.basis_domain_scale,
        dtype=dtype,
    )
    conversion_runtime = time.perf_counter() - conversion_start

    coefficients = extract_coefficients(
        basis_tensor,
        indices,
        dtype,
    )

    perturbed_coefficients, perturbation = deterministic_perturbation(
        coefficients=coefficients,
        unit_direction=perturbation_direction,
        magnitude=configuration.perturbation_magnitude,
        dtype=dtype,
    )

    coefficient_norm = float(
        np.linalg.norm(np.asarray(coefficients, dtype=np.float64))
    )
    perturbation_norm = float(
        np.linalg.norm(np.asarray(perturbation, dtype=np.float64))
    )

    records: list[DynamicConstraintRecord] = []
    cumulative_runtime = conversion_runtime
    previous_scale = float(schedule_values[0])
    maximum_volume = simplex_volume(
        configuration.maximum_constraint_scale,
        configuration.dimension,
    )

    for step, constraint_scale_value in enumerate(schedule_values):
        step_start = time.perf_counter()

        update_start = time.perf_counter()
        constraint_scale = float(constraint_scale_value)
        constraint_delta = (
            0.0 if step == 0 else constraint_scale - previous_scale
        )
        relative_constraint_delta = (
            0.0
            if step == 0
            else abs(constraint_delta) / max(abs(previous_scale), 1.0e-30)
        )
        current_volume = simplex_volume(
            constraint_scale,
            configuration.dimension,
        )
        relative_volume = current_volume / maximum_volume
        constraint_update_runtime = time.perf_counter() - update_start

        quadrature_start = time.perf_counter()
        quadrature_points, quadrature_weights = simplex_quadrature_rule(
            dimension=configuration.dimension,
            scale=constraint_scale,
            order=configuration.quadrature_order,
            dtype=dtype,
        )
        quadrature_runtime = time.perf_counter() - quadrature_start

        reference_integral = exact_simplex_integral(
            monomial_tensor,
            indices,
            constraint_scale,
        )

        condition_start = time.perf_counter()
        condition_points = sample_uniform_simplex_from_exponentials(
            exponential_samples=condition_base_samples[step],
            scale=constraint_scale,
            dtype=dtype,
        )
        condition_number = condition_number_from_points(
            basis=basis,
            points=condition_points,
            indices=indices,
            degree=configuration.degree,
            basis_domain_scale=configuration.basis_domain_scale,
            dtype=dtype,
            normalise_columns=condition_column_normalisation,
        )
        condition_runtime = time.perf_counter() - condition_start

        evaluation_start = time.perf_counter()
        design_matrix = multivariate_design_matrix(
            basis=basis,
            points=quadrature_points,
            indices=indices,
            degree=configuration.degree,
            scale=configuration.basis_domain_scale,
            dtype=dtype,
        )
        values = np.asarray(design_matrix @ coefficients, dtype=dtype)
        evaluation_runtime = time.perf_counter() - evaluation_start

        integration_start = time.perf_counter()
        computed_integral = integrate_values(
            values,
            quadrature_weights,
            dtype,
        )
        integration_runtime = time.perf_counter() - integration_start

        integration_absolute_error = abs(
            computed_integral - reference_integral
        )
        integration_relative_error = safe_relative_error(
            computed_integral,
            reference_integral,
        )

        perturbation_start = time.perf_counter()
        perturbed_values = np.asarray(
            design_matrix @ perturbed_coefficients,
            dtype=dtype,
        )
        perturbed_integral = integrate_values(
            perturbed_values,
            quadrature_weights,
            dtype,
        )
        perturbation_runtime = time.perf_counter() - perturbation_start

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

        step_runtime = time.perf_counter() - step_start
        cumulative_runtime += step_runtime

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
            constraint_update_runtime,
            quadrature_runtime,
            condition_runtime,
            evaluation_runtime,
            integration_runtime,
            perturbation_runtime,
            step_runtime,
            cumulative_runtime,
        )
        finite = all(np.isfinite(value) for value in finite_values)

        records.append(
            DynamicConstraintRecord(
                trial=configuration.trial,
                seed=configuration.seed,
                degree=configuration.degree,
                dimension=configuration.dimension,
                dtype=configuration.dtype,
                basis=basis,
                schedule=configuration.schedule,
                step=step,
                steps=configuration.steps,
                normalised_time=(
                    0.0
                    if configuration.steps == 1
                    else step / (configuration.steps - 1)
                ),
                previous_constraint_scale=previous_scale,
                constraint_scale=constraint_scale,
                constraint_delta=constraint_delta,
                relative_constraint_delta=relative_constraint_delta,
                simplex_volume=current_volume,
                relative_simplex_volume=relative_volume,
                basis_domain_scale=configuration.basis_domain_scale,
                coefficient_scale=configuration.coefficient_scale,
                perturbation_magnitude=configuration.perturbation_magnitude,
                quadrature_order=configuration.quadrature_order,
                coefficient_count=len(indices),
                quadrature_point_count=quadrature_points.shape[0],
                condition_sample_count=condition_points.shape[0],
                reference_integral=reference_integral,
                computed_integral=computed_integral,
                integration_absolute_error=integration_absolute_error,
                integration_relative_error=integration_relative_error,
                condition_number=condition_number,
                log10_condition_number=(
                    math.log10(condition_number)
                    if condition_number > 0.0
                    and np.isfinite(condition_number)
                    else float("inf")
                ),
                coefficient_norm=coefficient_norm,
                perturbation_norm=perturbation_norm,
                perturbed_integral=perturbed_integral,
                perturbation_absolute_change=(
                    perturbation_absolute_change
                ),
                perturbation_relative_change=(
                    perturbation_relative_change
                ),
                perturbation_sensitivity=perturbation_sensitivity,
                relative_perturbation_sensitivity=(
                    relative_perturbation_sensitivity
                ),
                basis_conversion_runtime_seconds=conversion_runtime,
                constraint_update_runtime_seconds=(
                    constraint_update_runtime
                ),
                quadrature_runtime_seconds=quadrature_runtime,
                condition_runtime_seconds=condition_runtime,
                evaluation_runtime_seconds=evaluation_runtime,
                integration_runtime_seconds=integration_runtime,
                perturbation_runtime_seconds=perturbation_runtime,
                step_runtime_seconds=step_runtime,
                cumulative_trajectory_runtime_seconds=cumulative_runtime,
                finite=finite,
            )
        )

        previous_scale = constraint_scale

    return records


def run_benchmark(
    *,
    degrees: Sequence[int],
    dimensions: Sequence[int],
    dtypes: Sequence[np.dtype],
    bases: Sequence[str],
    schedules: Sequence[str],
    steps: int,
    minimum_constraint_scale: float,
    maximum_constraint_scale: float,
    basis_domain_scale: float | None,
    trials: int,
    seed: int,
    coefficient_scale: float,
    perturbation_magnitude: float,
    quadrature_order: int,
    condition_sample_multiplier: int,
    maximum_condition_samples: int,
    condition_column_normalisation: bool,
) -> list[DynamicConstraintRecord]:
    """Run all requested dynamic-constraint configurations."""
    if trials <= 0:
        raise ValueError("trials must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if quadrature_order <= 0:
        raise ValueError("quadrature order must be positive.")
    if condition_sample_multiplier <= 0:
        raise ValueError("condition sample multiplier must be positive.")
    if maximum_condition_samples <= 0:
        raise ValueError("maximum condition samples must be positive.")

    require_positive(coefficient_scale, "coefficient scale")
    require_positive(perturbation_magnitude, "perturbation magnitude")
    require_positive(minimum_constraint_scale, "minimum constraint scale")
    require_positive(maximum_constraint_scale, "maximum constraint scale")

    if minimum_constraint_scale > maximum_constraint_scale:
        raise ValueError(
            "minimum constraint scale cannot exceed maximum constraint scale."
        )

    if basis_domain_scale is None:
        effective_basis_domain_scale = maximum_constraint_scale
    else:
        require_positive(basis_domain_scale, "basis domain scale")
        effective_basis_domain_scale = basis_domain_scale

    if effective_basis_domain_scale < maximum_constraint_scale:
        raise ValueError(
            "basis domain scale must be at least the maximum constraint scale."
        )

    for degree in degrees:
        if degree < 0:
            raise ValueError("degrees must be non-negative.")

    for dimension in dimensions:
        if dimension <= 0:
            raise ValueError("dimensions must be positive.")

    records: list[DynamicConstraintRecord] = []

    for dtype_index, dtype_value in enumerate(dtypes):
        dtype = np.dtype(dtype_value)
        dtype_name = dtype.name

        for dimension_index, dimension in enumerate(dimensions):
            for degree in degrees:
                indices = total_degree_multiindices(dimension, degree)
                sample_count = condition_sample_count(
                    coefficient_count=len(indices),
                    multiplier=condition_sample_multiplier,
                    maximum_samples=maximum_condition_samples,
                )

                for trial in range(trials):
                    base_seed = (
                        seed
                        + 10_000_000 * dtype_index
                        + 1_000_000 * dimension_index
                        + 10_000 * degree
                        + trial
                    )

                    coefficient_rng = np.random.default_rng(base_seed)
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

                    for schedule_index, schedule in enumerate(schedules):
                        trajectory_seed = (
                            base_seed
                            + 100_000_000
                            + schedule_index * 1_000_000
                        )

                        schedule_rng = np.random.default_rng(
                            trajectory_seed
                        )
                        schedule_values = generate_constraint_schedule(
                            schedule=schedule,
                            minimum=minimum_constraint_scale,
                            maximum=maximum_constraint_scale,
                            steps=steps,
                            rng=schedule_rng,
                        )

                        condition_rng = np.random.default_rng(
                            trajectory_seed + 100_000
                        )
                        condition_base_samples = [
                            condition_rng.exponential(
                                scale=1.0,
                                size=(sample_count, dimension + 1),
                            )
                            for _ in range(steps)
                        ]

                        for basis_index, basis in enumerate(bases):
                            perturbation_rng = np.random.default_rng(
                                trajectory_seed
                                + 200_000
                                + 10_000 * basis_index
                            )
                            perturbation_direction = make_unit_direction(
                                size=len(indices),
                                rng=perturbation_rng,
                            )

                            configuration = TrajectoryConfiguration(
                                trial=trial,
                                seed=trajectory_seed,
                                degree=degree,
                                dimension=dimension,
                                dtype=dtype_name,
                                schedule=schedule,
                                steps=steps,
                                minimum_constraint_scale=(
                                    minimum_constraint_scale
                                ),
                                maximum_constraint_scale=(
                                    maximum_constraint_scale
                                ),
                                basis_domain_scale=(
                                    effective_basis_domain_scale
                                ),
                                coefficient_scale=coefficient_scale,
                                perturbation_magnitude=(
                                    perturbation_magnitude
                                ),
                                quadrature_order=quadrature_order,
                            )

                            basis_records = run_basis_trajectory(
                                configuration=configuration,
                                basis=basis,
                                schedule_values=schedule_values,
                                monomial_tensor=monomial_tensor,
                                indices=indices,
                                condition_base_samples=(
                                    condition_base_samples
                                ),
                                perturbation_direction=(
                                    perturbation_direction
                                ),
                                condition_column_normalisation=(
                                    condition_column_normalisation
                                ),
                            )
                            records.extend(basis_records)

    return records


# =============================================================================
# Reporting
# =============================================================================

def create_summary(
    records: Sequence[DynamicConstraintRecord],
) -> DynamicConstraintSummary:
    """Create an aggregate summary."""
    if not records:
        return DynamicConstraintSummary(
            total_records=0,
            finite_records=0,
            non_finite_records=0,
            trajectory_count=0,
            maximum_integration_absolute_error=0.0,
            maximum_integration_relative_error=0.0,
            maximum_condition_number=0.0,
            maximum_perturbation_sensitivity=0.0,
            maximum_step_runtime_seconds=0.0,
            maximum_cumulative_trajectory_runtime_seconds=0.0,
        )

    finite_count = sum(record.finite for record in records)
    trajectories = {
        (
            record.trial,
            record.seed,
            record.degree,
            record.dimension,
            record.dtype,
            record.basis,
            record.schedule,
        )
        for record in records
    }

    return DynamicConstraintSummary(
        total_records=len(records),
        finite_records=finite_count,
        non_finite_records=len(records) - finite_count,
        trajectory_count=len(trajectories),
        maximum_integration_absolute_error=max(
            record.integration_absolute_error for record in records
        ),
        maximum_integration_relative_error=max(
            record.integration_relative_error for record in records
        ),
        maximum_condition_number=max(
            record.condition_number for record in records
        ),
        maximum_perturbation_sensitivity=max(
            record.perturbation_sensitivity for record in records
        ),
        maximum_step_runtime_seconds=max(
            record.step_runtime_seconds for record in records
        ),
        maximum_cumulative_trajectory_runtime_seconds=max(
            record.cumulative_trajectory_runtime_seconds
            for record in records
        ),
    )


def write_csv(
    records: Sequence[DynamicConstraintRecord],
    output_path: Path,
) -> None:
    """Write all detailed records to CSV."""
    if not records:
        raise ValueError("Cannot write an empty result set.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(records[0]).keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))


def environment_information() -> dict[str, object]:
    """Collect reproducibility metadata."""
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
    summary: DynamicConstraintSummary,
    configuration: dict[str, object],
    output_path: Path,
) -> None:
    """Write summary, configuration, and environment metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": asdict(summary),
        "configuration": configuration,
        "environment": environment_information(),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def group_records(
    records: Iterable[DynamicConstraintRecord],
) -> dict[
    tuple[str, int, str, str],
    list[DynamicConstraintRecord],
]:
    """Group records for compact terminal reporting."""
    groups: dict[
        tuple[str, int, str, str],
        list[DynamicConstraintRecord],
    ] = {}

    for record in records:
        key = (
            record.dtype,
            record.dimension,
            record.schedule,
            record.basis,
        )
        groups.setdefault(key, []).append(record)

    return groups


def print_table(
    records: Sequence[DynamicConstraintRecord],
) -> None:
    """Print aggregate results by dtype, dimension, schedule, and basis."""
    groups = group_records(records)

    header = (
        f"{'dtype':<9}"
        f"{'dim':>5}"
        f"{'schedule':>14}"
        f"{'basis':>12}"
        f"{'max rel err':>16}"
        f"{'max cond':>16}"
        f"{'max pert sens':>16}"
        f"{'mean step(s)':>16}"
    )

    print()
    print(header)
    print("-" * len(header))

    for key in sorted(groups):
        dtype_name, dimension, schedule, basis = key
        group = groups[key]

        print(
            f"{dtype_name:<9}"
            f"{dimension:>5d}"
            f"{schedule:>14}"
            f"{basis:>12}"
            f"{max(r.integration_relative_error for r in group):>16.6e}"
            f"{max(r.condition_number for r in group):>16.6e}"
            f"{max(r.perturbation_sensitivity for r in group):>16.6e}"
            f"{np.mean([r.step_runtime_seconds for r in group]):>16.6e}"
        )


def print_non_finite_records(
    records: Sequence[DynamicConstraintRecord],
    limit: int = 20,
) -> None:
    """Print a compact description of non-finite records."""
    failures = [record for record in records if not record.finite]

    if not failures:
        return

    print("\nNon-finite records:")

    for record in failures[:limit]:
        print(
            "  "
            f"basis={record.basis}, "
            f"dtype={record.dtype}, "
            f"dimension={record.dimension}, "
            f"degree={record.degree}, "
            f"schedule={record.schedule}, "
            f"step={record.step}, "
            f"scale={record.constraint_scale:.6g}"
        )

    if len(failures) > limit:
        print(f"  ... and {len(failures) - limit} more.")


# =============================================================================
# Command-line interface
# =============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark polynomial bases under per-step changing simplex "
            "constraints."
        )
    )

    parser.add_argument(
        "--degrees",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 5, 8, 10],
        help="Total polynomial degrees.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Simplex dimensions.",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        type=parse_dtype,
        default=[np.dtype(np.float64), np.dtype(np.float32)],
        help="Floating-point types.",
    )
    parser.add_argument(
        "--bases",
        nargs="+",
        type=parse_basis,
        default=list(SUPPORTED_BASES),
        help="Polynomial bases.",
    )
    parser.add_argument(
        "--schedules",
        nargs="+",
        type=parse_schedule,
        default=["shrink", "expand", "oscillate", "random_walk"],
        help="Dynamic constraint schedules.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=21,
        help="Constraint updates in each trajectory.",
    )
    parser.add_argument(
        "--minimum-constraint-scale",
        type=float,
        default=0.25,
        help="Smallest simplex scale.",
    )
    parser.add_argument(
        "--maximum-constraint-scale",
        type=float,
        default=1.0,
        help="Largest simplex scale.",
    )
    parser.add_argument(
        "--basis-domain-scale",
        type=float,
        default=None,
        help=(
            "Physical interval used to define Legendre/Chebyshev bases. "
            "Defaults to the maximum constraint scale."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Random polynomial trials per shared configuration.",
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
        help="Random coefficient standard deviation.",
    )
    parser.add_argument(
        "--perturbation-magnitude",
        type=float,
        default=1.0e-7,
        help="Relative coefficient perturbation norm.",
    )
    parser.add_argument(
        "--quadrature-order",
        type=int,
        default=16,
        help="Gauss-Legendre order per Duffy coordinate.",
    )
    parser.add_argument(
        "--condition-sample-multiplier",
        type=int,
        default=2,
        help="Condition rows per coefficient.",
    )
    parser.add_argument(
        "--maximum-condition-samples",
        type=int,
        default=2000,
        help="Maximum sampled rows for each condition estimate.",
    )
    parser.add_argument(
        "--no-condition-column-normalisation",
        action="store_true",
        help="Disable condition-matrix column normalisation.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/dynamic_constraints"),
        help="Structured output directory.",
    )
    parser.add_argument(
        "--allow-non-finite",
        action="store_true",
        help="Return success even if non-finite records occur.",
    )

    return parser


def main() -> int:
    """Run the complete dynamic-constraint benchmark."""
    parser = build_argument_parser()
    arguments = parser.parse_args()

    dtype_names = [np.dtype(dtype).name for dtype in arguments.dtypes]
    effective_basis_domain_scale = (
        arguments.maximum_constraint_scale
        if arguments.basis_domain_scale is None
        else arguments.basis_domain_scale
    )

    print("Dynamic-constraint benchmark")
    print("----------------------------")
    print(f"Degrees:                  {arguments.degrees}")
    print(f"Dimensions:               {arguments.dimensions}")
    print(f"Dtypes:                   {dtype_names}")
    print(f"Bases:                    {arguments.bases}")
    print(f"Schedules:                {arguments.schedules}")
    print(f"Steps:                    {arguments.steps}")
    print(
        "Constraint scale range:   "
        f"[{arguments.minimum_constraint_scale}, "
        f"{arguments.maximum_constraint_scale}]"
    )
    print(
        "Basis domain scale:       "
        f"{effective_basis_domain_scale}"
    )
    print(f"Trials:                   {arguments.trials}")
    print(f"Quadrature order:         {arguments.quadrature_order}")
    print(
        "Perturbation magnitude:   "
        f"{arguments.perturbation_magnitude:.3e}"
    )
    print(
        "Column-normalised cond.:  "
        f"{not arguments.no_condition_column_normalisation}"
    )

    records = run_benchmark(
        degrees=arguments.degrees,
        dimensions=arguments.dimensions,
        dtypes=arguments.dtypes,
        bases=arguments.bases,
        schedules=arguments.schedules,
        steps=arguments.steps,
        minimum_constraint_scale=arguments.minimum_constraint_scale,
        maximum_constraint_scale=arguments.maximum_constraint_scale,
        basis_domain_scale=arguments.basis_domain_scale,
        trials=arguments.trials,
        seed=arguments.seed,
        coefficient_scale=arguments.coefficient_scale,
        perturbation_magnitude=arguments.perturbation_magnitude,
        quadrature_order=arguments.quadrature_order,
        condition_sample_multiplier=(
            arguments.condition_sample_multiplier
        ),
        maximum_condition_samples=arguments.maximum_condition_samples,
        condition_column_normalisation=(
            not arguments.no_condition_column_normalisation
        ),
    )

    summary = create_summary(records)

    output_directory = arguments.output_directory.resolve()
    csv_path = (
        output_directory / "dynamic_constraints_results.csv"
    )
    json_path = (
        output_directory / "dynamic_constraints_summary.json"
    )

    configuration = {
        "degrees": arguments.degrees,
        "dimensions": arguments.dimensions,
        "dtypes": dtype_names,
        "bases": arguments.bases,
        "schedules": arguments.schedules,
        "steps": arguments.steps,
        "minimum_constraint_scale": (
            arguments.minimum_constraint_scale
        ),
        "maximum_constraint_scale": (
            arguments.maximum_constraint_scale
        ),
        "basis_domain_scale": effective_basis_domain_scale,
        "trials": arguments.trials,
        "seed": arguments.seed,
        "coefficient_scale": arguments.coefficient_scale,
        "perturbation_magnitude": (
            arguments.perturbation_magnitude
        ),
        "quadrature_order": arguments.quadrature_order,
        "condition_sample_multiplier": (
            arguments.condition_sample_multiplier
        ),
        "maximum_condition_samples": (
            arguments.maximum_condition_samples
        ),
        "condition_column_normalisation": (
            not arguments.no_condition_column_normalisation
        ),
        "constraint_definition": (
            "x_i >= 0 and sum_i x_i <= constraint_scale_t"
        ),
        "reference_method": (
            "analytical monomial simplex integral using Gamma identities"
        ),
        "integration_method": (
            "tensor Gauss-Legendre quadrature under Duffy transformation"
        ),
        "dynamic_protocol": (
            "fixed polynomial with a changing feasible simplex at each step"
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
    print(f"Trajectories:              {summary.trajectory_count}")
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
        "Maximum step runtime (s):  "
        f"{summary.maximum_step_runtime_seconds:.6e}"
    )
    print(
        "Maximum trajectory time:   "
        f"{summary.maximum_cumulative_trajectory_runtime_seconds:.6e}"
    )
    print(f"\nDetailed CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")

    if summary.non_finite_records == 0:
        print("\nRESULT: DYNAMIC-CONSTRAINT BENCHMARK COMPLETED")
        return 0

    print("\nRESULT: BENCHMARK PRODUCED NON-FINITE VALUES")
    return 0 if arguments.allow_non_finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
