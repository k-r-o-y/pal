#!/usr/bin/env python3
"""
Floating-point precision benchmark for the simplex experiments.

This benchmark compares float32 and float64 arithmetic while keeping the underlying
mathematical experiment fixed. For every dimension-degree-trial configuration, the
same polynomial, simplex points, and coefficient-perturbation samples are reused across
both precisions.

The benchmark records:

- basis-matrix rank and singular values;
- basis-matrix condition number;
- analytical float64 reference integral;
- computed integral in the tested precision;
- absolute and relative integration error;
- coefficient perturbation sensitivity;
- integration-stage runtime;
- paired float32-to-float64 error and runtime ratios.

The analytical reference is computed from the original float64 monomial coefficients.
The tested computation then quantises coefficients, points, basis conversion,
integration terms, and accumulation to float32 or float64 as requested.

Expected outputs
----------------
results/simplex/precision_results.csv
results/simplex/precision_summary.json

Examples
--------
Run the default benchmark:

    python -m analysis.simplex.run_precision_benchmark

Run a small smoke test:

    python -m analysis.simplex.run_precision_benchmark \
        --dimensions 1 2 \
        --maximum-degree 3 \
        --trials 2

Run selected degrees:

    python -m analysis.simplex.run_precision_benchmark \
        --dimensions 1 2 3 \
        --degrees 1 2 3 5 8 \
        --trials 10 \
        --precisions float32 float64
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from analysis.simplex.run_dimension_benchmark import (
    BASES,
    DEFAULT_DIMENSION_DEGREE_GRID,
    BasisName,
    CoefficientMap,
    MultiIndex,
    build_basis_matrix,
    convert_basis_to_monomial,
    convert_monomial_to_basis,
    enumerate_total_degree_indices,
    expected_basis_count,
    parse_dtype,
    perturbation_sensitivity,
    sample_monomial_coefficients,
    sample_scaled_simplex_points,
    scaled_simplex_monomial_integral,
)


DEFAULT_PRECISIONS: tuple[str, ...] = ("float32", "float64")

# A perturbation of 1e-6 remains representable in float32 while still being small
# relative to typical coefficient magnitudes.
DEFAULT_PERTURBATION_MAGNITUDE = 1.0e-6

# Default cross-precision grid. These configurations are large enough to expose
# precision effects without repeating the most expensive 4D and 5D experiments.
DEFAULT_PRECISION_DEGREE_GRID: dict[int, tuple[int, ...]] = {
    1: (0, 1, 2, 3, 5, 8, 10),
    2: (0, 1, 2, 3, 5, 8, 10),
    3: (0, 1, 2, 3, 5, 8),
}


@dataclass(frozen=True)
class PrecisionBenchmarkRecord:
    """One basis-precision-dimension-degree-trial benchmark result."""

    pair_id: str
    basis: str
    precision: str
    dtype_bits: int
    dimension: int
    degree: int
    basis_count: int
    trial: int
    seed: int
    point_seed: int
    perturbation_seed: int
    scale: float
    oversampling_factor: float
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
    coefficient_norm: float
    realised_perturbation_norm: float
    realised_relative_perturbation: float
    perturbed_integral: float
    perturbation_sensitivity: float
    runtime_seconds: float


def cast_scalar(value: float, dtype: np.dtype) -> float:
    """Round one scalar to the requested floating-point dtype."""
    return float(np.asarray(value, dtype=dtype))


def cast_coefficient_map(
    coefficients: Mapping[MultiIndex, float],
    *,
    dtype: np.dtype,
) -> CoefficientMap:
    """Round every coefficient to the requested floating-point dtype."""
    return {
        index: cast_scalar(value, dtype)
        for index, value in coefficients.items()
    }


def clean_coefficient_map(
    coefficients: Mapping[MultiIndex, float],
    *,
    dtype: np.dtype,
) -> CoefficientMap:
    """
    Quantise and remove coefficients that become exactly zero.

    An all-zero result is not expected for the generated test polynomials, but the
    function retains one zero coefficient defensively so downstream code still has a
    well-defined dimensionality.
    """
    result = {
        index: cast_scalar(value, dtype)
        for index, value in coefficients.items()
        if cast_scalar(value, dtype) != 0.0
    }

    if result:
        return result

    if not coefficients:
        raise ValueError("coefficient map cannot be empty")

    first_index = min(coefficients)
    return {first_index: cast_scalar(0.0, dtype)}


def convert_monomial_to_basis_in_dtype(
    monomial_coefficients: Mapping[MultiIndex, float],
    *,
    basis: BasisName,
    dtype: np.dtype,
) -> CoefficientMap:
    """
    Convert monomial coefficients and quantise the resulting basis coefficients.

    The shared conversion implementation evaluates the algebraic conversion in
    float64. Quantisation immediately after conversion models storage and subsequent
    computation in the tested precision.
    """
    converted = convert_monomial_to_basis(
        monomial_coefficients,
        basis=basis,
    )
    return clean_coefficient_map(converted, dtype=dtype)


def convert_basis_to_monomial_in_dtype(
    basis_coefficients: Mapping[MultiIndex, float],
    *,
    basis: BasisName,
    dtype: np.dtype,
) -> CoefficientMap:
    """
    Convert a tested basis representation back to monomials and quantise it.

    This preserves basis-conversion effects while ensuring the coefficients consumed by
    the simplex integrator are represented in the tested precision.
    """
    converted = convert_basis_to_monomial(
        basis_coefficients,
        basis=basis,
    )
    return clean_coefficient_map(converted, dtype=dtype)


def floating_point_sum(
    values: Sequence[float],
    *,
    dtype: np.dtype,
) -> float:
    """
    Sum values sequentially using the requested arithmetic precision.

    Python's built-in sum and math.fsum accumulate in Python float arithmetic. This
    routine explicitly rounds after each addition so float32 accumulation is genuinely
    tested.
    """
    accumulator = np.asarray(0.0, dtype=dtype)

    for value in values:
        term = np.asarray(value, dtype=dtype)
        accumulator = np.asarray(accumulator + term, dtype=dtype)

    return float(accumulator)


def integrate_monomial_polynomial_in_dtype(
    coefficients: Mapping[MultiIndex, float],
    *,
    scale: float,
    dtype: np.dtype,
) -> float:
    """
    Integrate a monomial polynomial with tested-precision term construction and sum.

    The analytical monomial moments are generated in float64 and then rounded to the
    tested dtype before multiplication and sequential accumulation.
    """
    terms: list[float] = []

    for alpha in sorted(coefficients):
        coefficient = np.asarray(coefficients[alpha], dtype=dtype)
        moment = np.asarray(
            scaled_simplex_monomial_integral(alpha, scale=scale),
            dtype=dtype,
        )
        term = np.asarray(coefficient * moment, dtype=dtype)
        terms.append(float(term))

    return floating_point_sum(terms, dtype=dtype)


def integrate_in_basis_in_dtype(
    coefficients: Mapping[MultiIndex, float],
    *,
    basis: BasisName,
    scale: float,
    dtype: np.dtype,
) -> float:
    """Integrate one basis representation using tested-precision arithmetic."""
    monomial_coefficients = convert_basis_to_monomial_in_dtype(
        coefficients,
        basis=basis,
        dtype=dtype,
    )
    return integrate_monomial_polynomial_in_dtype(
        monomial_coefficients,
        scale=scale,
        dtype=dtype,
    )


def guarded_relative_error(
    estimate: float,
    reference: float,
    *,
    dtype: np.dtype,
) -> float:
    """
    Compute relative error using a denominator appropriate to the tested precision.

    The reference itself remains the shared float64 analytical value.
    """
    denominator = max(
        abs(reference),
        float(np.finfo(dtype).eps),
    )
    return abs(estimate - reference) / denominator


def matrix_diagnostics_in_dtype(
    matrix: NDArray[np.floating],
    *,
    dtype: np.dtype,
) -> tuple[int, float, float, float]:
    """
    Compute matrix diagnostics without forcing float32 matrices to float64 first.

    NumPy's SVD is called on an array of the requested dtype. The numerical-rank
    threshold is also based on the corresponding machine epsilon.
    """
    tested_matrix = np.asarray(matrix, dtype=dtype)

    singular_values = np.linalg.svd(
        tested_matrix,
        full_matrices=False,
        compute_uv=False,
    )
    singular_values = np.asarray(singular_values, dtype=dtype)

    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])

    tolerance = (
        max(tested_matrix.shape)
        * float(np.finfo(dtype).eps)
        * sigma_max
    )
    rank = int(
        np.count_nonzero(
            singular_values.astype(np.float64) > tolerance
        )
    )

    if sigma_min <= 0.0:
        condition_number = math.inf
    else:
        condition_number = sigma_max / sigma_min

    return rank, sigma_max, sigma_min, float(condition_number)


def perturb_coefficients_with_shared_noise(
    coefficients: Mapping[MultiIndex, float],
    *,
    relative_magnitude: float,
    standard_normal_samples: Mapping[MultiIndex, float],
    dtype: np.dtype,
) -> tuple[CoefficientMap, float, float, float]:
    """
    Perturb coefficients using the same underlying normal samples in each precision.

    Returns:
        perturbed coefficient map,
        coefficient norm,
        realised perturbation norm,
        realised relative perturbation.
    """
    if relative_magnitude <= 0.0:
        raise ValueError("relative_magnitude must be positive")

    indices = sorted(coefficients)

    missing_noise = [
        index
        for index in indices
        if index not in standard_normal_samples
    ]
    if missing_noise:
        raise KeyError(
            "shared perturbation samples are missing coefficient indices: "
            f"{missing_noise[:5]}"
        )

    vector = np.asarray(
        [coefficients[index] for index in indices],
        dtype=dtype,
    )
    direction = np.asarray(
        [standard_normal_samples[index] for index in indices],
        dtype=dtype,
    )

    direction_norm = np.linalg.norm(direction)
    direction_norm_value = float(direction_norm)

    if not math.isfinite(direction_norm_value) or direction_norm_value == 0.0:
        raise FloatingPointError(
            "shared perturbation direction has an invalid norm"
        )

    direction = np.asarray(direction / direction_norm, dtype=dtype)

    coefficient_norm_array = np.linalg.norm(vector)
    coefficient_norm = float(coefficient_norm_array)
    coefficient_scale = max(
        coefficient_norm,
        float(np.finfo(dtype).eps),
    )

    perturbation_scale = np.asarray(
        relative_magnitude * coefficient_scale,
        dtype=dtype,
    )
    perturbation = np.asarray(
        perturbation_scale * direction,
        dtype=dtype,
    )
    perturbed_vector = np.asarray(
        vector + perturbation,
        dtype=dtype,
    )

    realised_perturbation_norm = float(
        np.linalg.norm(
            np.asarray(perturbed_vector - vector, dtype=dtype)
        )
    )
    realised_relative_perturbation = (
        realised_perturbation_norm
        / max(coefficient_norm, float(np.finfo(dtype).eps))
    )

    perturbed_coefficients = {
        index: float(value)
        for index, value in zip(
            indices,
            perturbed_vector,
            strict=True,
        )
    }

    return (
        perturbed_coefficients,
        coefficient_norm,
        realised_perturbation_norm,
        realised_relative_perturbation,
    )


def generate_shared_noise_map(
    *,
    indices: Sequence[MultiIndex],
    rng: np.random.Generator,
) -> dict[MultiIndex, float]:
    """Generate float64 standard-normal samples indexed by basis coefficient."""
    samples = rng.normal(size=len(indices))

    return {
        index: float(value)
        for index, value in zip(indices, samples, strict=True)
    }


def resolve_precision_grid(
    *,
    dimensions: Sequence[int],
    maximum_degree: int | None,
    explicit_degrees: Sequence[int] | None,
) -> dict[int, tuple[int, ...]]:
    """Resolve the dimension-degree grid for the precision experiment."""
    if not dimensions:
        raise ValueError("at least one dimension must be requested")

    if maximum_degree is not None and maximum_degree < 0:
        raise ValueError("maximum_degree must be non-negative")

    requested_explicit = (
        tuple(sorted(set(explicit_degrees)))
        if explicit_degrees is not None
        else None
    )

    if requested_explicit is not None and any(
        degree < 0 for degree in requested_explicit
    ):
        raise ValueError("degrees must be non-negative")

    grid: dict[int, tuple[int, ...]] = {}

    for dimension in dimensions:
        if dimension < 1:
            raise ValueError("dimensions must be positive")

        if dimension in DEFAULT_PRECISION_DEGREE_GRID:
            supported = DEFAULT_PRECISION_DEGREE_GRID[dimension]
        elif dimension in DEFAULT_DIMENSION_DEGREE_GRID:
            supported = DEFAULT_DIMENSION_DEGREE_GRID[dimension]
        else:
            raise ValueError(
                f"no supported degree grid is defined for dimension {dimension}"
            )

        if requested_explicit is None:
            degrees = supported
        else:
            degrees = tuple(
                degree
                for degree in requested_explicit
                if degree in supported
            )

            unsupported = sorted(
                set(requested_explicit) - set(supported)
            )
            if unsupported:
                print(
                    f"Warning: dimension {dimension} does not support "
                    f"requested degrees {unsupported}; they will be skipped."
                )

        if maximum_degree is not None:
            degrees = tuple(
                degree
                for degree in degrees
                if degree <= maximum_degree
            )

        if not degrees:
            raise ValueError(
                "no degree remains for dimension "
                f"{dimension} after applying the requested filters"
            )

        grid[dimension] = tuple(sorted(set(degrees)))

    return grid


def validate_precisions(precisions: Sequence[str]) -> tuple[str, ...]:
    """Validate and deduplicate the requested precision names."""
    if not precisions:
        raise ValueError("at least one precision must be requested")

    unique: list[str] = []

    for precision in precisions:
        parse_dtype(precision)
        if precision not in unique:
            unique.append(precision)

    return tuple(unique)


def run_precision_benchmark(
    *,
    grid: Mapping[int, Sequence[int]],
    precisions: Sequence[str],
    trials: int,
    seed: int,
    scale: float,
    oversampling_factor: float,
    perturbation_magnitude: float,
) -> list[PrecisionBenchmarkRecord]:
    """Execute all requested paired precision configurations."""
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if oversampling_factor < 1.0:
        raise ValueError("oversampling_factor must be at least 1")
    if perturbation_magnitude <= 0.0:
        raise ValueError("perturbation_magnitude must be positive")

    precision_names = validate_precisions(precisions)
    records: list[PrecisionBenchmarkRecord] = []

    for dimension, degrees in grid.items():
        for degree in degrees:
            indices = enumerate_total_degree_indices(
                dimension,
                degree,
            )
            basis_count = len(indices)
            expected_count = expected_basis_count(
                dimension,
                degree,
            )

            if basis_count != expected_count:
                raise AssertionError(
                    f"basis count mismatch for n={dimension}, d={degree}: "
                    f"expected {expected_count}, got {basis_count}"
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
                point_seed = trial_seed + 1
                perturbation_seed = trial_seed + 2

                # Generate one high-precision mathematical polynomial shared by both
                # tested precisions.
                coefficient_rng = np.random.default_rng(trial_seed)
                reference_coefficients = sample_monomial_coefficients(
                    indices=indices,
                    rng=coefficient_rng,
                    dtype=np.dtype(np.float64),
                )

                reference_integral = (
                    integrate_monomial_polynomial_in_dtype(
                        reference_coefficients,
                        scale=scale,
                        dtype=np.dtype(np.float64),
                    )
                )

                # Generate one float64 point cloud, then round that same cloud into
                # each tested precision.
                point_rng = np.random.default_rng(point_seed)
                shared_points = sample_scaled_simplex_points(
                    dimension=dimension,
                    count=num_points,
                    scale=scale,
                    rng=point_rng,
                    dtype=np.dtype(np.float64),
                )

                if not np.isfinite(shared_points).all():
                    raise FloatingPointError(
                        "non-finite shared simplex points for "
                        f"n={dimension}, d={degree}, trial={trial}"
                    )

                for basis in BASES:
                    # The same standard-normal coefficient samples are used for this
                    # basis in float32 and float64.
                    basis_indices = sorted(
                        convert_monomial_to_basis(
                            reference_coefficients,
                            basis=basis,
                        )
                    )
                    basis_noise_offset = BASES.index(basis) * 1_000_000
                    noise_rng = np.random.default_rng(
                        perturbation_seed + basis_noise_offset
                    )
                    shared_noise = generate_shared_noise_map(
                        indices=basis_indices,
                        rng=noise_rng,
                    )

                    pair_id = (
                        f"n{dimension}_d{degree}_trial{trial}_{basis}"
                    )

                    for precision in precision_names:
                        dtype = parse_dtype(precision)

                        tested_monomial_coefficients = (
                            cast_coefficient_map(
                                reference_coefficients,
                                dtype=dtype,
                            )
                        )

                        basis_coefficients = (
                            convert_monomial_to_basis_in_dtype(
                                tested_monomial_coefficients,
                                basis=basis,
                                dtype=dtype,
                            )
                        )

                        points = np.asarray(
                            shared_points,
                            dtype=dtype,
                        )

                        matrix = build_basis_matrix(
                            points,
                            indices=indices,
                            basis=basis,
                            scale=scale,
                            dtype=dtype,
                        )

                        if matrix.shape != (
                            num_points,
                            expected_count,
                        ):
                            raise AssertionError(
                                f"unexpected matrix shape for {basis}, "
                                f"{precision}, n={dimension}, d={degree}: "
                                f"{matrix.shape}"
                            )

                        if not np.isfinite(matrix).all():
                            raise FloatingPointError(
                                f"non-finite matrix for {basis}, {precision}, "
                                f"n={dimension}, d={degree}, trial={trial}"
                            )

                        (
                            matrix_rank,
                            sigma_max,
                            sigma_min,
                            condition_number,
                        ) = matrix_diagnostics_in_dtype(
                            matrix,
                            dtype=dtype,
                        )

                        start = time.perf_counter()
                        computed_integral = integrate_in_basis_in_dtype(
                            basis_coefficients,
                            basis=basis,
                            scale=scale,
                            dtype=dtype,
                        )
                        runtime_seconds = (
                            time.perf_counter() - start
                        )

                        absolute_error = abs(
                            computed_integral - reference_integral
                        )
                        relative_error = guarded_relative_error(
                            computed_integral,
                            reference_integral,
                            dtype=dtype,
                        )

                        (
                            perturbed_coefficients,
                            coefficient_norm,
                            realised_perturbation_norm,
                            realised_relative_perturbation,
                        ) = perturb_coefficients_with_shared_noise(
                            basis_coefficients,
                            relative_magnitude=perturbation_magnitude,
                            standard_normal_samples=shared_noise,
                            dtype=dtype,
                        )

                        perturbed_integral = (
                            integrate_in_basis_in_dtype(
                                perturbed_coefficients,
                                basis=basis,
                                scale=scale,
                                dtype=dtype,
                            )
                        )

                        sensitivity = perturbation_sensitivity(
                            original_integral=computed_integral,
                            perturbed_integral=perturbed_integral,
                            perturbation_magnitude=(
                                realised_relative_perturbation
                                if realised_relative_perturbation > 0.0
                                else perturbation_magnitude
                            ),
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
                                coefficient_norm,
                                realised_perturbation_norm,
                                realised_relative_perturbation,
                                perturbed_integral,
                                sensitivity,
                                runtime_seconds,
                            ],
                            dtype=np.float64,
                        )

                        if not np.isfinite(numeric_values).all():
                            raise FloatingPointError(
                                f"non-finite result for {basis}, "
                                f"{precision}, n={dimension}, d={degree}, "
                                f"trial={trial}"
                            )

                        record = PrecisionBenchmarkRecord(
                            pair_id=pair_id,
                            basis=basis,
                            precision=precision,
                            dtype_bits=np.finfo(dtype).bits,
                            dimension=dimension,
                            degree=degree,
                            basis_count=basis_count,
                            trial=trial,
                            seed=trial_seed,
                            point_seed=point_seed,
                            perturbation_seed=(
                                perturbation_seed
                                + basis_noise_offset
                            ),
                            scale=scale,
                            oversampling_factor=oversampling_factor,
                            num_points=num_points,
                            matrix_rows=matrix.shape[0],
                            matrix_columns=matrix.shape[1],
                            matrix_rank=matrix_rank,
                            sigma_max=sigma_max,
                            sigma_min=sigma_min,
                            condition_number=condition_number,
                            reference_integral=reference_integral,
                            computed_integral=computed_integral,
                            absolute_error=absolute_error,
                            relative_error=relative_error,
                            perturbation_magnitude=perturbation_magnitude,
                            coefficient_norm=coefficient_norm,
                            realised_perturbation_norm=(
                                realised_perturbation_norm
                            ),
                            realised_relative_perturbation=(
                                realised_relative_perturbation
                            ),
                            perturbed_integral=perturbed_integral,
                            perturbation_sensitivity=sensitivity,
                            runtime_seconds=runtime_seconds,
                        )
                        records.append(record)

                        print(
                            f"n={dimension} "
                            f"d={degree:2d} "
                            f"trial={trial} "
                            f"basis={basis:9s} "
                            f"dtype={precision:7s} "
                            f"M={basis_count:4d} "
                            f"rank={matrix_rank:4d} "
                            f"cond={condition_number:.3e} "
                            f"relerr={relative_error:.3e} "
                            f"sens={sensitivity:.3e} "
                            f"time={1e3 * runtime_seconds:.3f} ms"
                        )

    return records


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Compute a finite guarded ratio between paired measurements."""
    denominator_array = denominator.to_numpy(dtype=np.float64)
    numerator_array = numerator.to_numpy(dtype=np.float64)

    floor = np.finfo(np.float64).tiny
    ratio = numerator_array / np.maximum(
        np.abs(denominator_array),
        floor,
    )
    return pd.Series(ratio, index=numerator.index)


def add_paired_precision_metrics(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add float32-to-float64 ratios where both precision rows are available.

    Ratios are repeated on both rows of each pair to keep the detailed CSV convenient
    for filtering.
    """
    result = data.copy()

    ratio_columns = {
        "relative_error": "float32_to_float64_error_ratio",
        "absolute_error": "float32_to_float64_absolute_error_ratio",
        "condition_number": "float32_to_float64_condition_ratio",
        "perturbation_sensitivity": (
            "float32_to_float64_sensitivity_ratio"
        ),
        "runtime_seconds": "float32_to_float64_runtime_ratio",
    }

    for output_column in ratio_columns.values():
        result[output_column] = np.nan

    precision_set = set(result["precision"].unique())
    if not {"float32", "float64"}.issubset(precision_set):
        return result

    pair_keys = [
        "pair_id",
        "basis",
        "dimension",
        "degree",
        "trial",
    ]

    for _, group in result.groupby(pair_keys, sort=False):
        float32_rows = group[group["precision"] == "float32"]
        float64_rows = group[group["precision"] == "float64"]

        if len(float32_rows) != 1 or len(float64_rows) != 1:
            continue

        row32 = float32_rows.iloc[0]
        row64 = float64_rows.iloc[0]

        for metric, output_column in ratio_columns.items():
            numerator = float(row32[metric])
            denominator = max(
    abs(float(row64[metric])),
    np.finfo(np.float64).tiny,
)
            ratio = numerator / denominator
            result.loc[group.index, output_column] = ratio

    return result


def summarise_precision_results(
    data: pd.DataFrame,
) -> dict[str, object]:
    """Create aggregate precision results for JSON export."""
    grouped = (
        data.groupby(
            [
                "basis",
                "precision",
                "dimension",
                "degree",
            ],
            as_index=False,
        )
        .agg(
            trials=("trial", "nunique"),
            basis_count=("basis_count", "first"),
            minimum_rank=("matrix_rank", "min"),
            condition_median=("condition_number", "median"),
            condition_q1=(
                "condition_number",
                lambda values: values.quantile(0.25),
            ),
            condition_q3=(
                "condition_number",
                lambda values: values.quantile(0.75),
            ),
            absolute_error_median=("absolute_error", "median"),
            relative_error_median=("relative_error", "median"),
            relative_error_q1=(
                "relative_error",
                lambda values: values.quantile(0.25),
            ),
            relative_error_q3=(
                "relative_error",
                lambda values: values.quantile(0.75),
            ),
            sensitivity_median=(
                "perturbation_sensitivity",
                "median",
            ),
            runtime_median_seconds=("runtime_seconds", "median"),
            realised_relative_perturbation_median=(
                "realised_relative_perturbation",
                "median",
            ),
        )
        .sort_values(
            [
                "dimension",
                "degree",
                "basis",
                "precision",
            ]
        )
    )

    pair_summary: list[dict[str, object]] = []

    if {"float32", "float64"}.issubset(
        set(data["precision"].unique())
    ):
        paired = (
            data.groupby(
                ["basis", "dimension", "degree"],
                as_index=False,
            )
            .agg(
                float32_to_float64_error_ratio_median=(
                    "float32_to_float64_error_ratio",
                    "median",
                ),
                float32_to_float64_condition_ratio_median=(
                    "float32_to_float64_condition_ratio",
                    "median",
                ),
                float32_to_float64_sensitivity_ratio_median=(
                    "float32_to_float64_sensitivity_ratio",
                    "median",
                ),
                float32_to_float64_runtime_ratio_median=(
                    "float32_to_float64_runtime_ratio",
                    "median",
                ),
            )
            .sort_values(["dimension", "degree", "basis"])
        )
        pair_summary = paired.to_dict(orient="records")

    numeric_data = data.select_dtypes(include=[np.number]).to_numpy(
        dtype=np.float64
    )

    return {
        "record_count": int(len(data)),
        "non_finite_numeric_count": int(
            numeric_data.size
            - np.isfinite(numeric_data).sum()
        ),
        "bases": sorted(data["basis"].unique().tolist()),
        "precisions": sorted(
            data["precision"].unique().tolist()
        ),
        "dimensions": sorted(
            int(value)
            for value in data["dimension"].unique()
        ),
        "degrees_by_dimension": {
            str(int(dimension)): sorted(
                int(value)
                for value in subset["degree"].unique()
            )
            for dimension, subset in data.groupby("dimension")
        },
        "aggregates": grouped.to_dict(orient="records"),
        "paired_precision_ratios": pair_summary,
    }


def write_precision_outputs(
    records: Sequence[PrecisionBenchmarkRecord],
    *,
    output_path: Path,
    summary_path: Path,
) -> None:
    """Write detailed CSV and aggregate JSON outputs."""
    if not records:
        raise ValueError("cannot write an empty benchmark result")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.DataFrame(
        asdict(record)
        for record in records
    )

    base_columns = [
        field.name
        for field in fields(PrecisionBenchmarkRecord)
    ]
    data = data[base_columns]
    data = add_paired_precision_metrics(data)

    data.to_csv(output_path, index=False)

    summary = summarise_precision_results(data)
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
        description=(
            "Compare float32 and float64 arithmetic in the "
            "n-dimensional simplex benchmark."
        )
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Dimensions to evaluate. Default: 1 2 3.",
    )
    parser.add_argument(
        "--degrees",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional explicit degree list. Unsupported degrees are "
            "skipped separately for each dimension."
        ),
    )
    parser.add_argument(
        "--maximum-degree",
        type=int,
        default=None,
        help="Optionally remove degrees above this value.",
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=DEFAULT_PRECISIONS,
        default=list(DEFAULT_PRECISIONS),
        help="Floating-point precisions to compare.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help=(
            "Trials per basis-dimension-degree-precision "
            "configuration. Default: 10."
        ),
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
        help="Scaled-simplex size. Default: 1.0.",
    )
    parser.add_argument(
        "--oversampling-factor",
        type=float,
        default=2.0,
        help=(
            "Number of matrix rows divided by basis count. "
            "Must be at least 1."
        ),
    )
    parser.add_argument(
        "--perturbation-magnitude",
        type=float,
        default=DEFAULT_PERTURBATION_MAGNITUDE,
        help=(
            "Requested relative coefficient perturbation. "
            "Default: 1e-6."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/simplex/precision_results.csv"
        ),
        help="Detailed CSV output path.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/simplex/precision_summary.json"
        ),
        help="Aggregate JSON output path.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    grid = resolve_precision_grid(
        dimensions=args.dimensions,
        maximum_degree=args.maximum_degree,
        explicit_degrees=args.degrees,
    )
    precisions = validate_precisions(args.precisions)

    print("Simplex floating-point precision benchmark")
    print(f"dimensions: {list(grid)}")
    print(f"degree grid: {grid}")
    print(f"precisions: {list(precisions)}")
    print(f"trials: {args.trials}")
    print(f"scale: {args.scale}")
    print(f"oversampling factor: {args.oversampling_factor}")
    print(
        "requested perturbation magnitude: "
        f"{args.perturbation_magnitude}"
    )
    print(
        "reference arithmetic: float64 analytical monomial "
        "simplex integration"
    )
    print()

    records = run_precision_benchmark(
        grid=grid,
        precisions=precisions,
        trials=args.trials,
        seed=args.seed,
        scale=args.scale,
        oversampling_factor=args.oversampling_factor,
        perturbation_magnitude=args.perturbation_magnitude,
    )

    write_precision_outputs(
        records,
        output_path=args.output,
        summary_path=args.summary,
    )


if __name__ == "__main__":
    main()