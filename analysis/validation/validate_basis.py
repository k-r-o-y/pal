#!/usr/bin/env python3
"""
Validate polynomial basis conversion, evaluation, and integration.

This script checks that the monomial, Legendre, and Chebyshev
representations describe the same mathematical polynomial over a specified
physical interval.

Validation performed
--------------------
1. Monomial -> Legendre conversion.
2. Monomial -> Chebyshev conversion.
3. Evaluation equivalence on a dense grid.
4. Analytical integration equivalence.
5. Repeated trials over several:
   - polynomial degrees;
   - physical domains;
   - floating-point precisions;
   - random coefficient vectors.

Outputs
-------
results/validation/basis_validation_results.csv
results/validation/basis_validation_summary.json

Example
-------
Run from the repository root:

    python -m analysis.validation.validate_basis

or:

    python analysis/validation/validate_basis.py

A non-zero exit status is returned when any validation check fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.polynomial import Chebyshev, Legendre, Polynomial
from numpy.typing import NDArray


# =============================================================================
# Repository import setup
# =============================================================================

THIS_FILE = Path(__file__).resolve()
REPOSITORY_ROOT = THIS_FILE.parents[2]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


# =============================================================================
# Import the project implementation
# =============================================================================

def _import_project_basis_functions():
    """
    Import basis utilities from the PAL repository.

    The repository screenshot shows basis utilities under both:

        pal.analysis.basis_polynomials
        distribution.basis_polynomials

    The first available module is used. If neither module is importable,
    this script falls back to local NumPy implementations so that the
    validation framework can still run.

    Returns
    -------
    tuple
        Imported module name and callable functions.
    """
    candidate_modules = (
        "pal.analysis.basis_polynomials",
        "analysis.basis_polynomials",
        "distribution.basis_polynomials",
    )

    last_error: Exception | None = None

    for module_name in candidate_modules:
        try:
            module = __import__(module_name, fromlist=["*"])

            required_names = (
                "monomial_eval",
                "legendre_eval",
                "chebyshev_eval",
                "convert_monomial_to_legendre",
                "convert_monomial_to_chebyshev",
            )

            missing = [
                name for name in required_names
                if not hasattr(module, name)
            ]

            if missing:
                raise AttributeError(
                    f"{module_name} is missing required functions: {missing}"
                )

            return {
                "module_name": module_name,
                "monomial_eval": module.monomial_eval,
                "legendre_eval": module.legendre_eval,
                "chebyshev_eval": module.chebyshev_eval,
                "convert_monomial_to_legendre":
                    module.convert_monomial_to_legendre,
                "convert_monomial_to_chebyshev":
                    module.convert_monomial_to_chebyshev,
                "integrate_monomial":
                    getattr(module, "integrate_monomial", None),
                "integrate_legendre":
                    getattr(module, "integrate_legendre", None),
                "integrate_chebyshev":
                    getattr(module, "integrate_chebyshev", None),
            }

        except (ImportError, AttributeError) as exc:
            last_error = exc

    print(
        "Warning: project basis module could not be imported. "
        "Using NumPy fallback functions.\n"
        f"Last import error: {last_error}",
        file=sys.stderr,
    )

    return {
        "module_name": "numpy_fallback",
        "monomial_eval": fallback_monomial_eval,
        "legendre_eval": fallback_legendre_eval,
        "chebyshev_eval": fallback_chebyshev_eval,
        "convert_monomial_to_legendre":
            fallback_convert_monomial_to_legendre,
        "convert_monomial_to_chebyshev":
            fallback_convert_monomial_to_chebyshev,
        "integrate_monomial": fallback_integrate_monomial,
        "integrate_legendre": fallback_integrate_legendre,
        "integrate_chebyshev": fallback_integrate_chebyshev,
    }


# =============================================================================
# NumPy fallback implementation
# =============================================================================

def fallback_monomial_eval(
    coeffs: Sequence[float],
    x: NDArray[np.floating] | float,
) -> NDArray[np.floating] | np.floating:
    """Evaluate monomial coefficients in ascending power order."""
    return Polynomial(coeffs)(x)


def fallback_legendre_eval(
    coeffs: Sequence[float],
    x: NDArray[np.floating] | float,
    lower: float = -1.0,
    upper: float = 1.0,
) -> NDArray[np.floating] | np.floating:
    """Evaluate a Legendre series defined over the physical domain."""
    return Legendre(coeffs, domain=[lower, upper])(x)


def fallback_chebyshev_eval(
    coeffs: Sequence[float],
    x: NDArray[np.floating] | float,
    lower: float = -1.0,
    upper: float = 1.0,
) -> NDArray[np.floating] | np.floating:
    """Evaluate a Chebyshev series defined over the physical domain."""
    return Chebyshev(coeffs, domain=[lower, upper])(x)


def fallback_convert_monomial_to_legendre(
    monomial_coeffs: Sequence[float],
    lower: float = -1.0,
    upper: float = 1.0,
) -> NDArray[np.float64]:
    """Convert monomial coefficients to a Legendre representation."""
    polynomial = Polynomial(
        monomial_coeffs,
        domain=[lower, upper],
        window=[lower, upper],
    )

    converted = polynomial.convert(
        kind=Legendre,
        domain=[lower, upper],
        window=[-1.0, 1.0],
    )

    return np.asarray(converted.coef, dtype=np.float64)


def fallback_convert_monomial_to_chebyshev(
    monomial_coeffs: Sequence[float],
    lower: float = -1.0,
    upper: float = 1.0,
) -> NDArray[np.float64]:
    """Convert monomial coefficients to a Chebyshev representation."""
    polynomial = Polynomial(
        monomial_coeffs,
        domain=[lower, upper],
        window=[lower, upper],
    )

    converted = polynomial.convert(
        kind=Chebyshev,
        domain=[lower, upper],
        window=[-1.0, 1.0],
    )

    return np.asarray(converted.coef, dtype=np.float64)


def fallback_integrate_monomial(
    coeffs: Sequence[float],
    lower: float,
    upper: float,
) -> float:
    """Integrate a monomial-basis polynomial analytically."""
    antiderivative = Polynomial(coeffs).integ()
    return float(antiderivative(upper) - antiderivative(lower))


def fallback_integrate_legendre(
    coeffs: Sequence[float],
    lower: float,
    upper: float,
) -> float:
    """Integrate a Legendre series over its full physical domain."""
    polynomial = Legendre(coeffs, domain=[lower, upper])
    antiderivative = polynomial.integ()
    return float(antiderivative(upper) - antiderivative(lower))


def fallback_integrate_chebyshev(
    coeffs: Sequence[float],
    lower: float,
    upper: float,
) -> float:
    """Integrate a Chebyshev series over its full physical domain."""
    polynomial = Chebyshev(coeffs, domain=[lower, upper])
    antiderivative = polynomial.integ()
    return float(antiderivative(upper) - antiderivative(lower))


PROJECT_FUNCTIONS = _import_project_basis_functions()


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class Domain:
    """Closed physical interval."""

    lower: float
    upper: float

    def validate(self) -> None:
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError("Domain bounds must be finite.")

        if self.upper <= self.lower:
            raise ValueError(
                f"Invalid domain [{self.lower}, {self.upper}]. "
                "The upper bound must exceed the lower bound."
            )

    @property
    def label(self) -> str:
        return f"[{self.lower:g}, {self.upper:g}]"


@dataclass
class ValidationRecord:
    """One basis-validation result."""

    trial: int
    seed: int
    degree: int
    dtype: str
    lower: float
    upper: float
    basis: str
    coefficient_scale: float
    evaluation_absolute_error: float
    evaluation_relative_error: float
    evaluation_rms_error: float
    integral_reference: float
    integral_estimate: float
    integral_absolute_error: float
    integral_relative_error: float
    evaluation_tolerance: float
    integral_tolerance: float
    evaluation_passed: bool
    integral_passed: bool

    @property
    def passed(self) -> bool:
        return self.evaluation_passed and self.integral_passed


@dataclass
class ValidationSummary:
    """Overall validation summary."""

    implementation_module: str
    total_records: int
    passed_records: int
    failed_records: int
    maximum_evaluation_absolute_error: float
    maximum_evaluation_relative_error: float
    maximum_evaluation_rms_error: float
    maximum_integral_absolute_error: float
    maximum_integral_relative_error: float
    all_passed: bool


# =============================================================================
# Numerical helpers
# =============================================================================

def safe_relative_error(
    estimate: NDArray[np.floating] | float,
    reference: NDArray[np.floating] | float,
    floor: float = 1.0e-30,
) -> NDArray[np.float64] | float:
    """
    Compute a relative error with a safe denominator.

    The denominator is

        max(abs(reference), floor)

    to avoid division by zero.
    """
    estimate_array = np.asarray(estimate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)

    denominator = np.maximum(np.abs(reference_array), floor)
    result = np.abs(estimate_array - reference_array) / denominator

    if result.ndim == 0:
        return float(result)

    return result


def exact_monomial_integral(
    coefficients: Sequence[float],
    lower: float,
    upper: float,
) -> float:
    """
    Compute the analytical monomial integral.

    Coefficients use ascending power order:

        coefficients[k] multiplies x**k.
    """
    total = np.longdouble(0.0)
    lower_ld = np.longdouble(lower)
    upper_ld = np.longdouble(upper)

    for power, coefficient in enumerate(coefficients):
        exponent = power + 1
        contribution = (
            np.longdouble(coefficient)
            * (
                upper_ld ** exponent
                - lower_ld ** exponent
            )
            / np.longdouble(exponent)
        )
        total += contribution

    return float(total)


def generate_coefficients(
    rng: np.random.Generator,
    degree: int,
    coefficient_scale: float,
    dtype: np.dtype,
) -> NDArray[np.floating]:
    """
    Generate a random coefficient vector in ascending power order.

    Coefficients are scaled by 1 / (k + 1) to reduce the probability that
    high-degree terms completely dominate low-degree terms. The explicit
    coefficient_scale parameter still permits controlled stress testing.
    """
    coefficients = rng.normal(
        loc=0.0,
        scale=coefficient_scale,
        size=degree + 1,
    )

    decay = np.arange(1, degree + 2, dtype=np.float64)
    coefficients = coefficients / decay

    # Avoid a polynomial whose integral and values are both almost exactly zero.
    coefficients[0] += 1.0

    return np.asarray(coefficients, dtype=dtype)


def evaluation_tolerance_for(
    dtype: np.dtype,
    degree: int,
    domain: Domain,
) -> float:
    """
    Return a practical evaluation tolerance.

    The tolerance increases mildly with degree and physical-domain magnitude.
    """
    finfo = np.finfo(dtype)
    scale = max(1.0, abs(domain.lower), abs(domain.upper))
    degree_factor = max(1.0, float((degree + 1) ** 2))

    if dtype == np.dtype(np.float32):
        base = 2.0e-5
    else:
        base = 5.0e-11

    return max(
        base * degree_factor * math.sqrt(scale),
        100.0 * finfo.eps,
    )


def integral_tolerance_for(
    dtype: np.dtype,
    degree: int,
    domain: Domain,
) -> float:
    """
    Return a practical integration tolerance.

    Integration can accumulate more error than pointwise evaluation, so this
    tolerance is slightly larger.
    """
    evaluation_tolerance = evaluation_tolerance_for(
        dtype=dtype,
        degree=degree,
        domain=domain,
    )

    width = domain.upper - domain.lower
    return evaluation_tolerance * max(2.0, width)


def evaluate_project_monomial(
    coefficients: NDArray[np.floating],
    x: NDArray[np.floating],
) -> NDArray[np.float64]:
    values = PROJECT_FUNCTIONS["monomial_eval"](coefficients, x)
    return np.asarray(values, dtype=np.float64)


def evaluate_project_legendre(
    coefficients: NDArray[np.floating],
    x: NDArray[np.floating],
    domain: Domain,
) -> NDArray[np.float64]:
    function = PROJECT_FUNCTIONS["legendre_eval"]

    try:
        values = function(
            coefficients,
            x,
            lower=domain.lower,
            upper=domain.upper,
        )
    except TypeError:
        values = function(
            coefficients,
            x,
            domain.lower,
            domain.upper,
        )

    return np.asarray(values, dtype=np.float64)


def evaluate_project_chebyshev(
    coefficients: NDArray[np.floating],
    x: NDArray[np.floating],
    domain: Domain,
) -> NDArray[np.float64]:
    function = PROJECT_FUNCTIONS["chebyshev_eval"]

    try:
        values = function(
            coefficients,
            x,
            lower=domain.lower,
            upper=domain.upper,
        )
    except TypeError:
        values = function(
            coefficients,
            x,
            domain.lower,
            domain.upper,
        )

    return np.asarray(values, dtype=np.float64)


def convert_to_legendre(
    coefficients: NDArray[np.floating],
    domain: Domain,
) -> NDArray[np.float64]:
    function = PROJECT_FUNCTIONS["convert_monomial_to_legendre"]

    try:
        converted = function(
            coefficients,
            lower=domain.lower,
            upper=domain.upper,
        )
    except TypeError:
        converted = function(
            coefficients,
            domain.lower,
            domain.upper,
        )

    return np.asarray(converted, dtype=np.float64)


def convert_to_chebyshev(
    coefficients: NDArray[np.floating],
    domain: Domain,
) -> NDArray[np.float64]:
    function = PROJECT_FUNCTIONS["convert_monomial_to_chebyshev"]

    try:
        converted = function(
            coefficients,
            lower=domain.lower,
            upper=domain.upper,
        )
    except TypeError:
        converted = function(
            coefficients,
            domain.lower,
            domain.upper,
        )

    return np.asarray(converted, dtype=np.float64)


def integrate_project_basis(
    basis: str,
    coefficients: NDArray[np.floating],
    domain: Domain,
) -> float:
    """
    Integrate a basis representation using the project function when present.

    If the corresponding project integration utility is unavailable, NumPy's
    polynomial classes are used as an independent fallback.
    """
    function_key = f"integrate_{basis}"
    function = PROJECT_FUNCTIONS.get(function_key)

    if callable(function):
        try:
            return float(
                function(
                    coefficients,
                    lower=domain.lower,
                    upper=domain.upper,
                )
            )
        except TypeError:
            return float(
                function(
                    coefficients,
                    domain.lower,
                    domain.upper,
                )
            )

    if basis == "monomial":
        return fallback_integrate_monomial(
            coefficients,
            domain.lower,
            domain.upper,
        )

    if basis == "legendre":
        return fallback_integrate_legendre(
            coefficients,
            domain.lower,
            domain.upper,
        )

    if basis == "chebyshev":
        return fallback_integrate_chebyshev(
            coefficients,
            domain.lower,
            domain.upper,
        )

    raise ValueError(f"Unsupported basis: {basis}")


# =============================================================================
# Validation
# =============================================================================

def validate_single_basis(
    *,
    basis: str,
    original_coefficients: NDArray[np.floating],
    converted_coefficients: NDArray[np.floating],
    evaluation_points: NDArray[np.floating],
    reference_values: NDArray[np.float64],
    domain: Domain,
    dtype: np.dtype,
    degree: int,
    trial: int,
    seed: int,
    coefficient_scale: float,
) -> ValidationRecord:
    """Validate one converted basis representation."""
    if basis == "legendre":
        estimated_values = evaluate_project_legendre(
            converted_coefficients,
            evaluation_points,
            domain,
        )
    elif basis == "chebyshev":
        estimated_values = evaluate_project_chebyshev(
            converted_coefficients,
            evaluation_points,
            domain,
        )
    else:
        raise ValueError(f"Unsupported converted basis: {basis}")

    absolute_errors = np.abs(estimated_values - reference_values)
    relative_errors = safe_relative_error(
        estimated_values,
        reference_values,
    )

    maximum_absolute_error = float(np.max(absolute_errors))
    maximum_relative_error = float(np.max(relative_errors))
    rms_error = float(
        np.sqrt(
            np.mean(
                np.square(estimated_values - reference_values)
            )
        )
    )

    reference_integral = exact_monomial_integral(
        original_coefficients,
        domain.lower,
        domain.upper,
    )

    estimated_integral = integrate_project_basis(
        basis,
        converted_coefficients,
        domain,
    )

    integral_absolute_error = abs(
        estimated_integral - reference_integral
    )

    integral_relative_error = safe_relative_error(
        estimated_integral,
        reference_integral,
    )

    evaluation_tolerance = evaluation_tolerance_for(
        dtype=dtype,
        degree=degree,
        domain=domain,
    )

    integral_tolerance = integral_tolerance_for(
        dtype=dtype,
        degree=degree,
        domain=domain,
    )

    reference_value_scale = max(
        1.0,
        float(np.max(np.abs(reference_values))),
    )

    integral_scale = max(
        1.0,
        abs(reference_integral),
    )

    evaluation_passed = (
        np.all(np.isfinite(estimated_values))
        and maximum_absolute_error
        <= evaluation_tolerance * reference_value_scale
    )

    integral_passed = (
        np.isfinite(estimated_integral)
        and integral_absolute_error
        <= integral_tolerance * integral_scale
    )

    return ValidationRecord(
        trial=trial,
        seed=seed,
        degree=degree,
        dtype=np.dtype(dtype).name,
        lower=domain.lower,
        upper=domain.upper,
        basis=basis,
        coefficient_scale=coefficient_scale,
        evaluation_absolute_error=maximum_absolute_error,
        evaluation_relative_error=maximum_relative_error,
        evaluation_rms_error=rms_error,
        integral_reference=reference_integral,
        integral_estimate=estimated_integral,
        integral_absolute_error=integral_absolute_error,
        integral_relative_error=float(integral_relative_error),
        evaluation_tolerance=evaluation_tolerance,
        integral_tolerance=integral_tolerance,
        evaluation_passed=evaluation_passed,
        integral_passed=integral_passed,
    )


def validate_monomial_reference(
    *,
    coefficients: NDArray[np.floating],
    evaluation_points: NDArray[np.floating],
    domain: Domain,
    dtype: np.dtype,
    degree: int,
    trial: int,
    seed: int,
    coefficient_scale: float,
) -> ValidationRecord:
    """
    Validate the project's monomial evaluation and integration functions.

    NumPy's Polynomial evaluator and an explicit analytical integral are used
    as independent references.
    """
    reference_values = np.asarray(
        Polynomial(np.asarray(coefficients, dtype=np.float64))(
            np.asarray(evaluation_points, dtype=np.float64)
        ),
        dtype=np.float64,
    )

    estimated_values = evaluate_project_monomial(
        coefficients,
        evaluation_points,
    )

    absolute_errors = np.abs(estimated_values - reference_values)
    relative_errors = safe_relative_error(
        estimated_values,
        reference_values,
    )

    maximum_absolute_error = float(np.max(absolute_errors))
    maximum_relative_error = float(np.max(relative_errors))
    rms_error = float(
        np.sqrt(
            np.mean(
                np.square(estimated_values - reference_values)
            )
        )
    )

    reference_integral = exact_monomial_integral(
        coefficients,
        domain.lower,
        domain.upper,
    )

    estimated_integral = integrate_project_basis(
        "monomial",
        coefficients,
        domain,
    )

    integral_absolute_error = abs(
        estimated_integral - reference_integral
    )

    integral_relative_error = safe_relative_error(
        estimated_integral,
        reference_integral,
    )

    evaluation_tolerance = evaluation_tolerance_for(
        dtype=dtype,
        degree=degree,
        domain=domain,
    )

    integral_tolerance = integral_tolerance_for(
        dtype=dtype,
        degree=degree,
        domain=domain,
    )

    reference_value_scale = max(
        1.0,
        float(np.max(np.abs(reference_values))),
    )

    integral_scale = max(
        1.0,
        abs(reference_integral),
    )

    evaluation_passed = (
        np.all(np.isfinite(estimated_values))
        and maximum_absolute_error
        <= evaluation_tolerance * reference_value_scale
    )

    integral_passed = (
        np.isfinite(estimated_integral)
        and integral_absolute_error
        <= integral_tolerance * integral_scale
    )

    return ValidationRecord(
        trial=trial,
        seed=seed,
        degree=degree,
        dtype=np.dtype(dtype).name,
        lower=domain.lower,
        upper=domain.upper,
        basis="monomial",
        coefficient_scale=coefficient_scale,
        evaluation_absolute_error=maximum_absolute_error,
        evaluation_relative_error=maximum_relative_error,
        evaluation_rms_error=rms_error,
        integral_reference=reference_integral,
        integral_estimate=estimated_integral,
        integral_absolute_error=integral_absolute_error,
        integral_relative_error=float(integral_relative_error),
        evaluation_tolerance=evaluation_tolerance,
        integral_tolerance=integral_tolerance,
        evaluation_passed=evaluation_passed,
        integral_passed=integral_passed,
    )


def run_validation(
    *,
    degrees: Sequence[int],
    domains: Sequence[Domain],
    dtypes: Sequence[np.dtype],
    trials: int,
    grid_size: int,
    seed: int,
    coefficient_scale: float,
) -> list[ValidationRecord]:
    """Run all requested basis validation cases."""
    if trials <= 0:
        raise ValueError("trials must be positive.")

    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")

    if coefficient_scale <= 0.0:
        raise ValueError("coefficient_scale must be positive.")

    for degree in degrees:
        if degree < 0:
            raise ValueError("Polynomial degrees must be non-negative.")

    for domain in domains:
        domain.validate()

    records: list[ValidationRecord] = []

    for dtype_index, dtype in enumerate(dtypes):
        dtype = np.dtype(dtype)

        for domain_index, domain in enumerate(domains):
            points = np.linspace(
                domain.lower,
                domain.upper,
                grid_size,
                dtype=dtype,
            )

            for degree in degrees:
                for trial in range(trials):
                    trial_seed = (
                        seed
                        + 1_000_000 * dtype_index
                        + 10_000 * domain_index
                        + 100 * degree
                        + trial
                    )

                    rng = np.random.default_rng(trial_seed)

                    coefficients = generate_coefficients(
                        rng=rng,
                        degree=degree,
                        coefficient_scale=coefficient_scale,
                        dtype=dtype,
                    )

                    # Validate the baseline monomial implementation.
                    monomial_record = validate_monomial_reference(
                        coefficients=coefficients,
                        evaluation_points=points,
                        domain=domain,
                        dtype=dtype,
                        degree=degree,
                        trial=trial,
                        seed=trial_seed,
                        coefficient_scale=coefficient_scale,
                    )
                    records.append(monomial_record)

                    reference_values = evaluate_project_monomial(
                        coefficients,
                        points,
                    )

                    # Convert the same mathematical polynomial.
                    legendre_coefficients = convert_to_legendre(
                        coefficients,
                        domain,
                    ).astype(dtype, copy=False)

                    chebyshev_coefficients = convert_to_chebyshev(
                        coefficients,
                        domain,
                    ).astype(dtype, copy=False)

                    legendre_record = validate_single_basis(
                        basis="legendre",
                        original_coefficients=coefficients,
                        converted_coefficients=legendre_coefficients,
                        evaluation_points=points,
                        reference_values=reference_values,
                        domain=domain,
                        dtype=dtype,
                        degree=degree,
                        trial=trial,
                        seed=trial_seed,
                        coefficient_scale=coefficient_scale,
                    )
                    records.append(legendre_record)

                    chebyshev_record = validate_single_basis(
                        basis="chebyshev",
                        original_coefficients=coefficients,
                        converted_coefficients=chebyshev_coefficients,
                        evaluation_points=points,
                        reference_values=reference_values,
                        domain=domain,
                        dtype=dtype,
                        degree=degree,
                        trial=trial,
                        seed=trial_seed,
                        coefficient_scale=coefficient_scale,
                    )
                    records.append(chebyshev_record)

    return records


# =============================================================================
# Reporting
# =============================================================================

def create_summary(
    records: Sequence[ValidationRecord],
) -> ValidationSummary:
    """Create an aggregate validation summary."""
    if not records:
        return ValidationSummary(
            implementation_module=PROJECT_FUNCTIONS["module_name"],
            total_records=0,
            passed_records=0,
            failed_records=0,
            maximum_evaluation_absolute_error=0.0,
            maximum_evaluation_relative_error=0.0,
            maximum_evaluation_rms_error=0.0,
            maximum_integral_absolute_error=0.0,
            maximum_integral_relative_error=0.0,
            all_passed=False,
        )

    passed_records = sum(record.passed for record in records)
    total_records = len(records)

    return ValidationSummary(
        implementation_module=PROJECT_FUNCTIONS["module_name"],
        total_records=total_records,
        passed_records=passed_records,
        failed_records=total_records - passed_records,
        maximum_evaluation_absolute_error=max(
            record.evaluation_absolute_error
            for record in records
        ),
        maximum_evaluation_relative_error=max(
            record.evaluation_relative_error
            for record in records
        ),
        maximum_evaluation_rms_error=max(
            record.evaluation_rms_error
            for record in records
        ),
        maximum_integral_absolute_error=max(
            record.integral_absolute_error
            for record in records
        ),
        maximum_integral_relative_error=max(
            record.integral_relative_error
            for record in records
        ),
        all_passed=passed_records == total_records,
    )


def write_csv(
    records: Sequence[ValidationRecord],
    output_path: Path,
) -> None:
    """Write detailed validation records to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        raise ValueError("Cannot write an empty validation result set.")

    fieldnames = list(asdict(records[0]).keys()) + ["passed"]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            row = asdict(record)
            row["passed"] = record.passed
            writer.writerow(row)


def write_json(
    summary: ValidationSummary,
    output_path: Path,
    configuration: dict,
) -> None:
    """Write aggregate results and configuration to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": asdict(summary),
        "configuration": configuration,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
        )


def print_table(
    records: Sequence[ValidationRecord],
) -> None:
    """Print a compact aggregate table."""
    grouped: dict[tuple[str, str, str], list[ValidationRecord]] = {}

    for record in records:
        key = (
            record.dtype,
            f"[{record.lower:g}, {record.upper:g}]",
            record.basis,
        )
        grouped.setdefault(key, []).append(record)

    header = (
        f"{'dtype':<10}"
        f"{'domain':<18}"
        f"{'basis':<12}"
        f"{'max eval abs':>16}"
        f"{'max int abs':>16}"
        f"{'passed':>12}"
    )

    print()
    print(header)
    print("-" * len(header))

    for key in sorted(grouped):
        dtype_name, domain_label, basis = key
        group = grouped[key]

        maximum_evaluation_error = max(
            record.evaluation_absolute_error
            for record in group
        )
        maximum_integral_error = max(
            record.integral_absolute_error
            for record in group
        )
        passed = sum(record.passed for record in group)

        print(
            f"{dtype_name:<10}"
            f"{domain_label:<18}"
            f"{basis:<12}"
            f"{maximum_evaluation_error:>16.6e}"
            f"{maximum_integral_error:>16.6e}"
            f"{passed:>6}/{len(group):<5}"
        )


def print_failures(
    records: Sequence[ValidationRecord],
    maximum_to_show: int = 20,
) -> None:
    """Print individual failed validation cases."""
    failures = [
        record for record in records
        if not record.passed
    ]

    if not failures:
        return

    print("\nFailed validation cases:")

    for record in failures[:maximum_to_show]:
        print(
            "  "
            f"basis={record.basis}, "
            f"dtype={record.dtype}, "
            f"degree={record.degree}, "
            f"domain=[{record.lower:g}, {record.upper:g}], "
            f"trial={record.trial}, "
            f"eval_abs={record.evaluation_absolute_error:.6e}, "
            f"int_abs={record.integral_absolute_error:.6e}, "
            f"eval_pass={record.evaluation_passed}, "
            f"int_pass={record.integral_passed}"
        )

    if len(failures) > maximum_to_show:
        print(
            f"  ... and {len(failures) - maximum_to_show} "
            "additional failures."
        )


# =============================================================================
# CLI
# =============================================================================

def parse_domain(value: str) -> Domain:
    """
    Parse a domain supplied as LOWER:UPPER.

    Examples
    --------
    -1:1
    0:1
    -10:10
    """
    pieces = value.split(":")

    if len(pieces) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid domain '{value}'. Expected LOWER:UPPER."
        )

    try:
        domain = Domain(
            lower=float(pieces[0]),
            upper=float(pieces[1]),
        )
        domain.validate()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc

    return domain


def parse_dtype(value: str) -> np.dtype:
    """Parse an allowed NumPy floating-point dtype."""
    allowed = {
        "float32": np.dtype(np.float32),
        "float64": np.dtype(np.float64),
    }

    try:
        return allowed[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "dtype must be float32 or float64."
        ) from exc


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate monomial, Legendre, and Chebyshev polynomial "
            "representations."
        )
    )

    parser.add_argument(
        "--degrees",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 5, 8, 12],
        help="Polynomial degrees to validate.",
    )

    parser.add_argument(
        "--domains",
        nargs="+",
        type=parse_domain,
        default=[
            Domain(-1.0, 1.0),
            Domain(0.0, 1.0),
            Domain(-10.0, 10.0),
        ],
        help=(
            "Physical domains written as LOWER:UPPER. "
            "Example: --domains=-1:1 0:1 -10:10"
        ),
    )

    parser.add_argument(
        "--dtypes",
        nargs="+",
        type=parse_dtype,
        default=[
            np.dtype(np.float64),
            np.dtype(np.float32),
        ],
        help="Floating-point types to validate.",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of random coefficient vectors per configuration.",
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=1001,
        help="Number of evaluation points per polynomial.",
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
        help="Standard deviation used to generate coefficients.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/validation"),
        help="Directory for CSV and JSON outputs.",
    )

    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return exit code zero even if validation checks fail.",
    )

    return parser


def main() -> int:
    """Run basis validation from the command line."""
    parser = build_argument_parser()
    arguments = parser.parse_args()

    print(
        "Basis implementation module: "
        f"{PROJECT_FUNCTIONS['module_name']}"
    )

    print(
        "Running validation with "
        f"{len(arguments.degrees)} degrees, "
        f"{len(arguments.domains)} domains, "
        f"{len(arguments.dtypes)} dtypes, "
        f"{arguments.trials} trials."
    )

    records = run_validation(
        degrees=arguments.degrees,
        domains=arguments.domains,
        dtypes=arguments.dtypes,
        trials=arguments.trials,
        grid_size=arguments.grid_size,
        seed=arguments.seed,
        coefficient_scale=arguments.coefficient_scale,
    )

    summary = create_summary(records)

    output_directory = arguments.output_directory.resolve()
    csv_path = output_directory / "basis_validation_results.csv"
    json_path = output_directory / "basis_validation_summary.json"

    configuration = {
        "degrees": arguments.degrees,
        "domains": [
            {
                "lower": domain.lower,
                "upper": domain.upper,
            }
            for domain in arguments.domains
        ],
        "dtypes": [
            np.dtype(dtype).name
            for dtype in arguments.dtypes
        ],
        "trials": arguments.trials,
        "grid_size": arguments.grid_size,
        "seed": arguments.seed,
        "coefficient_scale": arguments.coefficient_scale,
        "implementation_module":
            PROJECT_FUNCTIONS["module_name"],
    }

    write_csv(
        records=records,
        output_path=csv_path,
    )

    write_json(
        summary=summary,
        output_path=json_path,
        configuration=configuration,
    )

    print_table(records)
    print_failures(records)

    print("\nValidation summary")
    print("------------------")
    print(f"Total records:  {summary.total_records}")
    print(f"Passed records: {summary.passed_records}")
    print(f"Failed records: {summary.failed_records}")
    print(
        "Maximum evaluation absolute error: "
        f"{summary.maximum_evaluation_absolute_error:.6e}"
    )
    print(
        "Maximum integral absolute error:   "
        f"{summary.maximum_integral_absolute_error:.6e}"
    )
    print(f"\nDetailed CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")

    if summary.all_passed:
        print("\nRESULT: ALL BASIS VALIDATION CHECKS PASSED")
        return 0

    print("\nRESULT: ONE OR MORE BASIS VALIDATION CHECKS FAILED")

    if arguments.allow_failures:
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())