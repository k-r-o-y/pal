from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from numpy.polynomial import Chebyshev, Legendre, Polynomial

from pal.experiments.common import (
    ensure_directory,
    environment_metadata,
    seed_everything,
    write_json,
)

BasisName = Literal["monomial", "legendre", "chebyshev"]
PrecisionName = Literal["float32", "float64"]
FamilyName = Literal["random_uniform", "alternating_decay"]


@dataclass(frozen=True)
class ExperimentConfig:
    degrees: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 16, 20)
    precisions: tuple[PrecisionName, ...] = ("float32", "float64")
    bases: tuple[BasisName, ...] = ("monomial", "legendre", "chebyshev")
    families: tuple[FamilyName, ...] = ("random_uniform", "alternating_decay")
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    grid_size: int = 2001
    domain_left: float = -1.0
    domain_right: float = 1.0
    reference_dps: int = 100
    repetitions: int = 5


@dataclass(frozen=True)
class ResultRow:
    experiment: str
    family: str
    seed: int
    degree: int
    basis: str
    precision: str
    grid_size: int
    domain_left: float
    domain_right: float
    max_abs_error: float
    rms_error: float
    max_relative_error: float
    coefficient_l2_norm: float
    evaluation_matrix_condition: float
    median_runtime_seconds: float
    status: str


def generate_monomial_coefficients(
    degree: int,
    family: FamilyName,
    seed: int,
) -> np.ndarray:
    """Return [a_0,...,a_d] for p(x)=sum_k a_k x^k."""
    rng = np.random.default_rng(seed)

    if family == "random_uniform":
        coefficients = rng.uniform(-1.0, 1.0, degree + 1)
    elif family == "alternating_decay":
        k = np.arange(degree + 1, dtype=np.float64)
        coefficients = ((-1.0) ** k) / (k + 1.0)
    else:
        raise ValueError(f"Unsupported coefficient family: {family}")

    if abs(coefficients[-1]) < 0.1:
        coefficients[-1] = math.copysign(0.1, coefficients[-1] or 1.0)

    return np.asarray(coefficients, dtype=np.float64)


def convert_from_monomial(
    coefficients: np.ndarray,
    basis: BasisName,
) -> np.ndarray:
    polynomial = Polynomial(np.asarray(coefficients, dtype=np.float64))

    if basis == "monomial":
        converted = polynomial.coef
    elif basis == "legendre":
        converted = polynomial.convert(kind=Legendre).coef
    elif basis == "chebyshev":
        converted = polynomial.convert(kind=Chebyshev).coef
    else:
        raise ValueError(f"Unsupported basis: {basis}")

    if len(converted) < len(coefficients):
        converted = np.pad(converted, (0, len(coefficients) - len(converted)))

    return np.asarray(converted[: len(coefficients)], dtype=np.float64)


def evaluate_monomial(
    coefficients: np.ndarray,
    x: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """Horner evaluation in the explicitly requested dtype."""
    c = np.asarray(coefficients, dtype=dtype)
    values = np.zeros_like(x, dtype=dtype)
    for coefficient in c[::-1]:
        values = values * x + coefficient
    return values


def evaluate_legendre(
    coefficients: np.ndarray,
    x: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """Legendre-series evaluation using the three-term recurrence."""
    c = np.asarray(coefficients, dtype=dtype)
    degree = len(c) - 1
    p0 = np.ones_like(x, dtype=dtype)
    values = c[0] * p0

    if degree == 0:
        return values

    p1 = x.astype(dtype, copy=False)
    values = values + c[1] * p1

    for n in range(1, degree):
        p2 = (
            dtype.type(2 * n + 1) * x * p1
            - dtype.type(n) * p0
        ) / dtype.type(n + 1)
        values = values + c[n + 1] * p2
        p0, p1 = p1, p2

    return values


def evaluate_chebyshev(
    coefficients: np.ndarray,
    x: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """Chebyshev-series evaluation using the three-term recurrence."""
    c = np.asarray(coefficients, dtype=dtype)
    degree = len(c) - 1
    t0 = np.ones_like(x, dtype=dtype)
    values = c[0] * t0

    if degree == 0:
        return values

    t1 = x.astype(dtype, copy=False)
    values = values + c[1] * t1

    for n in range(1, degree):
        t2 = dtype.type(2.0) * x * t1 - t0
        values = values + c[n + 1] * t2
        t0, t1 = t1, t2

    return values


def evaluator_for_basis(
    basis: BasisName,
) -> Callable[[np.ndarray, np.ndarray, np.dtype], np.ndarray]:
    if basis == "monomial":
        return evaluate_monomial
    if basis == "legendre":
        return evaluate_legendre
    if basis == "chebyshev":
        return evaluate_chebyshev
    raise ValueError(f"Unsupported basis: {basis}")


def high_precision_reference(
    monomial_coefficients: np.ndarray,
    x: np.ndarray,
    decimal_places: int,
) -> np.ndarray:
    """Evaluate the source polynomial using mpmath high precision."""
    mp.mp.dps = decimal_places
    coefficients_mp = [mp.mpf(str(value)) for value in monomial_coefficients]
    output = np.empty(len(x), dtype=np.float64)

    for index, point in enumerate(x):
        point_mp = mp.mpf(str(float(point)))
        value = mp.mpf("0")
        for coefficient in reversed(coefficients_mp):
            value = value * point_mp + coefficient
        output[index] = float(value)

    return output


def basis_evaluation_matrix(
    basis: BasisName,
    degree: int,
    x: np.ndarray,
) -> np.ndarray:
    """Matrix B where B[i,k] is basis function k evaluated at x_i."""
    matrix = np.empty((len(x), degree + 1), dtype=np.float64)
    matrix[:, 0] = 1.0

    if degree == 0:
        return matrix

    if basis == "monomial":
        for k in range(1, degree + 1):
            matrix[:, k] = matrix[:, k - 1] * x
        return matrix

    matrix[:, 1] = x

    if basis == "legendre":
        for n in range(1, degree):
            matrix[:, n + 1] = (
                (2 * n + 1) * x * matrix[:, n] - n * matrix[:, n - 1]
            ) / (n + 1)
        return matrix

    if basis == "chebyshev":
        for n in range(1, degree):
            matrix[:, n + 1] = (
                2.0 * x * matrix[:, n] - matrix[:, n - 1]
            )
        return matrix

    raise ValueError(f"Unsupported basis: {basis}")


def error_metrics(
    computed: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float, float]:
    difference = np.asarray(computed, dtype=np.float64) - reference
    absolute = np.abs(difference)

    max_abs = float(np.max(absolute))
    rms = float(np.sqrt(np.mean(np.square(difference))))

    denominator = np.maximum(
        np.abs(reference),
        np.finfo(np.float64).eps,
    )
    max_relative = float(np.max(absolute / denominator))
    return max_abs, rms, max_relative


def median_runtime(function: Callable[[], np.ndarray], repetitions: int) -> float:
    timings: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def run_experiment(
    config: ExperimentConfig,
    results_directory: Path,
    figures_directory: Path,
) -> list[ResultRow]:
    ensure_directory(results_directory)
    ensure_directory(figures_directory)

    x_reference = np.linspace(
        config.domain_left,
        config.domain_right,
        config.grid_size,
        dtype=np.float64,
    )

    rows: list[ResultRow] = []

    for family in config.families:
        for seed in config.seeds:
            seed_everything(seed)

            for degree in config.degrees:
                monomial_coefficients = generate_monomial_coefficients(
                    degree,
                    family,
                    seed,
                )
                reference = high_precision_reference(
                    monomial_coefficients,
                    x_reference,
                    config.reference_dps,
                )

                for basis in config.bases:
                    converted = convert_from_monomial(
                        monomial_coefficients,
                        basis,
                    )
                    condition = float(
                        np.linalg.cond(
                            basis_evaluation_matrix(
                                basis,
                                degree,
                                x_reference,
                            )
                        )
                    )
                    evaluator = evaluator_for_basis(basis)

                    for precision in config.precisions:
                        dtype = np.dtype(
                            np.float32
                            if precision == "float32"
                            else np.float64
                        )
                        x = x_reference.astype(dtype)

                        try:
                            computed = evaluator(converted, x, dtype)
                            max_abs, rms, max_relative = error_metrics(
                                computed,
                                reference,
                            )
                            runtime = median_runtime(
                                lambda: evaluator(converted, x, dtype),
                                config.repetitions,
                            )
                            status = "ok"
                        except (
                            FloatingPointError,
                            OverflowError,
                            ValueError,
                        ) as exc:
                            max_abs = float("nan")
                            rms = float("nan")
                            max_relative = float("nan")
                            runtime = float("nan")
                            status = (
                                f"failed: {type(exc).__name__}: {exc}"
                            )

                        rows.append(
                            ResultRow(
                                experiment="experiment1_polynomial_evaluation",
                                family=family,
                                seed=seed,
                                degree=degree,
                                basis=basis,
                                precision=precision,
                                grid_size=config.grid_size,
                                domain_left=config.domain_left,
                                domain_right=config.domain_right,
                                max_abs_error=max_abs,
                                rms_error=rms,
                                max_relative_error=max_relative,
                                coefficient_l2_norm=float(
                                    np.linalg.norm(converted)
                                ),
                                evaluation_matrix_condition=condition,
                                median_runtime_seconds=runtime,
                                status=status,
                            )
                        )

    csv_path = results_directory / "evaluation_results.csv"
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


def valid_rows(rows: list[ResultRow]) -> list[ResultRow]:
    return [
        row
        for row in rows
        if row.status == "ok"
        and np.isfinite(row.max_abs_error)
        and np.isfinite(row.rms_error)
    ]


def create_summary(
    rows: list[ResultRow],
    results_directory: Path,
) -> None:
    grouped: dict[tuple[str, str, int, str], list[ResultRow]] = {}

    for row in valid_rows(rows):
        key = (row.family, row.basis, row.degree, row.precision)
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []

    for (family, basis, degree, precision), group in sorted(grouped.items()):
        summary.append(
            {
                "family": family,
                "basis": basis,
                "degree": degree,
                "precision": precision,
                "runs": len(group),
                "median_max_abs_error": float(
                    np.median([r.max_abs_error for r in group])
                ),
                "median_rms_error": float(
                    np.median([r.rms_error for r in group])
                ),
                "median_max_relative_error": float(
                    np.median([r.max_relative_error for r in group])
                ),
                "median_runtime_seconds": float(
                    np.median(
                        [r.median_runtime_seconds for r in group]
                    )
                ),
                "median_condition_number": float(
                    np.median(
                        [r.evaluation_matrix_condition for r in group]
                    )
                ),
                "median_coefficient_l2_norm": float(
                    np.median(
                        [r.coefficient_l2_norm for r in group]
                    )
                ),
            }
        )

    write_json(
        results_directory / "evaluation_summary.json",
        summary,
    )


def save_plot(
    rows: list[ResultRow],
    figures_directory: Path,
    attribute: str,
    ylabel: str,
    filename: str,
) -> None:
    finite = valid_rows(rows)

    for precision in ("float32", "float64"):
        for family in sorted({row.family for row in finite}):
            plt.figure(figsize=(7.2, 4.8))

            for basis in ("monomial", "legendre", "chebyshev"):
                subset = [
                    row
                    for row in finite
                    if row.precision == precision
                    and row.family == family
                    and row.basis == basis
                ]
                degrees = sorted({row.degree for row in subset})
                values = [
                    float(
                        np.median(
                            [
                                getattr(row, attribute)
                                for row in subset
                                if row.degree == degree
                            ]
                        )
                    )
                    for degree in degrees
                ]
                plt.plot(degrees, values, marker="o", label=basis)

            plt.yscale("log")
            plt.xlabel("Polynomial degree")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel}: {family}, {precision}")
            plt.grid(True, which="both", alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                figures_directory
                / f"{filename}_{family}_{precision}.png",
                dpi=220,
            )
            plt.close()


def create_plots(
    rows: list[ResultRow],
    figures_directory: Path,
) -> None:
    save_plot(
        rows,
        figures_directory,
        "max_abs_error",
        "Maximum absolute error",
        "max_abs_error_vs_degree",
    )
    save_plot(
        rows,
        figures_directory,
        "rms_error",
        "RMS evaluation error",
        "rms_error_vs_degree",
    )
    save_plot(
        rows,
        figures_directory,
        "median_runtime_seconds",
        "Median evaluation runtime (seconds)",
        "runtime_vs_degree",
    )
    save_plot(
        rows,
        figures_directory,
        "evaluation_matrix_condition",
        "Evaluation-matrix condition number",
        "condition_number_vs_degree",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 1: polynomial evaluation stability."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced smoke-test configuration.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/experiment1"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures/experiment1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = (
        ExperimentConfig(
            degrees=(2, 6, 10),
            families=("random_uniform",),
            seeds=(0,),
            grid_size=501,
            reference_dps=80,
            repetitions=3,
        )
        if args.quick
        else ExperimentConfig()
    )

    rows = run_experiment(
        config,
        args.results_dir,
        args.figures_dir,
    )

    successful = sum(row.status == "ok" for row in rows)
    print(f"Completed {successful}/{len(rows)} runs.")
    print(
        "Results:",
        args.results_dir / "evaluation_results.csv",
    )
    print("Figures:", args.figures_dir)


if __name__ == "__main__":
    main()
