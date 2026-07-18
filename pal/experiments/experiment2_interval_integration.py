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
FamilyName = Literal[
    "random_uniform",
    "alternating_decay",
    "sparse_high_degree",
]


@dataclass(frozen=True)
class IntervalConfig:
    degrees: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 16, 20)
    intervals: tuple[tuple[float, float], ...] = (
        (-1.0, 1.0),
        (0.0, 1.0),
        (0.0, 10.0),
        (10.0, 20.0),
        (-100.0, 100.0),
    )
    precisions: tuple[PrecisionName, ...] = ("float32", "float64")
    bases: tuple[BasisName, ...] = ("monomial", "legendre", "chebyshev")
    families: tuple[FamilyName, ...] = (
        "random_uniform",
        "alternating_decay",
        "sparse_high_degree",
    )
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    reference_dps: int = 100
    repetitions: int = 5


@dataclass(frozen=True)
class IntervalResult:
    experiment: str
    family: str
    seed: int
    degree: int
    basis: str
    precision: str
    interval_left: float
    interval_right: float
    interval_width: float
    interval_midpoint: float
    reference_integral: float
    computed_integral: float
    absolute_error: float
    relative_error: float
    coefficient_l2_norm: float
    median_runtime_seconds: float
    status: str


def generate_monomial_coefficients(
    degree: int,
    family: FamilyName,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if family == "random_uniform":
        coefficients = rng.uniform(-1.0, 1.0, degree + 1)
    elif family == "alternating_decay":
        k = np.arange(degree + 1, dtype=np.float64)
        coefficients = ((-1.0) ** k) / (k + 1.0)
    elif family == "sparse_high_degree":
        coefficients = np.zeros(degree + 1, dtype=np.float64)
        coefficients[0] = 1.0
        if degree >= 1:
            coefficients[1] = -0.25
        coefficients[-1] = 0.5
    else:
        raise ValueError(f"Unsupported family: {family}")

    if abs(coefficients[-1]) < 0.1:
        coefficients[-1] = math.copysign(0.1, coefficients[-1] or 1.0)

    return np.asarray(coefficients, dtype=np.float64)


def affine_map_to_standard_domain(
    left: float,
    right: float,
) -> Polynomial:
    """
    Return x(t) mapping t in [-1,1] to x in [left,right].
    """
    midpoint = (left + right) / 2.0
    half_width = (right - left) / 2.0
    return Polynomial([midpoint, half_width])


def coefficients_for_basis(
    monomial_coefficients: np.ndarray,
    left: float,
    right: float,
    basis: BasisName,
) -> np.ndarray:
    """
    For orthogonal bases, represent q(t)=p(x(t)) on t in [-1,1].

    The monomial baseline remains in the physical x-coordinate.
    """
    polynomial = Polynomial(monomial_coefficients)

    if basis == "monomial":
        return np.asarray(polynomial.coef, dtype=np.float64)

    transformed = polynomial(affine_map_to_standard_domain(left, right))

    if basis == "legendre":
        return np.asarray(
            transformed.convert(kind=Legendre).coef,
            dtype=np.float64,
        )
    if basis == "chebyshev":
        return np.asarray(
            transformed.convert(kind=Chebyshev).coef,
            dtype=np.float64,
        )

    raise ValueError(f"Unsupported basis: {basis}")


def integrate_monomial(
    coefficients: np.ndarray,
    left: float,
    right: float,
    dtype: np.dtype,
) -> float:
    """
    Integrate sum_k a_k x^k directly in the requested dtype.
    """
    c = np.asarray(coefficients, dtype=dtype)
    left_t = dtype.type(left)
    right_t = dtype.type(right)
    total = dtype.type(0.0)

    for degree, coefficient in enumerate(c):
        power = degree + 1
        total += (
            coefficient
            * (right_t ** power - left_t ** power)
            / dtype.type(power)
        )

    return float(total)


def integrate_legendre(
    coefficients: np.ndarray,
    left: float,
    right: float,
    dtype: np.dtype,
) -> float:
    """
    Integrate a Legendre series represented on t in [-1,1].

    Only the degree-zero Legendre term has a non-zero integral over [-1,1]:
        integral_{-1}^{1} P_0(t) dt = 2.

    The physical-domain Jacobian is (right-left)/2.
    """
    c = np.asarray(coefficients, dtype=dtype)
    width = dtype.type(right - left)
    return float(width * c[0])


def integrate_chebyshev(
    coefficients: np.ndarray,
    left: float,
    right: float,
    dtype: np.dtype,
) -> float:
    """
    Integrate a Chebyshev series represented on t in [-1,1].

    For n >= 1:
      integral_{-1}^{1} T_n(t) dt = 0 for odd n
      integral_{-1}^{1} T_n(t) dt = 2/(1-n^2) for even n

    The physical-domain Jacobian is (right-left)/2.
    """
    c = np.asarray(coefficients, dtype=dtype)
    total = dtype.type(0.0)

    for degree, coefficient in enumerate(c):
        if degree == 0:
            basis_integral = dtype.type(2.0)
        elif degree % 2 == 1:
            basis_integral = dtype.type(0.0)
        else:
            basis_integral = dtype.type(2.0 / (1.0 - degree * degree))

        total += coefficient * basis_integral

    jacobian = dtype.type((right - left) / 2.0)
    return float(jacobian * total)


def integrator_for_basis(
    basis: BasisName,
) -> Callable[[np.ndarray, float, float, np.dtype], float]:
    if basis == "monomial":
        return integrate_monomial
    if basis == "legendre":
        return integrate_legendre
    if basis == "chebyshev":
        return integrate_chebyshev
    raise ValueError(f"Unsupported basis: {basis}")


def high_precision_reference_integral(
    monomial_coefficients: np.ndarray,
    left: float,
    right: float,
    decimal_places: int,
) -> float:
    mp.mp.dps = decimal_places

    left_mp = mp.mpf(str(left))
    right_mp = mp.mpf(str(right))
    total = mp.mpf("0")

    for degree, coefficient in enumerate(monomial_coefficients):
        coefficient_mp = mp.mpf(str(coefficient))
        power = degree + 1
        total += (
            coefficient_mp
            * (right_mp ** power - left_mp ** power)
            / power
        )

    return float(total)


def error_metrics(
    computed: float,
    reference: float,
) -> tuple[float, float]:
    absolute_error = abs(computed - reference)
    relative_error = absolute_error / max(
        abs(reference),
        np.finfo(np.float64).eps,
    )
    return absolute_error, relative_error


def median_runtime(
    function: Callable[[], float],
    repetitions: int,
) -> float:
    timings: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def run_interval_experiment(
    config: IntervalConfig,
    results_directory: Path,
    figures_directory: Path,
) -> list[IntervalResult]:
    ensure_directory(results_directory)
    ensure_directory(figures_directory)

    rows: list[IntervalResult] = []

    for family in config.families:
        for seed in config.seeds:
            seed_everything(seed)

            for degree in config.degrees:
                source_coefficients = generate_monomial_coefficients(
                    degree=degree,
                    family=family,
                    seed=seed,
                )

                for left, right in config.intervals:
                    if not left < right:
                        raise ValueError(
                            f"Invalid interval [{left}, {right}]"
                        )

                    reference = high_precision_reference_integral(
                        source_coefficients,
                        left,
                        right,
                        config.reference_dps,
                    )

                    for basis in config.bases:
                        coefficients = coefficients_for_basis(
                            source_coefficients,
                            left,
                            right,
                            basis,
                        )
                        integrator = integrator_for_basis(basis)

                        for precision in config.precisions:
                            dtype = np.dtype(
                                np.float32
                                if precision == "float32"
                                else np.float64
                            )

                            calculation = lambda: integrator(
                                coefficients,
                                left,
                                right,
                                dtype,
                            )

                            try:
                                computed = calculation()
                                absolute_error, relative_error = error_metrics(
                                    computed,
                                    reference,
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
                            ) as exc:
                                computed = float("nan")
                                absolute_error = float("nan")
                                relative_error = float("nan")
                                runtime = float("nan")
                                status = (
                                    f"failed: {type(exc).__name__}: {exc}"
                                )

                            rows.append(
                                IntervalResult(
                                    experiment="experiment2_interval_integration",
                                    family=family,
                                    seed=seed,
                                    degree=degree,
                                    basis=basis,
                                    precision=precision,
                                    interval_left=left,
                                    interval_right=right,
                                    interval_width=right - left,
                                    interval_midpoint=(left + right) / 2.0,
                                    reference_integral=reference,
                                    computed_integral=computed,
                                    absolute_error=absolute_error,
                                    relative_error=relative_error,
                                    coefficient_l2_norm=float(
                                        np.linalg.norm(coefficients)
                                    ),
                                    median_runtime_seconds=runtime,
                                    status=status,
                                )
                            )

    csv_path = results_directory / "interval_results.csv"
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


def valid_rows(rows: list[IntervalResult]) -> list[IntervalResult]:
    return [
        row
        for row in rows
        if row.status == "ok"
        and np.isfinite(row.absolute_error)
        and np.isfinite(row.relative_error)
    ]


def create_summary(
    rows: list[IntervalResult],
    results_directory: Path,
) -> None:
    grouped: dict[
        tuple[str, int, float, float, str, str],
        list[IntervalResult],
    ] = {}

    for row in valid_rows(rows):
        key = (
            row.family,
            row.degree,
            row.interval_left,
            row.interval_right,
            row.basis,
            row.precision,
        )
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []

    for key, group in sorted(grouped.items()):
        (
            family,
            degree,
            left,
            right,
            basis,
            precision,
        ) = key

        summary.append(
            {
                "family": family,
                "degree": degree,
                "interval_left": left,
                "interval_right": right,
                "interval_width": right - left,
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
                "median_coefficient_l2_norm": float(
                    np.median(
                        [row.coefficient_l2_norm for row in group]
                    )
                ),
            }
        )

    write_json(
        results_directory / "interval_summary.json",
        summary,
    )


def create_plots(
    rows: list[IntervalResult],
    figures_directory: Path,
) -> None:
    finite = valid_rows(rows)
    intervals = sorted(
        {
            (row.interval_left, row.interval_right)
            for row in finite
        }
    )
    families = sorted({row.family for row in finite})

    for family in families:
        for left, right in intervals:
            for precision in ("float32", "float64"):
                plt.figure(figsize=(7.2, 4.8))

                for basis in ("monomial", "legendre", "chebyshev"):
                    subset = [
                        row
                        for row in finite
                        if row.family == family
                        and row.interval_left == left
                        and row.interval_right == right
                        and row.precision == precision
                        and row.basis == basis
                    ]
                    degrees = sorted({row.degree for row in subset})
                    values = [
                        float(
                            np.median(
                                [
                                    row.relative_error
                                    for row in subset
                                    if row.degree == degree
                                ]
                            )
                        )
                        for degree in degrees
                    ]
                    plt.plot(
                        degrees,
                        values,
                        marker="o",
                        label=basis,
                    )

                plt.yscale("log")
                plt.xlabel("Polynomial degree")
                plt.ylabel("Median relative integral error")
                plt.title(
                    f"Interval [{left:g}, {right:g}], "
                    f"{family}, {precision}"
                )
                plt.grid(True, which="both", alpha=0.3)
                plt.legend()
                plt.tight_layout()
                safe_interval = (
                    f"{left:g}_{right:g}"
                    .replace("-", "m")
                    .replace(".", "p")
                )
                plt.savefig(
                    figures_directory
                    / (
                        "relative_error_vs_degree_"
                        f"{family}_interval_{safe_interval}_"
                        f"{precision}.png"
                    ),
                    dpi=220,
                )
                plt.close()

    max_degree = max(row.degree for row in finite)

    for family in families:
        for precision in ("float32", "float64"):
            plt.figure(figsize=(7.2, 4.8))

            for basis in ("monomial", "legendre", "chebyshev"):
                subset = [
                    row
                    for row in finite
                    if row.family == family
                    and row.degree == max_degree
                    and row.precision == precision
                    and row.basis == basis
                ]

                widths = sorted({row.interval_width for row in subset})
                values = [
                    float(
                        np.median(
                            [
                                row.relative_error
                                for row in subset
                                if row.interval_width == width
                            ]
                        )
                    )
                    for width in widths
                ]

                plt.plot(widths, values, marker="o", label=basis)

            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Interval width")
            plt.ylabel("Median relative integral error")
            plt.title(
                f"Domain-scale sensitivity: degree={max_degree}, "
                f"{family}, {precision}"
            )
            plt.grid(True, which="both", alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                figures_directory
                / (
                    "relative_error_vs_interval_width_"
                    f"{family}_{precision}.png"
                ),
                dpi=220,
            )
            plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 2: analytical one-dimensional integration "
            "across polynomial bases, degrees, intervals and precisions."
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
        default=Path("results/experiment2"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures/experiment2"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = (
        IntervalConfig(
            degrees=(2, 6, 10),
            intervals=((-1.0, 1.0), (0.0, 10.0)),
            families=("random_uniform",),
            seeds=(0,),
            repetitions=2,
        )
        if args.quick
        else IntervalConfig()
    )

    rows = run_interval_experiment(
        config=config,
        results_directory=args.results_dir,
        figures_directory=args.figures_dir,
    )

    successful = sum(row.status == "ok" for row in rows)
    print(f"Completed {successful}/{len(rows)} runs.")
    print("Results:", args.results_dir / "interval_results.csv")
    print("Summary:", args.results_dir / "interval_summary.json")
    print("Figures:", args.figures_dir)


if __name__ == "__main__":
    main()
