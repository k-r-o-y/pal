#!/usr/bin/env python3
"""
Geometric-scale benchmark for Chapter 5.

This script reuses the validated helpers in
analysis/simplex/run_simplex_benchmark.py and varies only geometric scale.

Within each trial, exactly the same polynomial coefficient tensor is reused at
every scale. This prevents random coefficient differences from confounding the
effect of domain scaling.

Default outputs:
    results/simplex/geometric_scale_results.csv
    results/simplex/geometric_scale_summary.json

Run:
    python -m analysis.simplex.run_geometric_scale_benchmark

Smoke test:
    python -m analysis.simplex.run_geometric_scale_benchmark \
        --scales 0.1 1 10 --degree 5 --trials 2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from analysis.simplex.run_simplex_benchmark import (
    BenchmarkConfiguration,
    SUPPORTED_BASES,
    SimplexBenchmarkRecord,
    create_summary,
    environment_information,
    exact_simplex_integral,
    generate_monomial_tensor,
    parse_basis,
    parse_dtype,
    print_non_finite_records,
    print_table,
    run_basis_case,
    simplex_quadrature_rule,
    total_degree_multiindices,
    validate_positive_finite,
    write_csv,
)

DEFAULT_SCALES = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def run_scale_benchmark(
    *,
    degree: int,
    dimension: int,
    scales: Sequence[float],
    dtype: np.dtype,
    bases: Sequence[str],
    trials: int,
    seed: int,
    coefficient_scale: float,
    perturbation_magnitude: float,
    quadrature_order: int,
    condition_sample_multiplier: int,
    maximum_condition_samples: int,
) -> list[SimplexBenchmarkRecord]:
    """Run the scale sweep while holding all other variables fixed."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive")
    if condition_sample_multiplier <= 0:
        raise ValueError("condition_sample_multiplier must be positive")
    if maximum_condition_samples <= 0:
        raise ValueError("maximum_condition_samples must be positive")

    validate_positive_finite(coefficient_scale, "coefficient_scale")
    validate_positive_finite(perturbation_magnitude, "perturbation_magnitude")

    scale_values = [float(value) for value in scales]
    for scale in scale_values:
        validate_positive_finite(scale, "scale")

    dtype = np.dtype(dtype)
    indices = total_degree_multiindices(dimension, degree)
    records: list[SimplexBenchmarkRecord] = []

    for trial in range(trials):
        # Generate one polynomial and reuse it for every scale in this trial.
        trial_seed = seed + trial
        coefficient_rng = np.random.default_rng(trial_seed)
        monomial_tensor, generated_indices = generate_monomial_tensor(
            rng=coefficient_rng,
            dimension=dimension,
            degree=degree,
            coefficient_scale=coefficient_scale,
            dtype=dtype,
        )

        if generated_indices != indices:
            raise RuntimeError("Internal multi-index ordering mismatch")

        for scale_index, scale in enumerate(scale_values):
            points, weights = simplex_quadrature_rule(
                dimension=dimension,
                scale=scale,
                order=quadrature_order,
                dtype=dtype,
            )
            reference = exact_simplex_integral(
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
                dtype=dtype.name,
                coefficient_scale=coefficient_scale,
                perturbation_magnitude=perturbation_magnitude,
                quadrature_order=quadrature_order,
            )

            for basis_index, basis in enumerate(bases):
                condition_rng = np.random.default_rng(
                    seed
                    + 100_000_000
                    + 1_000_000 * trial
                    + 10_000 * scale_index
                    + 100 * basis_index
                )
                perturbation_rng = np.random.default_rng(
                    seed
                    + 200_000_000
                    + 1_000_000 * trial
                    + 10_000 * scale_index
                    + 100 * basis_index
                )

                records.append(
                    run_basis_case(
                        configuration=configuration,
                        basis=basis,
                        monomial_tensor=monomial_tensor,
                        indices=indices,
                        reference_integral=reference,
                        quadrature_points=points,
                        quadrature_weights=weights,
                        condition_rng=condition_rng,
                        perturbation_rng=perturbation_rng,
                        condition_sample_multiplier=condition_sample_multiplier,
                        maximum_condition_samples=maximum_condition_samples,
                    )
                )

    return records


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark numerical behaviour versus simplex scale."
    )
    p.add_argument("--degree", type=int, default=10)
    p.add_argument("--dimension", type=int, default=2)
    p.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=list(DEFAULT_SCALES),
    )
    p.add_argument(
        "--dtype",
        type=parse_dtype,
        default=np.dtype(np.float64),
    )
    p.add_argument(
        "--bases",
        nargs="+",
        type=parse_basis,
        default=list(SUPPORTED_BASES),
    )
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--coefficient-scale", type=float, default=1.0)
    p.add_argument("--perturbation-magnitude", type=float, default=1.0e-7)
    p.add_argument("--quadrature-order", type=int, default=16)
    p.add_argument("--condition-sample-multiplier", type=int, default=2)
    p.add_argument("--maximum-condition-samples", type=int, default=2000)
    p.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/simplex"),
    )
    p.add_argument("--allow-non-finite", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    dtype = np.dtype(args.dtype)

    print("Geometric-scale benchmark")
    print(f"degree={args.degree}, dimension={args.dimension}")
    print(f"scales={args.scales}")
    print(f"dtype={dtype.name}, trials={args.trials}")
    print(f"bases={args.bases}")

    records = run_scale_benchmark(
        degree=args.degree,
        dimension=args.dimension,
        scales=args.scales,
        dtype=dtype,
        bases=args.bases,
        trials=args.trials,
        seed=args.seed,
        coefficient_scale=args.coefficient_scale,
        perturbation_magnitude=args.perturbation_magnitude,
        quadrature_order=args.quadrature_order,
        condition_sample_multiplier=args.condition_sample_multiplier,
        maximum_condition_samples=args.maximum_condition_samples,
    )

    output_dir = args.output_directory
    csv_path = output_dir / "geometric_scale_results.csv"
    json_path = output_dir / "geometric_scale_summary.json"

    write_csv(records, csv_path)
    summary = create_summary(records)
    payload = {
        "experiment": "geometric_scale_benchmark",
        "summary": asdict(summary),
        "configuration": {
            "degree": args.degree,
            "dimension": args.dimension,
            "scales": [float(x) for x in args.scales],
            "dtype": dtype.name,
            "bases": list(args.bases),
            "trials": args.trials,
            "seed": args.seed,
            "coefficient_scale": args.coefficient_scale,
            "perturbation_magnitude": args.perturbation_magnitude,
            "quadrature_order": args.quadrature_order,
            "condition_sample_multiplier": args.condition_sample_multiplier,
            "maximum_condition_samples": args.maximum_condition_samples,
        },
        "environment": environment_information(),
        "design_note": (
            "The same monomial coefficient tensor is reused across all scales "
            "within each trial."
        ),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print_table(records)
    print_non_finite_records(records)

    failures = sum(not record.finite for record in records)
    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Records: {len(records)}")
    print(f"Non-finite records: {failures}")

    return 0 if args.allow_non_finite or failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
