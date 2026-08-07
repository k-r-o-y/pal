#!/usr/bin/env python3
"""
Self-contained representative PAL benchmark.

Produces CSV/JSON results for monomial, Legendre, and Chebyshev polynomial
representations over a box-with-obstacle constrained domain. Replace
`integrate_case` with the repository's native PAL integration call when ready.

Run:
    python -m analysis.pal.run_representative_pal_benchmark
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
from typing import Sequence

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from numpy.polynomial.legendre import legvander

BASES = ("monomial", "legendre", "chebyshev")
DTYPES = {"float32": np.float32, "float64": np.float64}
EPS = np.finfo(np.float64).eps


@dataclass
class Record:
    dimension: int
    source_degree: int
    density_degree: int
    trial: int
    basis: str
    dtype: str
    seed: int
    coefficient_count: int
    basis_condition_number: float
    numerical_rank_fraction: float
    conversion_relative_residual: float
    reference_partition: float
    computed_partition: float
    partition_relative_error: float
    reference_query_probability: float
    computed_query_probability: float
    query_absolute_error: float
    query_relative_error: float
    partition_perturbation_sensitivity: float
    query_perturbation_sensitivity: float
    minimum_sampled_density: float
    negative_density_detected: bool
    conversion_ms: float
    integration_ms: float
    total_ms: float


def multi_indices(dimension: int, degree: int) -> list[tuple[int, ...]]:
    return sorted(
        [a for a in product(range(degree + 1), repeat=dimension) if sum(a) <= degree],
        key=lambda a: (sum(a), a),
    )


def basis_matrix(points, indices, basis, dtype=np.float64):
    points = np.asarray(points, dtype=dtype)
    degree = max(max(a) for a in indices) if indices else 0
    tables = []
    for j in range(points.shape[1]):
        x = points[:, j]
        if basis == "monomial":
            table = np.column_stack([x ** k for k in range(degree + 1)])
        elif basis == "legendre":
            table = legvander(2.0 * x - 1.0, degree)
        elif basis == "chebyshev":
            table = chebvander(2.0 * x - 1.0, degree)
        else:
            raise ValueError(f"Unknown basis: {basis}")
        tables.append(np.asarray(table, dtype=dtype))

    out = np.ones((len(points), len(indices)), dtype=dtype)
    for col, alpha in enumerate(indices):
        for axis, exponent in enumerate(alpha):
            out[:, col] *= tables[axis][:, exponent]
    return out


def values(points, coefficients, indices, basis, dtype=np.float64):
    return basis_matrix(points, indices, basis, dtype) @ np.asarray(coefficients, dtype=dtype)


def fit_coefficients(points, target, indices, basis):
    matrix = basis_matrix(points, indices, basis, np.float64)
    coeff, _, rank, singular = np.linalg.lstsq(matrix, target, rcond=None)
    residual = np.linalg.norm(matrix @ coeff - target) / max(np.linalg.norm(target), EPS)
    condition = math.inf if singular[-1] == 0 else float(singular[0] / singular[-1])
    return coeff, float(residual), condition, float(rank / matrix.shape[1])


def deterministic_points(dimension: int, count: int, seed: int):
    k = np.arange(1, count + 1, dtype=np.float64)[:, None]
    factors = np.sqrt(np.arange(2, dimension + 2, dtype=np.float64))[None, :]
    pts = np.mod(k * factors, 1.0)
    rng = np.random.default_rng(seed)
    return np.clip(pts[rng.permutation(count)], 1e-9, 1 - 1e-9)


def construct_density(dimension, source_degree, seed, scale, offset):
    rng = np.random.default_rng(seed)
    q_idx = multi_indices(dimension, source_degree)
    q_coeff = rng.normal(0.0, scale, len(q_idx))
    q_coeff[0] *= 0.5

    p_degree = 2 * source_degree
    p_idx = multi_indices(dimension, p_degree)
    fit_pts = deterministic_points(dimension, max(3 * len(p_idx), len(p_idx) + 32), seed + 7)
    q = values(fit_pts, q_coeff, q_idx, "monomial")
    target = q * q + offset
    p_coeff, residual, _, _ = fit_coefficients(fit_pts, target, p_idx, "monomial")
    if residual > 1e-8:
        print(f"warning: density reconstruction residual={residual:.3e}")
    return p_idx, p_coeff


def quadrature(dimension, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    grid = np.asarray(list(product(range(order), repeat=dimension)), dtype=int)
    return nodes[grid], np.prod(weights[grid], axis=1)


def masks(points, obstacle_half_width, query_upper):
    obstacle = np.all(np.abs(points - 0.5) <= obstacle_half_width, axis=1)
    feasible = ~obstacle
    query = feasible & np.all(points <= query_upper, axis=1)
    return feasible, query


def integrate_case(coeff, indices, basis, dtype, points, weights, mask):
    """Integration hook. Replace this body with the native PAL integration API."""
    v = values(points.astype(dtype), np.asarray(coeff, dtype=dtype), indices, basis, dtype)
    return float(np.sum(v[mask] * weights.astype(dtype)[mask], dtype=dtype))


def rel_error(computed, reference):
    return abs(computed - reference) / max(abs(reference), EPS)


def perturb(coeff, magnitude, seed, dtype):
    rng = np.random.default_rng(seed)
    coeff = np.asarray(coeff, dtype=dtype)
    direction = rng.normal(size=coeff.shape).astype(dtype)
    direction /= max(float(np.linalg.norm(direction.astype(np.float64))), EPS)
    scale = max(float(np.linalg.norm(coeff.astype(np.float64))), EPS)
    return coeff + dtype(magnitude * scale) * direction


def sensitivity(perturbed, baseline, magnitude):
    return abs(perturbed - baseline) / (magnitude * max(abs(baseline), EPS))


def evaluate(args, dimension, degree, trial, basis, dtype_name):
    start = time.perf_counter()
    seed = args.base_seed + dimension * 1_000_003 + degree * 10_007 + trial * 101
    indices, mono_coeff = construct_density(
        dimension, degree, seed, args.coefficient_scale, args.density_offset
    )

    fit_pts = deterministic_points(dimension, max(3 * len(indices), len(indices) + 32), seed + 17)
    target = values(fit_pts, mono_coeff, indices, "monomial")

    t0 = time.perf_counter()
    coeff, residual, condition, rank_fraction = fit_coefficients(fit_pts, target, indices, basis)
    conversion_ms = (time.perf_counter() - t0) * 1000.0

    points, weights = quadrature(dimension, args.quadrature_order)
    feasible, query = masks(points, args.obstacle_half_width, args.query_upper)

    ref_z = integrate_case(mono_coeff, indices, "monomial", np.float64, points, weights, feasible)
    ref_a = integrate_case(mono_coeff, indices, "monomial", np.float64, points, weights, query)
    ref_q = ref_a / max(ref_z, EPS)

    dtype = DTYPES[dtype_name]
    t1 = time.perf_counter()
    z = integrate_case(coeff, indices, basis, dtype, points, weights, feasible)
    a = integrate_case(coeff, indices, basis, dtype, points, weights, query)
    q = a / max(z, EPS)

    pert = perturb(coeff, args.perturbation_magnitude, seed + 29, dtype)
    z_pert = integrate_case(pert, indices, basis, dtype, points, weights, feasible)
    a_pert = integrate_case(pert, indices, basis, dtype, points, weights, query)
    q_pert = a_pert / max(z_pert, EPS)
    integration_ms = (time.perf_counter() - t1) * 1000.0

    check_pts = deterministic_points(dimension, max(256, 4 * len(indices)), seed + 43)
    minimum = float(np.min(values(check_pts, coeff, indices, basis, dtype)))

    return Record(
        dimension=dimension,
        source_degree=degree,
        density_degree=2 * degree,
        trial=trial,
        basis=basis,
        dtype=dtype_name,
        seed=seed,
        coefficient_count=len(indices),
        basis_condition_number=condition,
        numerical_rank_fraction=rank_fraction,
        conversion_relative_residual=residual,
        reference_partition=ref_z,
        computed_partition=z,
        partition_relative_error=rel_error(z, ref_z),
        reference_query_probability=ref_q,
        computed_query_probability=q,
        query_absolute_error=abs(q - ref_q),
        query_relative_error=rel_error(q, ref_q),
        partition_perturbation_sensitivity=sensitivity(z_pert, z, args.perturbation_magnitude),
        query_perturbation_sensitivity=sensitivity(q_pert, q, args.perturbation_magnitude),
        minimum_sampled_density=minimum,
        negative_density_detected=minimum < -1e-6,
        conversion_ms=conversion_ms,
        integration_ms=integration_ms,
        total_ms=(time.perf_counter() - start) * 1000.0,
    )


def metric_summary(records, field):
    x = np.asarray([getattr(r, field) for r in records], dtype=np.float64)
    return {
        "count": int(len(x)),
        "minimum": float(np.min(x)),
        "q1": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "q3": float(np.quantile(x, 0.75)),
        "maximum": float(np.max(x)),
    }


def write_outputs(args, records, elapsed):
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    fields = list(asdict(records[0]))
    with args.results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)

    metrics = [
        "partition_relative_error",
        "query_absolute_error",
        "query_relative_error",
        "partition_perturbation_sensitivity",
        "query_perturbation_sensitivity",
        "basis_condition_number",
        "numerical_rank_fraction",
        "conversion_relative_residual",
        "total_ms",
    ]
    grouped = {}
    for basis in BASES:
        grouped[basis] = {}
        for dtype in args.dtypes:
            subset = [r for r in records if r.basis == basis and r.dtype == dtype]
            grouped[basis][dtype] = {m: metric_summary(subset, m) for m in metrics}

    summary = {
        "configuration": {
            "dimensions": args.dimensions,
            "degrees": args.degrees,
            "trials": args.trials,
            "dtypes": args.dtypes,
            "quadrature_order": args.quadrature_order,
            "query_upper": args.query_upper,
            "obstacle_half_width": args.obstacle_half_width,
            "density_offset": args.density_offset,
            "coefficient_scale": args.coefficient_scale,
            "perturbation_magnitude": args.perturbation_magnitude,
            "base_seed": args.base_seed,
        },
        "records": len(records),
        "finite_records": sum(
            all(not isinstance(v, float) or math.isfinite(v) for v in asdict(r).values())
            for r in records
        ),
        "negative_density_records": sum(r.negative_density_detected for r in records),
        "elapsed_seconds": elapsed,
        "grouped_metrics": grouped,
    }
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--dimensions", nargs="+", type=int, default=[2, 3])
    p.add_argument("--degrees", nargs="+", type=int, default=[0, 1, 2, 3, 5])
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--dtypes", nargs="+", choices=DTYPES, default=["float32", "float64"])
    p.add_argument("--quadrature-order", type=int, default=14)
    p.add_argument("--query-upper", type=float, default=0.75)
    p.add_argument("--obstacle-half-width", type=float, default=0.15)
    p.add_argument("--density-offset", type=float, default=0.1)
    p.add_argument("--coefficient-scale", type=float, default=0.35)
    p.add_argument("--perturbation-magnitude", type=float, default=1e-5)
    p.add_argument("--base-seed", type=int, default=20260806)
    p.add_argument(
        "--results-path",
        type=Path,
        default=Path("results/pal/representative_pal_results.csv"),
    )
    p.add_argument(
        "--summary-path",
        type=Path,
        default=Path("results/pal/representative_pal_summary.json"),
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    if any(d < 1 for d in args.dimensions):
        raise ValueError("dimensions must be positive")
    if any(d < 0 for d in args.degrees):
        raise ValueError("degrees must be non-negative")
    if args.trials < 1 or args.quadrature_order < 2:
        raise ValueError("trials >= 1 and quadrature-order >= 2 are required")

    outer = len(args.dimensions) * len(args.degrees) * args.trials
    print("Representative PAL benchmark")
    print("============================")
    print(f"dimensions: {args.dimensions}")
    print(f"source degrees: {args.degrees}")
    print(f"bases: {list(BASES)}")
    print(f"dtypes: {args.dtypes}")
    print(f"expected records: {outer * len(BASES) * len(args.dtypes)}")
    print()

    started = time.perf_counter()
    records = []
    current = 0
    for dimension in args.dimensions:
        for degree in args.degrees:
            for trial in range(args.trials):
                current += 1
                print(
                    f"[{current:4d}/{outer:<4d}] dimension={dimension}, "
                    f"source_degree={degree}, density_degree={2 * degree}, trial={trial}",
                    flush=True,
                )
                for basis in BASES:
                    for dtype_name in args.dtypes:
                        records.append(
                            evaluate(args, dimension, degree, trial, basis, dtype_name)
                        )

    elapsed = time.perf_counter() - started
    write_outputs(args, records, elapsed)
    print(f"\nCompleted {len(records)} records in {elapsed:.3f} seconds.")
    print(f"Wrote {args.results_path}")
    print(f"Wrote {args.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
