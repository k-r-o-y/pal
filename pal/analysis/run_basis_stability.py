import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from basis_polynomials import (
    monomial_eval,
    legendre_eval,
    chebyshev_eval,
    convert_monomial_to_legendre,
    convert_monomial_to_chebyshev,
    integrate_monomial,
    integrate_legendre,
    integrate_chebyshev,
)


def relative_error(a, b, eps=1e-12):
    return abs(a - b) / max(abs(b), eps)


def run_experiment(max_degree, perturbation, seed, output_dir):
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    x = np.linspace(-1.0, 1.0, 1000)

    for degree in range(2, max_degree + 1):
        mono_coeffs = rng.normal(0.0, 1.0, degree + 1)

        leg_coeffs = convert_monomial_to_legendre(mono_coeffs)
        cheb_coeffs = convert_monomial_to_chebyshev(mono_coeffs)

        mono_original = monomial_eval(mono_coeffs, x)
        leg_original = legendre_eval(leg_coeffs, x)
        cheb_original = chebyshev_eval(cheb_coeffs, x)

        delta_mono = rng.normal(0.0, perturbation, mono_coeffs.shape)
        delta_leg = rng.normal(0.0, perturbation, leg_coeffs.shape)
        delta_cheb = rng.normal(0.0, perturbation, cheb_coeffs.shape)

        mono_perturbed = monomial_eval(mono_coeffs + delta_mono, x)
        leg_perturbed = legendre_eval(leg_coeffs + delta_leg, x)
        cheb_perturbed = chebyshev_eval(cheb_coeffs + delta_cheb, x)

        mono_sensitivity = np.linalg.norm(mono_perturbed - mono_original) / np.linalg.norm(delta_mono)
        leg_sensitivity = np.linalg.norm(leg_perturbed - leg_original) / np.linalg.norm(delta_leg)
        cheb_sensitivity = np.linalg.norm(cheb_perturbed - cheb_original) / np.linalg.norm(delta_cheb)

        ref_integral = integrate_monomial(mono_coeffs, -1.0, 1.0)
        leg_integral = integrate_legendre(leg_coeffs, -1.0, 1.0)
        cheb_integral = integrate_chebyshev(cheb_coeffs, -1.0, 1.0)

        rows.append({
            "degree": degree,
            "basis": "monomial",
            "sensitivity": mono_sensitivity,
            "integral": ref_integral,
            "relative_integral_error": 0.0,
        })

        rows.append({
            "degree": degree,
            "basis": "legendre",
            "sensitivity": leg_sensitivity,
            "integral": leg_integral,
            "relative_integral_error": relative_error(leg_integral, ref_integral),
        })

        rows.append({
            "degree": degree,
            "basis": "chebyshev",
            "sensitivity": cheb_sensitivity,
            "integral": cheb_integral,
            "relative_integral_error": relative_error(cheb_integral, ref_integral),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "basis_stability_results.csv", index=False)
    print(df)
    print(f"\nSaved results to {output_dir / 'basis_stability_results.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-degree", type=int, default=20)
    parser.add_argument("--perturbation", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    run_experiment(
        max_degree=args.max_degree,
        perturbation=args.perturbation,
        seed=args.seed,
        output_dir=args.output_dir,
    )