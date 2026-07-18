import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_results(csv_path, output_dir):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    plt.figure()
    for basis in df["basis"].unique():
        subset = df[df["basis"] == basis]
        plt.plot(subset["degree"], subset["sensitivity"], marker="o", label=basis)

    plt.xlabel("Polynomial degree")
    plt.ylabel("Perturbation sensitivity")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "basis_sensitivity.pdf")
    plt.savefig(output_dir / "basis_sensitivity.png", dpi=300)

    plt.figure()
    for basis in df["basis"].unique():
        subset = df[df["basis"] == basis]
        plt.plot(subset["degree"], subset["relative_integral_error"], marker="o", label=basis)

    plt.xlabel("Polynomial degree")
    plt.ylabel("Relative integral error")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "basis_integral_error.pdf")
    plt.savefig(output_dir / "basis_integral_error.png", dpi=300)

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/basis_stability_results.csv")
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

    plot_results(args.csv, args.output_dir)