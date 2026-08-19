# PAL - A Probabilistic Neuro-symbolic Layer for Algebraic Constraint Satisfaction

[![Python application](https://github.com/april-tools/pal/actions/workflows/python-app.yml/badge.svg)](https://github.com/april-tools/pal/actions/workflows/python-app.yml)

This repository contains the code for **PAL (Probabilistic Algebraic Layer)**, a probabilistic neuro-symbolic framework for **algebraic constraint satisfaction under probabilistic inference**.

The original implementation focuses on **spline-based constrained probabilistic inference** and accompanies the paper:

> **Leander Kurscheidt, Paolo Morettin, Roberto Sebastiani, Andrea Passerini, Antonio Vergari.**
>
> *A Probabilistic Neuro-symbolic Layer for Algebraic Constraint Satisfaction.*
>
> Proceedings of the 41st Conference on Uncertainty in Artificial Intelligence (UAI 2025).

This fork extends the original PAL implementation with a comprehensive experimental framework for investigating the **numerical stability of polynomial basis representations in constrained probabilistic inference**.

The repository now contains the complete experimental framework developed for the MSc dissertation:

> **Investigating the Numerical Stability of Polynomial Basis Representations in Constrained Probabilistic Inference**
>
> University of Edinburgh
>
> School of Informatics
>
> MSc Informatics (High Performance Computing with Data Science)

---

# Overview

Polynomial representations are fundamental to constrained probabilistic inference.

However, high-degree polynomial approximations are often numerically unstable, particularly when represented using monomial bases.

This repository investigates how different polynomial bases influence:

- Numerical conditioning
- Coefficient recovery
- Constraint integration
- Partition function estimation
- Query probability estimation
- Numerical precision
- High-dimensional inference
- Constrained polytope integration
- End-to-end PAL inference

Three polynomial bases are compared throughout the experiments:

- **Monomial basis**
- **Legendre basis**
- **Chebyshev basis**

---

# Repository Structure

```text
.
├── analysis/              # Benchmark implementations
├── data/                  # PAL datasets
├── docs/                  # Experimental documentation
├── figures/               # Dissertation figures
├── pal/                   # PAL implementation
├── tests_pal/             # Testing utilities
├── README.md
└── setup.sh
```

---

# Original PAL Implementation

The original PAL framework provides:

- Probabilistic algebraic constraint satisfaction
- Spline-based inference
- Weighted model integration
- Constrained trajectory prediction
- Neural network integration
- Constrained Stanford Drone Dataset experiments

The original PAL paper is available here:

https://proceedings.mlr.press/v286/kurscheidt25a.html

---

# Example PAL Prediction

PAL predicts a probability distribution over future trajectories while guaranteeing constraint satisfaction.

Constrained prediction:

![Example prediction](data/sdd_spline_example.png)

Unconditional prediction:

![Example prediction](data/sdd_unconditional_spline_example.png)

---

# Installation

Clone the repository:

```bash
git clone https://github.com/k-r-o-y/pal.git
cd pal
```

Install all dependencies:

```bash
./setup.sh
```

---

# Original PAL Experiments

## Constrained Stanford Drone Dataset

Train a simple MLP on the constrained SDD dataset:

```bash
python pal/training/train_mlp_sdd.py \
    --epochs 10 \
    --init_last_layer_positive \
    --seed 1744909132
```

Expected mean test log-likelihood:

```text
-1.9149
```

---

## Unconditional SDD Experiment

Train the unconditional variant:

```bash
python pal/training/train_unconditional_sdd.py \
    --init_positive \
    --use_float64 \
    --num_knots 14 \
    --num_mixtures 10 \
    --lr 0.01 \
    --epochs 1500 \
    --seed 1764087361
```

Expected mean test log-likelihood:

```text
-2.9493
```

---

# Numerical Stability Extension

The numerical stability framework investigates how polynomial basis representations affect constrained probabilistic inference.

The experiments systematically vary:

- Polynomial degree
- Domain scaling
- Constraint complexity
- Dimensionality
- Numerical precision
- Perturbation magnitude

The framework evaluates both **representation-level stability** and **downstream probabilistic accuracy**.

---

# Implemented Experiments

## 1. Basis Equivalence and Representation Analysis

Verifies that equivalent polynomial functions remain equivalent after conversion between basis representations.

Evaluates:

- Basis conversion accuracy
- Functional equivalence
- Integration consistency
- Representation stability

Generated figures:

```text
generated_polynomial_1d_examples
generated_polynomial_2d_heatmaps
generated_polynomial_2d_surfaces
generated_polynomial_basis_equivalence
generated_polynomial_constrained_region
basis_integral_error
basis_sensitivity
```

---

## 2. Conditioning Analysis

Investigates how polynomial conditioning changes as polynomial degree increases.

Evaluates:

- Vandermonde conditioning
- Singular value spectra
- Constraint-induced Gram matrices
- Effective numerical rank

Generated figures:

```text
condition_number_vs_degree
condition_number_vs_scale
constraint_gram_condition_number_vs_degree
constraint_gram_rank_fraction_vs_degree
```

---

## 3. Higher-Dimensional Benchmarks

Investigates the effect of increasing dimensionality.

Evaluates:

- Dimensions 1-5
- Basis growth
- Runtime scaling
- Numerical stability

Generated figures:

```text
condition_number_vs_dimension
relative_error_vs_dimension
runtime_and_basis_count_vs_dimension
```

---

## 4. Precision Experiments

Compares reduced and full precision arithmetic.

Evaluates:

- Float32
- Float64

Measures:

- Conditioning
- Relative error
- Rank preservation
- Runtime

Generated figures:

```text
condition_number_vs_precision
rank_fraction_vs_precision
relative_error_vs_precision
runtime_vs_precision
```

---

## 5. Perturbation and Coefficient Recovery

Introduces controlled perturbations into coefficient reconstruction.

Measures:

- Noise amplification
- Recovery accuracy
- Sensitivity to perturbations

Generated figures:

```text
perturbation_sensitivity_vs_degree
perturbation_sensitivity_vs_dimension
perturbation_sensitivity_vs_precision
```

---

## 6. Downstream Probability Experiments

Evaluates whether representation-level numerical errors propagate into probabilistic quantities.

Measures:

- Partition function accuracy
- Query probability accuracy
- Sensitivity
- Runtime

Generated figures:

```text
partition_function_error_vs_degree
query_probability_error_vs_degree
query_probability_error_vs_query_scale
query_probability_sensitivity_vs_degree
downstream_runtime_vs_degree
downstream_error_summary
```

---

## 7. Constrained Polytope Benchmarks

Extends the original simplex experiments to SMT(LRA)-style constrained polytopes.

Evaluates:

- Static constraints
- Dynamic constraints
- Scheduling constraints
- Trajectory constraints
- Obstacle-avoidance geometries
- Noisy observations

Measures:

- Condition number
- Numerical rank
- Integration error
- Recovery error
- Runtime

Generated figures:

```text
polytope_condition_number_vs_degree
polytope_rank_fraction_vs_degree
polytope_integration_error_vs_degree
polytope_perturbation_sensitivity_vs_degree
polytope_recovered_integral_error_vs_degree
polytope_coefficient_noise_amplification_vs_degree
polytope_runtime_vs_degree
polytope_runtime_vs_dimension
polytope_dynamic_schedule_error
polytope_dynamic_trajectory_error
polytope_error_summary
```

---

## 8. PAL-Specific Benchmarks

Evaluates numerical stability within the PAL inference pipeline itself.

Measures:

- Partition function accuracy
- Query probability accuracy
- Basis conversion stability
- Conditioning
- Rank preservation

Generated figures:

```text
pal_condition_number_vs_degree
pal_rank_fraction_vs_degree
pal_partition_function_error_vs_degree
pal_partition_function_error_vs_dimension
pal_query_probability_error_vs_degree
pal_query_probability_error_vs_dimension
pal_perturbation_sensitivity_vs_degree
pal_error_summary
```

---

# Figures

All figures used in the MSc dissertation are available in the `figures/` directory.

Figures are provided in three formats:

- PDF (publication quality)
- PNG (quick visualization)
- SVG (vector graphics)

The repository currently contains more than 130 generated figures covering:

- Polynomial basis visualization
- Conditioning analysis
- Constraint-induced conditioning
- Higher-dimensional inference
- Numerical precision
- Perturbation analysis
- Coefficient recovery
- Downstream probability estimation
- Polytope integration
- PAL inference

---

# Documentation

Additional documentation is available in the `docs/` directory.

| Document | Description |
| --- | --- |
| `experiments.md` | Complete experimental methodology |
| `experimental_results.md` | Experimental findings and analysis |

---

# Reproducibility

The repository includes:

- Deterministic random seeds
- Benchmark implementations
- Stored experimental outputs
- Plot-generation scripts
- Figure generation pipelines
- Experiment metadata

All figures reported in the dissertation can be reproduced directly from the repository.

---

# GASP

The GASP dependency was added via subtree from:

```text
https://github.com/april-tools/gasp.git
```

Update the subtree:

```bash
git subtree pull \
    --prefix pal/wmi/gasp \
    https://github.com/april-tools/gasp.git \
    main \
    --squash
```

Push the subtree:

```bash
git subtree push \
    --prefix pal/wmi/gasp \
    https://github.com/april-tools/gasp.git \
    main
```

---

# Citation

Original PAL paper:

```bibtex
@inproceedings{kurscheidt2025probabilistic,
    title={A Probabilistic Neuro-symbolic Layer for Algebraic Constraint Satisfaction},
    author={Kurscheidt, Leander and Morettin, Paolo and Sebastiani, Roberto and Passerini, Andrea and Vergari, Antonio},
    booktitle={Conference on Uncertainty in Artificial Intelligence},
    pages={2431--2471},
    year={2025},
    organization={PMLR}
}
```

If you use the numerical stability extensions developed in this repository, please also cite the accompanying MSc dissertation.
