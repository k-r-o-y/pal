# PAL - A Probabilistic Neuro-symbolic Layer for Algebraic Constraint Satisfaction

[![Python application](https://github.com/april-tools/pal/actions/workflows/python-app.yml/badge.svg)](https://github.com/april-tools/pal/actions/workflows/python-app.yml)

This repository contains the code for **PAL (Probabilistic Algebraic Layer)**, a probabilistic neuro-symbolic layer for algebraic constraint satisfaction.

The original PAL implementation focuses on spline-based constrained probabilistic inference and accompanies the paper:

> Leander Kurscheidt, Paolo Morettin, Roberto Sebastiani, Andrea Passerini, Antonio Vergari.
> *A Probabilistic Neuro-symbolic Layer for Algebraic Constraint Satisfaction.*
> UAI 2025.

This fork extends the original implementation with an experimental framework for investigating the **numerical stability of polynomial basis representations in constrained probabilistic inference**.

The repository now includes the complete experimental code used in the MSc dissertation:

> *Investigating the Numerical Stability of Polynomial Basis Representations in Constrained Probabilistic Inference.*

---

# Repository Structure

The repository now contains two complementary components:

- **Original PAL implementation**
    - Spline-based constrained probabilistic inference
    - Constrained Stanford Drone Dataset experiments
    - PAL training and evaluation pipelines

- **Numerical stability extension**
    - Monomial, Legendre, and Chebyshev basis implementations
    - Conditioning diagnostics
    - Coefficient recovery experiments
    - Precision studies
    - Perturbation analysis
    - Constrained integration benchmarks
    - Downstream probability evaluation
    - Representative PAL benchmarks
    - Reproducibility utilities

---

# Example PAL Prediction

This is an example prediction of PAL on the Constrained Stanford Drone Dataset.

We predict a probability distribution over future trajectories while guaranteeing constraint satisfaction.

![Example image](data/sdd_spline_example.png)

![Example image](data/sdd_unconditional_spline_example.png)

---

# Installation

Clone the repository and run:

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

# Numerical Stability Experiments

The numerical stability framework extends PAL with experiments that investigate how polynomial basis representations affect constrained probabilistic inference.

The experiments compare:

- Monomial bases
- Legendre bases
- Chebyshev bases

under increasing:

- Polynomial degree
- Dimensionality
- Constraint complexity
- Numerical precision
- Perturbation magnitude

---

# Implemented Experiments

## 1. Basis Conversion and Equivalence

Verifies that equivalent polynomial functions remain equivalent after conversion between basis representations.

Evaluates:

- Basis conversion error
- Numerical consistency
- Functional equivalence

---

## 2. Conditioning Experiments

Measures the conditioning of polynomial representations as degree increases.

Evaluates:

- Vandermonde condition numbers
- Singular-value spectra
- Constraint-induced Gram matrices
- Numerical rank

Produces:

- `condition_number_vs_degree`
- `constraint_gram_condition_number_vs_degree`
- `constraint_gram_rank_fraction_vs_degree`

---

## 3. Higher-Dimensional Experiments

Investigates how conditioning changes as dimensionality increases.

Evaluates:

- Dimensions 1–5
- Basis growth
- Runtime scaling
- Rank preservation

Produces:

- `condition_number_vs_dimension`
- `relative_error_vs_dimension`
- `runtime_and_basis_count_vs_dimension`

---

## 4. Precision Experiments

Compares reduced and full precision arithmetic.

Evaluates:

- `float32`
- `float64`

Measures:

- Conditioning
- Relative error
- Numerical rank
- Runtime

Produces:

- `rank_fraction_vs_precision`
- `relative_error_vs_precision`
- `runtime_vs_precision`

---

## 5. Perturbation and Coefficient Recovery

Introduces controlled perturbations into coefficient recovery.

Measures:

- Noise amplification
- Coefficient reconstruction error
- Sensitivity to perturbations

Produces:

- `perturbation_sensitivity_vs_degree`
- `perturbation_sensitivity_vs_dimension`
- `perturbation_sensitivity_vs_precision`

---

## 6. Downstream Probability Experiments

Evaluates whether representation-level numerical differences propagate into probabilistic quantities.

Measures:

- Partition-function error
- Normalized query probability error
- Runtime
- Sensitivity

Produces:

- `partition_function_error_vs_degree`
- `query_probability_error_vs_degree`
- `query_probability_error_vs_query_scale`
- `query_probability_sensitivity_vs_degree`
- `downstream_runtime_vs_degree`
- `downstream_error_summary`

---

## 7. Constrained Polytope Benchmarks

Replaces simplex integration with SMT(LRA)-style constrained polytopes.

Evaluates:

- Static constraints
- Dynamic constraints
- Box-with-obstacle geometries
- Recovery under noisy observations

Measures:

- Condition number
- Rank fraction
- Integration error
- Recovery error
- Runtime

Produces:

- `polytope_condition_number_vs_degree`
- `polytope_rank_fraction_vs_degree`
- `polytope_integration_error_vs_degree`
- `polytope_perturbation_sensitivity_vs_degree`
- `polytope_recovered_integral_error_vs_degree`
- `polytope_coefficient_noise_amplification_vs_degree`
- `polytope_runtime_vs_degree`
- `polytope_runtime_vs_dimension`
- `polytope_dynamic_schedule_error`
- `polytope_dynamic_trajectory_error`
- `polytope_error_summary`

---

## 8. Representative PAL Benchmarks

Evaluates numerical stability within the PAL inference pipeline itself.

Measures:

- Partition-function accuracy
- Query probability accuracy
- Coefficient construction stability
- Basis conversion stability

Produces:

- `pal_condition_number_vs_degree`
- `pal_partition_function_error_vs_degree`
- `pal_query_probability_error_vs_degree`

---

# Reproducibility

The repository includes:

- Deterministic seeds
- Stored row-level benchmark results
- Plot-generation scripts
- Configuration files
- Environment recording
- Experiment metadata

All figures in the dissertation can be reproduced directly from the repository.

---

# GASP!

The dependency was added via subtree from:

https://github.com/april-tools/gasp.git

Update via:

```bash
git subtree pull \
    --prefix pal/wmi/gasp \
    https://github.com/april-tools/gasp.git \
    main \
    --squash
```

Push via:

```bash
git subtree push \
    --prefix pal/wmi/gasp \
    https://github.com/april-tools/gasp.git \
    main
```

---

# Citation

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
