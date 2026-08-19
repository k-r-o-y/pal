# Experiments: Numerical Stability of Polynomial Basis Representations in Constrained Probabilistic Inference

---

# Overview

This document describes the complete experimental framework developed to investigate the numerical stability of polynomial basis representations in constrained probabilistic inference.

The experiments were designed to answer four primary research questions:

1. How does the choice of polynomial basis affect numerical conditioning?

2. How does increasing polynomial degree influence numerical stability?

3. How do higher-dimensional constrained probabilistic problems affect polynomial representations?

4. Do representation-level numerical instabilities propagate into downstream probabilistic inference?

The experimental framework extends the original PAL implementation by introducing a comprehensive benchmarking suite that evaluates polynomial representations across multiple problem settings, including:

- Simplex-constrained domains
- Arbitrary convex polytopes
- Higher-dimensional probability spaces
- Floating-point precision studies
- Perturbation and noise experiments
- Downstream probabilistic inference
- Representative PAL workloads

---

# Experimental Design

## Polynomial Bases

Three polynomial representations are evaluated throughout all experiments:

- Monomial basis
- Legendre basis
- Chebyshev basis

The monomial basis serves as the baseline because it is the standard polynomial representation used in many numerical implementations.

Orthogonal polynomial bases (Legendre and Chebyshev) are introduced to investigate whether improved conditioning translates into improved probabilistic inference.

---

## Controlled Variables

The experimental framework varies five independent variables:

| Variable | Range |
| --- | --- |
| Polynomial degree | 1-20 |
| Dimensionality | 1-5 |
| Numerical precision | float32, float64 |
| Constraint complexity | Simplex and polytope domains |
| Perturbation magnitude | Controlled coefficient noise |

---

## Evaluation Metrics

Each experiment evaluates one or more of the following metrics.

### Conditioning

Measures:

- Vandermonde condition number
- Constraint Gram matrix condition number

---

### Numerical Rank

Measures:

- Effective matrix rank
- Rank preservation under increasing degree

---

### Reconstruction Accuracy

Measures:

- Relative coefficient error
- Polynomial reconstruction error

---

### Downstream Probabilistic Accuracy

Measures:

- Partition function error
- Query probability error

---

### Stability

Measures:

- Noise amplification
- Perturbation sensitivity

---

### Computational Performance

Measures:

- Runtime
- Basis growth
- Dimensional scaling

---

# Experiment 1: Basis Equivalence

## Objective

Verify that different polynomial bases represent identical underlying functions.

---

## Methodology

Equivalent polynomials are constructed using:

- Monomial coefficients
- Legendre coefficients
- Chebyshev coefficients

The resulting functions are evaluated over identical domains.

---

## Evaluation

Metrics:

- Function reconstruction error
- Basis conversion error

---

## Figures

- generated_polynomial_1d_examples.pdf
- generated_polynomial_2d_surfaces.pdf
- generated_polynomial_2d_heatmaps.pdf
- generated_polynomial_basis_equivalence.pdf

---

# Experiment 2: Polynomial Conditioning

## Objective

Measure how polynomial conditioning changes as polynomial degree increases.

---

## Methodology

Vandermonde matrices are constructed for polynomial degrees ranging from 1 to 20.

Condition numbers are computed for each polynomial basis.

Constraint matrices are then incorporated to determine whether constrained systems amplify numerical instability.

---

## Evaluation

Metrics:

- Vandermonde condition number
- Constraint Gram matrix condition number

---

## Findings

- Monomial representations become increasingly ill-conditioned.
- Orthogonal bases remain significantly more stable.
- Constraint matrices amplify existing numerical instabilities.

---

## Figures

- condition_number_vs_degree.pdf
- constraint_gram_condition_number_vs_degree.pdf

---

# Experiment 3: Rank Preservation

## Objective

Determine whether increasing polynomial degree causes numerical rank degradation.

---

## Methodology

Matrix rank is estimated from singular values across increasing polynomial degrees.

Experiments are repeated for all basis representations.

---

## Evaluation

Metrics:

- Effective numerical rank
- Rank fraction

---

## Findings

- Rank degradation occurs much earlier in monomial systems.
- Orthogonal polynomial representations preserve rank more effectively.

---

## Figures

- constraint_gram_rank_fraction_vs_degree.pdf
- rank_fraction_vs_precision.png

---

# Experiment 4: Higher-Dimensional Scaling

## Objective

Evaluate whether numerical stability deteriorates as dimensionality increases.

---

## Methodology

Experiments are repeated for dimensions ranging from 1 to 5.

The total number of basis functions, condition numbers, reconstruction error, and runtime are measured.

---

## Evaluation

Metrics:

- Condition number
- Relative error
- Runtime
- Basis growth

---

## Findings

- Increasing dimensionality significantly increases conditioning problems.
- The combinatorial growth of polynomial terms becomes a major source of instability.
- Orthogonal bases scale more effectively than monomial representations.

---

## Figures

- condition_number_vs_dimension.png
- relative_error_vs_dimension.png
- runtime_and_basis_count_vs_dimension.png

---

# Experiment 5: Precision Studies

## Objective

Determine how floating-point precision affects constrained probabilistic inference.

---

## Methodology

Experiments are repeated under:

- float32
- float64

---

## Evaluation

Metrics:

- Relative error
- Numerical rank
- Runtime

---

## Findings

- Reduced precision accelerates the onset of numerical instability.
- Orthogonal polynomial representations are less sensitive to precision reduction.

---

## Figures

- relative_error_vs_precision.png
- runtime_vs_precision.png
- rank_fraction_vs_precision.png

---

# Experiment 6: Perturbation Analysis

## Objective

Measure the sensitivity of polynomial systems to coefficient perturbations.

---

## Methodology

Controlled Gaussian noise is introduced into polynomial coefficients.

Recovered coefficients are compared against known ground-truth solutions.

---

## Evaluation

Metrics:

- Noise amplification
- Reconstruction error

---

## Findings

- Small perturbations can produce substantial reconstruction errors in ill-conditioned systems.
- Orthogonal polynomial bases exhibit significantly improved robustness.

---

## Figures

- perturbation_sensitivity_vs_degree.pdf
- perturbation_sensitivity_vs_dimension.png
- perturbation_sensitivity_vs_precision.png

---

# Experiment 7: Downstream Probabilistic Inference

## Objective

Determine whether numerical instability propagates into probabilistic inference.

---

## Methodology

Instead of evaluating only polynomial reconstruction, probabilistic quantities are directly measured.

---

## Evaluation

Metrics:

- Partition function error
- Query probability error

---

## Findings

- Numerical instability directly affects downstream probabilistic inference.
- Errors observed in coefficient reconstruction propagate into normalization constants and query probabilities.

---

## Figures

- partition_function_error_vs_degree.png
- query_probability_error_vs_degree.png
- query_probability_error_vs_query_scale.png
- query_probability_sensitivity_vs_degree.png
- downstream_runtime_vs_degree.png
- downstream_error_summary.png

---

# Experiment 8: Constrained Polytope Benchmarks

## Objective

Extend the analysis beyond simplex constraints into more general SMT(LRA)-style constrained domains.

---

## Methodology

Polynomial inference is evaluated on arbitrary convex polytopes, including:

- Static constraints
- Dynamic constraints
- Scheduling problems
- Obstacle-avoidance problems
- Trajectory constraints

---

## Evaluation

Metrics:

- Condition number
- Rank fraction
- Integration error
- Runtime
- Perturbation sensitivity

---

## Findings

- Constraint geometry strongly influences numerical stability.
- More complex feasible regions amplify instability.

---

## Figures

- polytope_condition_number_vs_degree.pdf
- polytope_rank_fraction_vs_degree.pdf
- polytope_integration_error_vs_degree.pdf
- polytope_perturbation_sensitivity_vs_degree.pdf
- polytope_runtime_vs_degree.pdf
- polytope_runtime_vs_dimension.png
- polytope_dynamic_schedule_error.pdf
- polytope_dynamic_trajectory_error.pdf
- polytope_error_summary.pdf

---

# Experiment 9: PAL Benchmarks

## Objective

Evaluate numerical stability within the PAL framework itself.

---

## Methodology

Representative PAL workloads are used to determine whether the numerical trends observed in synthetic experiments remain visible in real probabilistic inference systems.

---

## Evaluation

Metrics:

- Partition function error
- Query probability error
- Conditioning

---

## Findings

- The numerical behavior observed in synthetic experiments is reproduced inside PAL.
- Conditioning remains the primary predictor of downstream probabilistic error.
- Orthogonal polynomial representations consistently outperform monomial representations.

---

## Figures

- pal_condition_number_vs_degree.pdf
- pal_partition_function_error_vs_degree.pdf
- pal_query_probability_error_vs_degree.pdf

---

# Reproducibility

All experiments were designed to be fully reproducible.

The repository includes:

- Fixed random seeds
- Deterministic experiment configurations
- Stored benchmark outputs
- Plot-generation scripts
- Experiment metadata
- Configuration files

All figures presented in the accompanying MSc dissertation can be reproduced directly from the code contained in this repository.
