# Experimental Results

---

# Overview

This document summarizes the principal findings from the experimental evaluation of polynomial basis representations in constrained probabilistic inference.

The experiments compared three polynomial representations:

- Monomial
- Legendre
- Chebyshev

across multiple experimental settings, including:

- Increasing polynomial degree
- Higher-dimensional domains
- Reduced numerical precision
- Coefficient perturbations
- Downstream probabilistic inference
- Constrained polytope integration
- Representative PAL workloads

The central question was whether orthogonal polynomial representations improve the numerical stability of constrained probabilistic inference.

---

# Summary of Principal Findings

| Finding | Supported |
| --- | :---: |
| Orthogonal bases improve conditioning | ✓ |
| Orthogonal bases preserve numerical rank | ✓ |
| Orthogonal bases scale better with degree | ✓ |
| Orthogonal bases scale better with dimension | ✓ |
| Orthogonal bases reduce perturbation sensitivity | ✓ |
| Conditioning improvements improve downstream inference | Partially |
| Basis choice alone solves numerical instability | ✗ |

---

# Result 1: Orthogonal Bases Improve Conditioning

## Observation

Condition numbers increased rapidly as polynomial degree increased.

Monomial representations consistently exhibited the worst numerical conditioning.

Legendre and Chebyshev representations remained substantially better conditioned across all degrees.

---

## Interpretation

The experimental results confirm the classical numerical analysis literature:

- Monomial bases produce poorly conditioned Vandermonde systems.
- Orthogonal polynomial bases significantly improve matrix conditioning.

---

## Supporting Figures

- condition_number_vs_degree.pdf
- constraint_gram_condition_number_vs_degree.pdf

---

# Result 2: Constraints Amplify Numerical Instability

## Observation

Introducing constraints increased matrix condition numbers across all polynomial representations.

Constraint-induced Gram matrices were consistently more ill-conditioned than their unconstrained counterparts.

---

## Interpretation

The constrained inference problem itself contributes to numerical instability.

Numerical errors do not originate exclusively from polynomial representation.

The geometry of the constrained domain also plays a significant role.

---

## Supporting Figures

- constraint_gram_condition_number_vs_degree.pdf

---

# Result 3: Orthogonal Bases Preserve Numerical Rank

## Observation

Increasing polynomial degree reduced the effective numerical rank of polynomial systems.

Rank loss occurred earlier in monomial representations.

Legendre and Chebyshev representations maintained a larger fraction of their theoretical rank.

---

## Interpretation

Numerical rank provides an additional explanation for the instability observed in monomial systems.

Improved conditioning and improved rank preservation are closely related.

---

## Supporting Figures

- constraint_gram_rank_fraction_vs_degree.pdf
- rank_fraction_vs_precision.png

---

# Result 4: Dimensionality Significantly Increases Instability

## Observation

Condition numbers increased as dimensionality increased.

The total number of basis functions also grew rapidly.

Runtime increased alongside dimensionality.

---

## Interpretation

The experiments demonstrated that increasing dimensionality introduces two simultaneous challenges:

1. Worse conditioning.
2. Combinatorial growth in basis size.

These effects compound one another.

---

## Supporting Figures

- condition_number_vs_dimension.png
- relative_error_vs_dimension.png
- runtime_and_basis_count_vs_dimension.png

---

# Result 5: Reduced Precision Accelerates Numerical Degradation

## Observation

Reduced precision (float32) accelerated the onset of numerical instability.

Relative errors increased.

Numerical rank deteriorated more rapidly.

---

## Interpretation

Floating-point precision strongly influences constrained probabilistic inference.

Systems operating in reduced precision become unstable at lower polynomial degrees.

---

## Supporting Figures

- relative_error_vs_precision.png
- runtime_vs_precision.png
- rank_fraction_vs_precision.png

---

# Result 6: Perturbations Are Amplified by Ill-Conditioned Systems

## Observation

Small coefficient perturbations produced disproportionately large reconstruction errors.

Monomial representations amplified perturbations much more aggressively.

Orthogonal polynomial representations were significantly more robust.

---

## Interpretation

Condition numbers are not merely theoretical quantities.

They directly predict how numerical errors propagate through constrained probabilistic systems.

---

## Supporting Figures

- perturbation_sensitivity_vs_degree.pdf
- perturbation_sensitivity_vs_dimension.png
- perturbation_sensitivity_vs_precision.png

---

# Result 7: Downstream Inference Errors Follow Numerical Errors

## Observation

Partition-function errors increased as conditioning deteriorated.

Query-probability errors followed similar trends.

---

## Interpretation

Representation-level numerical instability propagates into probabilistic inference.

Errors introduced during polynomial construction eventually affect:

- Normalization constants
- Partition functions
- Probability queries

---

## Supporting Figures

- partition_function_error_vs_degree.png
- query_probability_error_vs_degree.png
- query_probability_error_vs_query_scale.png
- query_probability_sensitivity_vs_degree.png

---

# Result 8: Better Conditioning Does Not Guarantee Better End-to-End Inference

## Observation

The downstream experiments revealed that improved conditioning alone did not always produce proportionally better probabilistic inference.

In some cases, coefficient recovery, basis conversion, normalization, and constrained integration introduced additional numerical errors.

---

## Interpretation

Improving the polynomial basis is necessary but not sufficient.

Constrained probabilistic inference should be treated as an end-to-end computational pipeline.

Multiple numerical components contribute to the final inference error.

---

## Supporting Figures

- downstream_runtime_vs_degree.png
- downstream_error_summary.png

---

# Result 9: Constraint Geometry Matters

## Observation

Constrained polytope experiments demonstrated that the shape of the feasible region significantly affected numerical stability.

Static constraints, dynamic constraints, scheduling constraints, and obstacle-avoidance constraints produced different numerical behaviors.

---

## Interpretation

The geometry of the constrained domain directly influences:

- Conditioning
- Integration accuracy
- Runtime
- Error propagation

Constraint complexity should therefore be considered when designing probabilistic inference systems.

---

## Supporting Figures

- polytope_condition_number_vs_degree.pdf
- polytope_rank_fraction_vs_degree.pdf
- polytope_integration_error_vs_degree.pdf
- polytope_perturbation_sensitivity_vs_degree.pdf
- polytope_dynamic_schedule_error.pdf
- polytope_dynamic_trajectory_error.pdf

---

# Result 10: PAL Reproduces the Same Numerical Trends

## Observation

Representative PAL benchmarks reproduced the numerical behavior observed in the synthetic experiments.

Partition-function errors and query-probability errors followed the same conditioning trends.

---

## Interpretation

The numerical phenomena identified in controlled experiments are not artifacts of synthetic benchmark construction.

They remain visible within a real constrained probabilistic inference framework.

---

## Supporting Figures

- pal_condition_number_vs_degree.pdf
- pal_partition_function_error_vs_degree.pdf
- pal_query_probability_error_vs_degree.pdf

---

# Key Conclusions

The experiments support five primary conclusions.

---

## 1. Polynomial basis choice matters.

Orthogonal polynomial representations consistently outperform monomial representations.

---

## 2. Conditioning is a useful predictor of numerical stability.

Condition numbers strongly correlate with reconstruction accuracy and perturbation sensitivity.

---

## 3. Dimensionality and constraints amplify instability.

Increasing dimensions and increasing constraint complexity both worsen numerical behavior.

---

## 4. Downstream inference is sensitive to numerical error.

Representation-level numerical errors propagate into partition functions and probability queries.

---

## 5. Numerical stability must be treated as an end-to-end problem.

The experiments suggest that future constrained probabilistic inference systems should optimize the entire computational pipeline rather than focusing exclusively on polynomial basis selection.

---

# Future Work

Potential future research directions include:

- Adaptive basis selection
- Sparse polynomial representations
- Alternative orthogonal bases
- Stable coefficient-construction algorithms
- Arbitrary-precision arithmetic
- GPU implementations
- Runtime-adaptive basis switching
- Integration into larger neuro-symbolic inference systems
