from __future__ import annotations

import numpy as np
import pytest

from pal.experiments.experiment1_polynomial_evaluation import (
    basis_evaluation_matrix,
    convert_from_monomial,
    error_metrics,
    evaluate_chebyshev,
    evaluate_legendre,
    evaluate_monomial,
    generate_monomial_coefficients,
)


@pytest.mark.parametrize("degree", [0, 1, 3, 8, 12])
@pytest.mark.parametrize("basis", ["legendre", "chebyshev"])
def test_basis_conversion_preserves_polynomial(
    degree: int,
    basis: str,
) -> None:
    coefficients = generate_monomial_coefficients(
        degree=degree,
        family="random_uniform",
        seed=42,
    )
    converted = convert_from_monomial(coefficients, basis)
    x = np.linspace(-1.0, 1.0, 1001, dtype=np.float64)

    expected = evaluate_monomial(
        coefficients,
        x,
        np.dtype(np.float64),
    )
    actual = (
        evaluate_legendre(converted, x, np.dtype(np.float64))
        if basis == "legendre"
        else evaluate_chebyshev(
            converted,
            x,
            np.dtype(np.float64),
        )
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize(
    "evaluator",
    [
        evaluate_monomial,
        evaluate_legendre,
        evaluate_chebyshev,
    ],
)
def test_evaluator_preserves_requested_dtype(evaluator) -> None:
    coefficients = np.array([1.0, -0.5, 0.25])
    x = np.linspace(-1.0, 1.0, 17, dtype=np.float32)
    output = evaluator(
        coefficients,
        x,
        np.dtype(np.float32),
    )
    assert output.dtype == np.float32


@pytest.mark.parametrize(
    "basis",
    ["monomial", "legendre", "chebyshev"],
)
def test_evaluation_matrix_shape_and_finiteness(
    basis: str,
) -> None:
    x = np.linspace(-1.0, 1.0, 31)
    matrix = basis_evaluation_matrix(
        basis,
        degree=7,
        x=x,
    )
    assert matrix.shape == (31, 8)
    assert np.all(np.isfinite(matrix))


def test_error_metrics_zero_for_identical_arrays() -> None:
    values = np.array([1.0, 2.0, 3.0])
    assert error_metrics(values, values) == (0.0, 0.0, 0.0)
