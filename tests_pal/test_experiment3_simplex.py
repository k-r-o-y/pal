from __future__ import annotations

import math

import numpy as np
import pytest

from pal.experiments.experiment3_simplex import (
    coefficients_for_basis,
    duffy_nodes_and_weights,
    exact_separable_simplex_integral,
    integrate_basis_over_simplex,
)


def test_simplex_volume_from_exact_formula() -> None:
    coefficients = np.array([1.0])
    for dimension in (1, 2, 3, 4, 5):
        value = exact_separable_simplex_integral(
            coefficients=coefficients,
            dimension=dimension,
            scale=1.0,
            decimal_places=80,
        )
        assert value == pytest.approx(
            1.0 / math.factorial(dimension),
            rel=1e-14,
            abs=1e-14,
        )


@pytest.mark.parametrize("dimension", [2, 3, 4])
@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
def test_duffy_weights_reproduce_simplex_volume(
    dimension: int,
    scale: float,
) -> None:
    _, weights = duffy_nodes_and_weights(
        dimension=dimension,
        scale=scale,
        order=8,
        dtype=np.dtype(np.float64),
    )
    expected = scale ** dimension / math.factorial(dimension)
    assert float(np.sum(weights)) == pytest.approx(
        expected,
        rel=1e-12,
        abs=1e-12,
    )


@pytest.mark.parametrize("basis", ["monomial", "legendre", "chebyshev"])
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("scale", [0.5, 1.0, 3.0])
def test_quadrature_matches_exact_reference(
    basis: str,
    dimension: int,
    scale: float,
) -> None:
    source = np.array([0.5, -0.25, 0.125], dtype=np.float64)
    converted = coefficients_for_basis(source, scale, basis)

    exact = exact_separable_simplex_integral(
        coefficients=source,
        dimension=dimension,
        scale=scale,
        decimal_places=100,
    )
    computed = integrate_basis_over_simplex(
        coefficients=converted,
        dimension=dimension,
        scale=scale,
        basis=basis,
        precision="float64",
        quadrature_order=8,
    )

    assert computed == pytest.approx(exact, rel=1e-11, abs=1e-12)


def test_float32_path_returns_finite_value() -> None:
    source = np.array([0.5, -0.25, 0.125], dtype=np.float64)
    converted = coefficients_for_basis(source, 1.0, "chebyshev")

    computed = integrate_basis_over_simplex(
        coefficients=converted,
        dimension=3,
        scale=1.0,
        basis="chebyshev",
        precision="float32",
        quadrature_order=6,
    )

    assert np.isfinite(computed)
