from __future__ import annotations

import numpy as np
import pytest

from pal.experiments.experiment2_interval_integration import (
    coefficients_for_basis,
    high_precision_reference_integral,
    integrate_chebyshev,
    integrate_legendre,
    integrate_monomial,
)


@pytest.mark.parametrize(
    ("coefficients", "left", "right", "expected"),
    [
        (np.array([1.0]), -1.0, 1.0, 2.0),
        (np.array([0.0, 1.0]), 0.0, 1.0, 0.5),
        (np.array([1.0, 2.0, 3.0]), 0.0, 1.0, 3.0),
        (np.array([0.0, 0.0, 1.0]), -1.0, 1.0, 2.0 / 3.0),
    ],
)
def test_high_precision_reference_known_polynomials(
    coefficients: np.ndarray,
    left: float,
    right: float,
    expected: float,
) -> None:
    actual = high_precision_reference_integral(
        coefficients,
        left,
        right,
        decimal_places=80,
    )
    assert actual == pytest.approx(expected, rel=1e-14, abs=1e-14)


@pytest.mark.parametrize("basis", ["monomial", "legendre", "chebyshev"])
@pytest.mark.parametrize(
    ("left", "right"),
    [
        (-1.0, 1.0),
        (0.0, 1.0),
        (0.0, 10.0),
        (10.0, 20.0),
    ],
)
def test_all_bases_match_reference(
    basis: str,
    left: float,
    right: float,
) -> None:
    source = np.array([0.5, -0.25, 0.125, -0.0625], dtype=np.float64)
    coefficients = coefficients_for_basis(
        source,
        left,
        right,
        basis,
    )

    reference = high_precision_reference_integral(
        source,
        left,
        right,
        decimal_places=100,
    )

    if basis == "monomial":
        computed = integrate_monomial(
            coefficients,
            left,
            right,
            np.dtype(np.float64),
        )
    elif basis == "legendre":
        computed = integrate_legendre(
            coefficients,
            left,
            right,
            np.dtype(np.float64),
        )
    else:
        computed = integrate_chebyshev(
            coefficients,
            left,
            right,
            np.dtype(np.float64),
        )

    assert computed == pytest.approx(
        reference,
        rel=1e-11,
        abs=1e-11,
    )


def test_float32_paths_are_finite() -> None:
    source = np.array([0.5, -0.25, 0.125], dtype=np.float64)

    for basis in ("monomial", "legendre", "chebyshev"):
        coefficients = coefficients_for_basis(
            source,
            0.0,
            10.0,
            basis,
        )

        if basis == "monomial":
            computed = integrate_monomial(
                coefficients,
                0.0,
                10.0,
                np.dtype(np.float32),
            )
        elif basis == "legendre":
            computed = integrate_legendre(
                coefficients,
                0.0,
                10.0,
                np.dtype(np.float32),
            )
        else:
            computed = integrate_chebyshev(
                coefficients,
                0.0,
                10.0,
                np.dtype(np.float32),
            )

        assert np.isfinite(computed)
