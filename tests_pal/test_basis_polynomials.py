import numpy as np

from pal.analysis.basis_polynomials import (
    monomial_eval,
    legendre_eval,
    chebyshev_eval,
    convert_monomial_to_legendre,
    convert_monomial_to_chebyshev,
    integrate_monomial,
    integrate_legendre,
    integrate_chebyshev,
)


def test_legendre_conversion_matches_monomial():
    coeffs = np.array([1.0, -2.0, 0.5, 3.0])
    x = np.linspace(-1.0, 1.0, 200)

    leg_coeffs = convert_monomial_to_legendre(coeffs)

    y_mono = monomial_eval(coeffs, x)
    y_leg = legendre_eval(leg_coeffs, x)

    assert np.allclose(y_mono, y_leg, atol=1e-10)


def test_chebyshev_conversion_matches_monomial():
    coeffs = np.array([1.0, -2.0, 0.5, 3.0])
    x = np.linspace(-1.0, 1.0, 200)

    cheb_coeffs = convert_monomial_to_chebyshev(coeffs)

    y_mono = monomial_eval(coeffs, x)
    y_cheb = chebyshev_eval(cheb_coeffs, x)

    assert np.allclose(y_mono, y_cheb, atol=1e-10)


def test_integrals_match_after_conversion():
    coeffs = np.array([1.0, 0.5, -0.25, 0.1])

    leg_coeffs = convert_monomial_to_legendre(coeffs)
    cheb_coeffs = convert_monomial_to_chebyshev(coeffs)

    mono_int = integrate_monomial(coeffs, -1.0, 1.0)
    leg_int = integrate_legendre(leg_coeffs, -1.0, 1.0)
    cheb_int = integrate_chebyshev(cheb_coeffs, -1.0, 1.0)

    assert np.isclose(mono_int, leg_int, atol=1e-10)
    assert np.isclose(mono_int, cheb_int, atol=1e-10)