import numpy as np

from pal.distribution.polynomial_factory import PolynomialFactory


def test_monomial_legendre_chebyshev_evaluate_same_polynomial():
    """
    Checks that converting a polynomial from the monomial basis into Legendre
    and Chebyshev bases preserves the represented function.
    """

    monomial_coeffs = np.array([1.0, -0.5, 0.25, 0.75, -0.2, 0.1])
    x = np.linspace(-1.0, 1.0, 200)

    monomial_factory = PolynomialFactory("monomial")
    legendre_factory = PolynomialFactory("legendre")
    chebyshev_factory = PolynomialFactory("chebyshev")

    monomial_values = monomial_factory.evaluate(monomial_coeffs, x)

    legendre_coeffs = legendre_factory.convert_from_monomial(monomial_coeffs)
    chebyshev_coeffs = chebyshev_factory.convert_from_monomial(monomial_coeffs)

    legendre_values = legendre_factory.evaluate(legendre_coeffs, x)
    chebyshev_values = chebyshev_factory.evaluate(chebyshev_coeffs, x)

    np.testing.assert_allclose(
        monomial_values,
        legendre_values,
        rtol=1e-10,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        monomial_values,
        chebyshev_values,
        rtol=1e-10,
        atol=1e-10,
    )


def test_monomial_legendre_chebyshev_integrate_same_polynomial():
    """
    Checks that integration is preserved after basis conversion.
    """

    monomial_coeffs = np.array([1.0, -0.5, 0.25, 0.75, -0.2, 0.1])

    monomial_factory = PolynomialFactory("monomial")
    legendre_factory = PolynomialFactory("legendre")
    chebyshev_factory = PolynomialFactory("chebyshev")

    monomial_integral = monomial_factory.integrate(monomial_coeffs)

    legendre_coeffs = legendre_factory.convert_from_monomial(monomial_coeffs)
    chebyshev_coeffs = chebyshev_factory.convert_from_monomial(monomial_coeffs)

    legendre_integral = legendre_factory.integrate(legendre_coeffs)
    chebyshev_integral = chebyshev_factory.integrate(chebyshev_coeffs)

    np.testing.assert_allclose(
        monomial_integral,
        legendre_integral,
        rtol=1e-10,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        monomial_integral,
        chebyshev_integral,
        rtol=1e-10,
        atol=1e-10,
    )