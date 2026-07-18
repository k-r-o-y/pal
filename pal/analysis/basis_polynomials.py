import numpy as np
from numpy.polynomial import Polynomial, Legendre, Chebyshev


def scale_to_minus_one_one(x, lower, upper):
    return (2.0 * x - (upper + lower)) / (upper - lower)


def monomial_eval(coeffs, x):
    return Polynomial(coeffs)(x)


def legendre_eval(coeffs, x, lower=-1.0, upper=1.0):
    x_scaled = scale_to_minus_one_one(x, lower, upper)
    return Legendre(coeffs)(x_scaled)


def chebyshev_eval(coeffs, x, lower=-1.0, upper=1.0):
    x_scaled = scale_to_minus_one_one(x, lower, upper)
    return Chebyshev(coeffs)(x_scaled)


def convert_monomial_to_legendre(monomial_coeffs, lower=-1.0, upper=1.0):
    poly = Polynomial(monomial_coeffs, domain=[lower, upper])
    return poly.convert(kind=Legendre).coef


def convert_monomial_to_chebyshev(monomial_coeffs, lower=-1.0, upper=1.0):
    poly = Polynomial(monomial_coeffs, domain=[lower, upper])
    return poly.convert(kind=Chebyshev).coef


def integrate_monomial(coeffs, lower, upper):
    poly = Polynomial(coeffs)
    anti = poly.integ()
    return anti(upper) - anti(lower)


def integrate_legendre(coeffs, lower=-1.0, upper=1.0):
    poly = Legendre(coeffs, domain=[lower, upper])
    anti = poly.integ()
    return anti(upper) - anti(lower)


def integrate_chebyshev(coeffs, lower=-1.0, upper=1.0):
    poly = Chebyshev(coeffs, domain=[lower, upper])
    anti = poly.integ()
    return anti(upper) - anti(lower)