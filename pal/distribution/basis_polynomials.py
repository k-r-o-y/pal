"""
Polynomial basis utilities for dissertation experiments.

This module provides NumPy-based helpers for converting monomial polynomial
coefficients into Legendre and Chebyshev representations, evaluating those
representations, and integrating them on an interval.

Coefficient convention:
    coeffs[k] is the coefficient of x**k.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import Polynomial, Legendre, Chebyshev


def monomial_eval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate a monomial-basis polynomial."""
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    return Polynomial(coeffs)(x)


def convert_monomial_to_legendre(coeffs: np.ndarray) -> np.ndarray:
    """Convert monomial coefficients to Legendre coefficients."""
    coeffs = np.asarray(coeffs, dtype=float)
    return Polynomial(coeffs).convert(kind=Legendre).coef


def convert_monomial_to_chebyshev(coeffs: np.ndarray) -> np.ndarray:
    """Convert monomial coefficients to Chebyshev coefficients."""
    coeffs = np.asarray(coeffs, dtype=float)
    return Polynomial(coeffs).convert(kind=Chebyshev).coef


def legendre_eval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate a Legendre-basis polynomial."""
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    return Legendre(coeffs)(x)


def chebyshev_eval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate a Chebyshev-basis polynomial."""
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    return Chebyshev(coeffs)(x)


def integrate_monomial(coeffs: np.ndarray, lower: float = -1.0, upper: float = 1.0) -> float:
    """Integrate a monomial-basis polynomial over [lower, upper]."""
    coeffs = np.asarray(coeffs, dtype=float)
    integral_poly = Polynomial(coeffs).integ()
    return float(integral_poly(upper) - integral_poly(lower))


def integrate_legendre(coeffs: np.ndarray, lower: float = -1.0, upper: float = 1.0) -> float:
    """Integrate a Legendre-basis polynomial over [lower, upper]."""
    coeffs = np.asarray(coeffs, dtype=float)
    integral_poly = Legendre(coeffs).integ()
    return float(integral_poly(upper) - integral_poly(lower))


def integrate_chebyshev(coeffs: np.ndarray, lower: float = -1.0, upper: float = 1.0) -> float:
    """Integrate a Chebyshev-basis polynomial over [lower, upper]."""
    coeffs = np.asarray(coeffs, dtype=float)
    integral_poly = Chebyshev(coeffs).integ()
    return float(integral_poly(upper) - integral_poly(lower))


def rescale_to_unit_interval(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Map x from [lower, upper] to [-1, 1]."""
    x = np.asarray(x, dtype=float)
    if upper == lower:
        raise ValueError("upper and lower must be different.")
    return (2.0 * x - (upper + lower)) / (upper - lower)


def rescale_from_unit_interval(x_tilde: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Map x_tilde from [-1, 1] back to [lower, upper]."""
    x_tilde = np.asarray(x_tilde, dtype=float)
    if upper == lower:
        raise ValueError("upper and lower must be different.")
    return 0.5 * ((upper - lower) * x_tilde + upper + lower)


def basis_eval(
    coeffs: np.ndarray,
    x: np.ndarray,
    basis: str,
) -> np.ndarray:
    """Evaluate coefficients in the requested basis."""
    if basis == "monomial":
        return monomial_eval(coeffs, x)
    if basis == "legendre":
        return legendre_eval(coeffs, x)
    if basis == "chebyshev":
        return chebyshev_eval(coeffs, x)
    raise ValueError(f"Unknown polynomial basis: {basis}")


def basis_integral(
    coeffs: np.ndarray,
    basis: str,
    lower: float = -1.0,
    upper: float = 1.0,
) -> float:
    """Integrate coefficients in the requested basis."""
    if basis == "monomial":
        return integrate_monomial(coeffs, lower, upper)
    if basis == "legendre":
        return integrate_legendre(coeffs, lower, upper)
    if basis == "chebyshev":
        return integrate_chebyshev(coeffs, lower, upper)
    raise ValueError(f"Unknown polynomial basis: {basis}")