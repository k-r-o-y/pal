from enum import Enum
from typing import Iterable

import numpy as np

from pal.config import get_polynomial_basis
from pal.distribution.basis_polynomials import (
    chebyshev_eval,
    convert_monomial_to_chebyshev,
    convert_monomial_to_legendre,
    integrate_chebyshev,
    integrate_legendre,
    integrate_monomial,
    legendre_eval,
    monomial_eval,
)
from pal.distribution.torch_polynomial import TorchPolynomial


class PolynomialBasis(str, Enum):
    MONOMIAL = "monomial"
    LEGENDRE = "legendre"
    CHEBYSHEV = "chebyshev"


class PolynomialFactory:
    """
    Routing layer for polynomial basis representations.
    """

    def __init__(self, basis: str | PolynomialBasis = PolynomialBasis.MONOMIAL):
        self.basis = PolynomialBasis(basis)

    def convert_from_monomial(self, coeffs: Iterable[float]) -> np.ndarray:
        coeffs = np.asarray(coeffs, dtype=float)

        if self.basis == PolynomialBasis.MONOMIAL:
            return coeffs
        if self.basis == PolynomialBasis.LEGENDRE:
            return convert_monomial_to_legendre(coeffs)
        if self.basis == PolynomialBasis.CHEBYSHEV:
            return convert_monomial_to_chebyshev(coeffs)

        raise ValueError(f"Unsupported polynomial basis: {self.basis}")

    def evaluate(self, coeffs: Iterable[float], x: Iterable[float] | float) -> np.ndarray:
        coeffs = np.asarray(coeffs, dtype=float)

        if self.basis == PolynomialBasis.MONOMIAL:
            return monomial_eval(coeffs, x)
        if self.basis == PolynomialBasis.LEGENDRE:
            return legendre_eval(coeffs, x)
        if self.basis == PolynomialBasis.CHEBYSHEV:
            return chebyshev_eval(coeffs, x)

        raise ValueError(f"Unsupported polynomial basis: {self.basis}")

    def integrate(
        self,
        coeffs: Iterable[float],
        lower: float = -1.0,
        upper: float = 1.0,
    ) -> float:
        coeffs = np.asarray(coeffs, dtype=float)

        if self.basis == PolynomialBasis.MONOMIAL:
            return integrate_monomial(coeffs, lower, upper)
        if self.basis == PolynomialBasis.LEGENDRE:
            return integrate_legendre(coeffs, lower, upper)
        if self.basis == PolynomialBasis.CHEBYSHEV:
            return integrate_chebyshev(coeffs, lower, upper)

        raise ValueError(f"Unsupported polynomial basis: {self.basis}")

    def convert_evaluate_integrate(
        self,
        monomial_coeffs: Iterable[float],
        x: Iterable[float] | float,
        lower: float = -1.0,
        upper: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        basis_coeffs = self.convert_from_monomial(monomial_coeffs)
        values = self.evaluate(basis_coeffs, x)
        integral = self.integrate(basis_coeffs, lower, upper)
        return basis_coeffs, values, integral


def build_polynomial(
    coeffs,
    powers,
    variable_map_dict,
    absolute: bool = True,
    basis: str | PolynomialBasis | None = None,
) -> TorchPolynomial:
    """
    Build a PAL-compatible TorchPolynomial using the selected polynomial basis.

    The current integration point preserves the original PAL TorchPolynomial
    representation while routing basis selection through the dissertation
    configuration layer.
    """

    if basis is None:
        basis = get_polynomial_basis()

    basis = PolynomialBasis(basis)

    if basis in {
        PolynomialBasis.MONOMIAL,
        PolynomialBasis.LEGENDRE,
        PolynomialBasis.CHEBYSHEV,
    }:
        return TorchPolynomial(
            coeffs=coeffs,
            powers=powers,
            variable_map_dict=variable_map_dict,
            absolute=absolute,
        )

    raise ValueError(f"Unsupported polynomial basis: {basis}")