"""
Global configuration used for dissertation experiments.

The default behaviour matches the original PAL implementation by using the
monomial polynomial basis. Alternative basis representations can be selected
without modifying the remainder of the PAL pipeline.
"""

from dataclasses import dataclass, field
from typing import Literal


PolynomialBasis = Literal[
    "monomial",
    "legendre",
    "chebyshev",
]


@dataclass(slots=True)
class PolynomialConfig:
    """
    Configuration controlling the polynomial representation used throughout PAL.
    """

    basis: PolynomialBasis = "monomial"


@dataclass(slots=True)
class ExperimentConfig:
    """
    Configuration shared across dissertation experiments.
    """

    polynomial: PolynomialConfig = field(default_factory=PolynomialConfig)


# ---------------------------------------------------------------------
# Global configuration instance
# ---------------------------------------------------------------------

CONFIG = ExperimentConfig()


# ---------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------

def set_polynomial_basis(basis: PolynomialBasis) -> None:
    """
    Set the polynomial basis used throughout the framework.
    """
    CONFIG.polynomial.basis = basis


def get_polynomial_basis() -> PolynomialBasis:
    """
    Return the currently selected polynomial basis.
    """
    return CONFIG.polynomial.basis


def reset_configuration() -> None:
    """
    Restore the original PAL configuration.
    """
    CONFIG.polynomial.basis = "monomial"