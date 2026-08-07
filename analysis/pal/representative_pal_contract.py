from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class AxisAlignedBox:
    """
    Closed axis-aligned box:

        [lower_1, upper_1] × ... × [lower_n, upper_n]
    """

    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.lower) == 0:
            raise ValueError("Box must contain at least one dimension.")

        if len(self.lower) != len(self.upper):
            raise ValueError("Lower/upper bounds must have equal length.")

        for lo, hi in zip(self.lower, self.upper):
            if lo >= hi:
                raise ValueError("Each lower bound must be strictly less than the upper bound.")

    @property
    def dimension(self) -> int:
        return len(self.lower)

    @property
    def volume(self) -> float:
        return float(
            np.prod(
                np.asarray(self.upper) -
                np.asarray(self.lower)
            )
        )


@dataclass(frozen=True)
class BoxMinusObstacle:
    """
    PAL benchmark domain.

    Integration region is

        OuterBox \ Obstacle

    where the obstacle is completely contained inside the outer box.
    """

    outer: AxisAlignedBox
    obstacle: AxisAlignedBox

    def __post_init__(self) -> None:

        if self.outer.dimension != self.obstacle.dimension:
            raise ValueError("Outer box and obstacle must have equal dimensions.")

        for outer_lo, outer_hi, inner_lo, inner_hi in zip(
            self.outer.lower,
            self.outer.upper,
            self.obstacle.lower,
            self.obstacle.upper,
        ):

            if not (
                outer_lo <= inner_lo <
                inner_hi <= outer_hi
            ):
                raise ValueError(
                    "Obstacle must lie completely inside the outer box."
                )


@dataclass(frozen=True)
class PolynomialDensity:
    """
    Polynomial density represented in a monomial basis.

    Density:

        f(x) = Σ c_i x^α_i

    where α_i is a multi-index.
    """

    coefficients: np.ndarray

    multi_indices: tuple[tuple[int, ...], ...]

    dimension: int

    degree: int

    offset: float

    def __post_init__(self) -> None:

        coeffs = np.asarray(self.coefficients)

        if coeffs.ndim != 1:
            raise ValueError("Coefficients must be one-dimensional.")

        if len(coeffs) != len(self.multi_indices):
            raise ValueError(
                "Coefficient count does not match number of monomials."
            )

        if self.dimension < 1:
            raise ValueError("Dimension must be positive.")

        if self.degree < 0:
            raise ValueError("Degree cannot be negative.")

        if self.offset < 0:
            raise ValueError("Offset must be non-negative.")

        if not np.all(np.isfinite(coeffs)):
            raise ValueError("Polynomial coefficients must be finite.")

        for alpha in self.multi_indices:

            if len(alpha) != self.dimension:
                raise ValueError(
                    "Every multi-index must match the declared dimension."
                )

            if any(power < 0 for power in alpha):
                raise ValueError("Negative exponents are not permitted.")

            if sum(alpha) > self.degree:
                raise ValueError(
                    "Multi-index exceeds declared polynomial degree."
                )


@dataclass(frozen=True)
class PALRequest:
    """
    Canonical request object passed to any PAL backend.

    The benchmark converts the polynomial into the requested basis
    before invoking the PAL implementation.
    """

    density: PolynomialDensity

    basis: str

    dtype: str

    basis_coefficients: np.ndarray

    basis_multi_indices: tuple[tuple[int, ...], ...]

    domain: BoxMinusObstacle

    query_box: AxisAlignedBox

    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PALResult:
    """
    Canonical response returned by a PAL backend.
    """

    partition_function: float

    query_mass: float

    query_probability: float

    runtime_ms: float

    extra_metrics: Mapping[str, float]


class PALCallable(Protocol):
    """
    Callable protocol for a PAL backend.
    """

    def __call__(
        self,
        request: PALRequest,
    ) -> PALResult:
        ...


def ensure_pal_result(value: Any) -> PALResult:
    """
    Accept either

        PALResult

    or

        dict

    returned by a backend implementation.
    """

    if isinstance(value, PALResult):
        return value

    if not isinstance(value, Mapping):
        raise TypeError(
            "PAL backend must return either PALResult "
            "or a compatible dictionary."
        )

    required = (
        "partition_function",
        "query_mass",
        "query_probability",
        "runtime_ms",
    )

    missing = [
        key
        for key in required
        if key not in value
    ]

    if missing:
        raise KeyError(
            f"Missing PAL result fields: {missing}"
        )

    extra = value.get("extra_metrics", {})

    if not isinstance(extra, Mapping):
        raise TypeError(
            "extra_metrics must be a mapping."
        )

    result = PALResult(
        partition_function=float(value["partition_function"]),
        query_mass=float(value["query_mass"]),
        query_probability=float(value["query_probability"]),
        runtime_ms=float(value["runtime_ms"]),
        extra_metrics={
            str(k): float(v)
            for k, v in extra.items()
        },
    )

    numbers = (
        result.partition_function,
        result.query_mass,
        result.query_probability,
        result.runtime_ms,
        *result.extra_metrics.values(),
    )

    if not all(np.isfinite(v) for v in numbers):
        raise ValueError(
            "PAL backend returned non-finite values."
        )

    return result