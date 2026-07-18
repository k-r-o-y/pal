from typing import Callable, TypeVar
from pysmt.shortcuts import Bool, Real, get_env
from pysmt.fnode import FNode
import torch

from pal.logic.lra import box_to_lra
from pal.logic.lra_pysmt import translate_to_pysmt
from pal.distribution.constrained_distribution import (
    ConditionalConstraintedDistribution,
    ConstrainedDistributionBuilder,
)
from pal.wmi.gasp.gasp.torch.wmipa.numerical_symb_integrator_pa import (
    FunctionMode,
    NumericalSymbIntegratorPA,
)
from wmipa.wmi import WMI as WMI_PA
from wmipa import WMI


def compute_integral(
    constraints: FNode,
    f: Callable[[torch.Tensor], torch.Tensor],
    output_shape: tuple[int, ...],
    total_degree: int,
    variable_map: dict[str, int],
    device: torch.device,
    precision: torch.dtype = torch.float64,
    gasp_kwargs: dict | None = None,
    wmi_pa_mode: str = "SAE4WMI",
    basis: str = "monomial",
) -> torch.Tensor:
    get_env().enable_infix_notation = True

    if gasp_kwargs is None:
        gasp_kwargs = {}

    mode = FunctionMode(f, output_shape)

    integrator = NumericalSymbIntegratorPA(
        mode=mode,
        total_degree=total_degree,
        variable_map=variable_map,
        **gasp_kwargs,
    )

    integrator.set_device(device)
    integrator.set_dtype(precision)

    wmi = WMI(chi=constraints, weight=Real(1), integrator=integrator)
    phi = Bool(True)

    assert wmi_pa_mode in WMI_PA.MODES
    with torch.no_grad():
        result_pa, _ = wmi.computeWMI(phi, mode=wmi_pa_mode)

    return result_pa.to(device)


A = TypeVar("A", bound=ConditionalConstraintedDistribution)


def integrate_distribution(
    d: ConstrainedDistributionBuilder[A],
    device: torch.device,
    precision: torch.dtype = torch.float64,
    gasp_kwargs: dict | None = None,
    wmi_pa_mode: str = "SAE4WMI",
    basis: str = "monomial",
) -> A:
    """
    Integrate the distribution over the constraints.

    The basis option is passed through so experiments can run the same
    integration pipeline under monomial, Legendre, or Chebyshev polynomial
    parameterisations.
    """
    constraints_all_lra = d.constraints
    variable_map = d.var_positions
    total_degree = d.total_degree

    integrals = {}

    for box, f, shape in d.enumerate_pieces():
        constraints_boxed = constraints_all_lra.expression & box_to_lra(box)
        constraints_smt, _ = translate_to_pysmt(constraints_boxed)

        box_integral = compute_integral(
            constraints=constraints_smt,
            f=f,
            output_shape=shape,
            total_degree=total_degree,
            variable_map=variable_map,
            device=device,
            precision=precision,
            gasp_kwargs=gasp_kwargs,
            wmi_pa_mode=wmi_pa_mode,
            basis=basis,
        )

        integrals[box.id] = box_integral

    return d.get_distribution(integrated=integrals)