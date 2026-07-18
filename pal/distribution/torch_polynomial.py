from typing import Dict, List, Self, Any
from frozendict import frozendict
import torch
import numpy as np
from hashlib import sha1


def generate_monomial(
    n: int, max_total_degree: int, max_individual_degree: int
) -> List[List[int]]:
    max_exponent = min(max_individual_degree, max_total_degree)
    if n == 1:
        return [[i] for i in range(max_exponent + 1)]
    return [
        [i] + monomial
        for i in range(max_exponent + 1)
        for monomial in generate_monomial(
            n - 1, max_total_degree - i, max_individual_degree
        )
    ]


def hash_torch_tensor(tensor: torch.Tensor) -> int:
    np_tensor = tensor.detach().cpu().numpy()
    np_tensor = np.ascontiguousarray(np_tensor)
    dig = sha1(np_tensor.tobytes()).digest()
    return int.from_bytes(dig, byteorder="big")


def stable_np_sum(x: np.ndarray, axis: int) -> np.ndarray:
    x_plus = np.sort(x.clip(min=0), axis=axis)
    x_minus = np.sort(x.clip(max=0), axis=axis)[..., ::-1]
    return np.sum(x_plus, axis=axis) + np.sum(x_minus, axis=axis)


class StateMixin:
    @classmethod
    def from_state(cls, state_dict: Any, **kwargs) -> Self:
        init_args = {key: state_dict[key] for key in state_dict.keys()}
        init_args.update(kwargs)
        return cls(**init_args)


class VariableMapMixin:
    def __init__(self, variable_map_dict: Dict[str, int] | frozendict[str, int]) -> None:
        if not isinstance(variable_map_dict, frozendict):
            variable_map_dict = frozendict(variable_map_dict)
        self._variable_map_dict = variable_map_dict

    def get_variable_map_dict(self) -> Dict[str, int]:
        return self._variable_map_dict  # type: ignore

    def to_state(self) -> Any:
        return {"variable_map_dict": self._variable_map_dict}


class TorchPolynomial(torch.nn.Module, VariableMapMixin, StateMixin):
    coeffs: torch.Tensor
    powers: torch.Tensor

    def __init__(
        self,
        coeffs: torch.Tensor,
        powers: torch.Tensor,
        variable_map_dict: Dict[str, int] | frozendict[str, int],
        absolute: bool = True,
    ) -> None:
        torch.nn.Module.__init__(self)
        VariableMapMixin.__init__(self, variable_map_dict)
        self.register_buffer("coeffs", coeffs)
        self.register_buffer("powers", powers)
        assert coeffs.shape[0] == powers.shape[0]
        self.absolute = absolute

    @classmethod
    def construct(
        cls, max_order: int, max_terms: int, var_map_dict: Dict[str, int]
    ) -> Self:
        n = len(var_map_dict)
        monomials = generate_monomial(n, max_order, max_terms)
        coeffs = torch.ones(len(monomials))
        powers = torch.tensor(monomials, dtype=torch.int64)

        return cls(
            coeffs=coeffs,
            powers=powers,
            variable_map_dict=var_map_dict,
        )

    def reorder_parameter_positions(
        self, param_index_map: Dict[int, int]
    ) -> "TorchPolynomial":
        assert set(param_index_map.keys()) == set(param_index_map.values())

        new_coeffs = torch.zeros_like(self.coeffs)
        new_powers = torch.zeros_like(self.powers)

        for i in range(self.powers.shape[0]):
            new_position = param_index_map[i]
            new_coeffs[new_position] = self.coeffs[i]
            new_powers[new_position] = self.powers[i]

        return TorchPolynomial(
            coeffs=new_coeffs,
            powers=new_powers,
            variable_map_dict=self._variable_map_dict,
            absolute=self.absolute,
        )

    def eval_tensor(
        self,
        y_tensor: torch.Tensor | None,
        param_tensor: torch.Tensor,
        sq_epsilon: float = -1,
    ) -> torch.Tensor:
        assert y_tensor is not None

        def eval_single_monomial(
            y: torch.Tensor, monomial: torch.Tensor
        ) -> torch.Tensor:
            return torch.pow(y, monomial).prod(dim=-1)

        def eval_polynomial_single(
            y: torch.Tensor, param: torch.Tensor
        ) -> torch.Tensor:
            monomials = torch.vmap(lambda mon: eval_single_monomial(y, mon))(
                self.powers
            )
            weighted = (self.coeffs * param) * monomials
            return weighted.sum(dim=-1)

        polys = torch.vmap(eval_polynomial_single)(y_tensor, param_tensor)

        if self.absolute:
            polys = polys.abs()

        if sq_epsilon != -1:
            with torch.no_grad():
                polys[polys < sq_epsilon] = sq_epsilon

        return polys

    def eval_monomials(self, y_tensor: torch.Tensor) -> torch.Tensor:
        def eval_single_monomial(
            y: torch.Tensor, monomial: torch.Tensor
        ) -> torch.Tensor:
            return torch.pow(y, monomial).prod(dim=-1)

        return torch.vmap(
            lambda mon: eval_single_monomial(y_tensor, mon),
            out_dims=-1,
        )(self.powers)

    def square(self) -> "SquaredSymbolicTorchPolynomial":
        return SquaredSymbolicTorchPolynomial(
            coeffs=self.coeffs,
            powers=self.powers,
            variable_map_dict=self._variable_map_dict,
        )

    def to_state(self) -> Any:
        state = super().to_state()
        state.update(self.state_dict())
        return state

    def __hash__(self) -> int:
        return (
            hash_torch_tensor(self.coeffs)
            + hash_torch_tensor(self.powers)
            + hash(self._variable_map_dict)
        )

class SquaredParamsTorchPolynomial(torch.nn.Module, VariableMapMixin, StateMixin):
    coeffs: torch.Tensor
    indices_params: torch.Tensor

    def __init__(
        self,
        coeffs: torch.Tensor,
        indices_params: torch.Tensor,
        variable_map_dict: Dict[str, int] | frozendict[str, int],
    ) -> None:
        torch.nn.Module.__init__(self)
        VariableMapMixin.__init__(self, variable_map_dict)
        self.register_buffer("coeffs", coeffs)
        self.register_buffer("indices_params", indices_params)

    def eval_tensor(
        self,
        param_tensor: torch.Tensor,
        sq_epsilon: float = -1,
    ) -> torch.Tensor:
        def eval_param_instance(param: torch.Tensor) -> torch.Tensor:
            combs = self.coeffs * torch.combinations(
                param, 2, with_replacement=True
            ).prod(dim=-1)
            return combs.sum(dim=-1)

        params_combinations = torch.vmap(eval_param_instance)(param_tensor)

        if sq_epsilon != -1:
            with torch.no_grad():
                params_combinations[params_combinations < sq_epsilon] = sq_epsilon

        return params_combinations

    def to_state(self) -> Any:
        state = super().to_state()
        state.update(self.state_dict())
        return state

    def __hash__(self) -> int:
        return hash_torch_tensor(self.coeffs) + hash(self._variable_map_dict)


class SquaredTorchPolynomial(torch.nn.Module, VariableMapMixin, StateMixin):
    coeffs: torch.Tensor
    powers: torch.Tensor
    params: torch.Tensor

    def __init__(
        self,
        coeffs: torch.Tensor,
        powers: torch.Tensor,
        params: torch.Tensor,
        variable_map_dict: Dict[str, int] | frozendict[str, int],
    ) -> None:
        torch.nn.Module.__init__(self)
        VariableMapMixin.__init__(self, variable_map_dict)

        self.register_buffer("coeffs", coeffs)
        self.register_buffer("powers", powers)

        if params.shape[0] == 1:
            params = params.squeeze(0)

        self.register_buffer("params", params)

        assert coeffs.shape[0] == powers.shape[0]

    def get_max_total_degree(self) -> int:
        return 2 * int(self.powers.sum(dim=-1).max().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def eval_single_monomial(
            y: torch.Tensor,
            monomial: torch.Tensor,
            coeff: torch.Tensor,
            param: torch.Tensor,
        ) -> torch.Tensor:
            return coeff * torch.pow(y, monomial).prod(dim=-1) * param

        def eval_polynomial(y: torch.Tensor, the_params: torch.Tensor) -> torch.Tensor:
            monomials = torch.vmap(
                lambda m, c, p: eval_single_monomial(y, m, c, p)
            )(self.powers, self.coeffs, the_params)
            return monomials.sum(dim=-1) ** 2

        if len(self.params.shape) == 1:
            return torch.vmap(lambda x: eval_polynomial(x, self.params))(x)

        return torch.vmap(
            lambda x, p: eval_polynomial(x, p),
            in_dims=(0, 0),
            out_dims=1,
        )(x, self.params)

    def to_state(self) -> Any:
        state = super().to_state()
        state.update(self.state_dict())
        return state


class SquaredSymbolicTorchPolynomial(torch.nn.Module, VariableMapMixin, StateMixin):
    coeffs: torch.Tensor
    powers: torch.Tensor
    combinations_coefficient: torch.Tensor
    indices_params: torch.Tensor

    def __init__(
        self,
        coeffs: torch.Tensor,
        powers: torch.Tensor,
        variable_map_dict: Dict[str, int] | frozendict[str, int],
        indices_params: torch.Tensor | None = None,
        combinations_coefficient: torch.Tensor | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        VariableMapMixin.__init__(self, variable_map_dict)

        self.register_buffer("coeffs", coeffs)
        self.register_buffer("powers", powers)

        if indices_params is None:
            assert combinations_coefficient is None

            indexes = torch.arange(0, powers.shape[0], device=powers.device)
            combinations = torch.combinations(indexes, 2, with_replacement=True)

            self.register_buffer("indices_params", combinations, persistent=False)

            combinations_coefficient = torch.ones(
                combinations.shape[0],
                dtype=torch.float32,
                device=powers.device,
            )
            combinations_coefficient[combinations[:, 0] != combinations[:, 1]] = 2

            self.register_buffer(
                "combinations_coefficient",
                combinations_coefficient,
                persistent=False,
            )
        else:
            assert combinations_coefficient is not None
            assert indices_params.shape[0] == combinations_coefficient.shape[0]

            self.register_buffer("indices_params", indices_params, persistent=False)
            self.register_buffer(
                "combinations_coefficient",
                combinations_coefficient,
                persistent=False,
            )

    def get_max_total_degree(self) -> int:
        return 2 * int(self.powers.sum(dim=-1).max().item())

    def get_num_vars(self) -> int:
        return self.powers.shape[-1]

    def to_integrated_polynomial(
        self,
        coeff: torch.Tensor,
    ) -> SquaredParamsTorchPolynomial:
        return SquaredParamsTorchPolynomial(
            coeffs=coeff,
            indices_params=self.indices_params,
            variable_map_dict=self._variable_map_dict,
        )

    def to_applied_polynomial(
        self,
        params: torch.Tensor,
    ) -> SquaredTorchPolynomial:
        return SquaredTorchPolynomial(
            coeffs=self.coeffs,
            powers=self.powers,
            params=params,
            variable_map_dict=self._variable_map_dict,
        )

    def eval_tensor(
        self,
        y_tensor: torch.Tensor,
        param_tensor: torch.Tensor | None,
        sq_epsilon: float = -1,
        vectorized: bool = False,
    ) -> torch.Tensor:
        def eval_single_monomial(
            y: torch.Tensor,
            monomial: torch.Tensor,
        ) -> torch.Tensor:
            return torch.pow(y, monomial).prod(dim=-1)

        if not vectorized:
            assert param_tensor is not None

            def eval_polynomial_single(
                y: torch.Tensor,
                param: torch.Tensor,
            ) -> torch.Tensor:
                monomials = torch.vmap(lambda mon: eval_single_monomial(y, mon))(
                    self.powers
                )
                weighted = (self.coeffs * param) * monomials
                return weighted.sum(dim=-1) ** 2

            polys = torch.vmap(eval_polynomial_single)(y_tensor, param_tensor)

            if sq_epsilon != -1:
                with torch.no_grad():
                    polys[polys < sq_epsilon] = sq_epsilon

            return polys

        assert param_tensor is None
        assert sq_epsilon == -1

        def eval_vectorized_polynomial(y: torch.Tensor) -> torch.Tensor:
            monomials = torch.vmap(lambda mon: eval_single_monomial(y, mon))(
                self.powers
            )
            monomials = self.coeffs * monomials
            monomials_combinations = torch.combinations(
                monomials,
                2,
                with_replacement=True,
            ).prod(dim=-1)
            return monomials_combinations * self.combinations_coefficient

        return torch.vmap(
            eval_vectorized_polynomial,
            in_dims=0,
            out_dims=1,
        )(y_tensor)

    def eval_tensor_vectorized(
        self,
        y_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self.eval_tensor(y_tensor, None, vectorized=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.eval_tensor_vectorized(x)

    def to_state(self) -> Any:
        state = super().to_state()
        state.update(self.state_dict())
        return state

    def __hash__(self) -> int:
        return (
            hash_torch_tensor(self.coeffs)
            + hash_torch_tensor(self.powers)
            + hash_torch_tensor(self.combinations_coefficient)
            + hash(self._variable_map_dict)
        )


def calculate_coefficients_from_hermite_spline(
    knots: torch.Tensor,
    differences: torch.Tensor,
    y: torch.Tensor,
    dy: torch.Tensor,
    shift: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert y.shape[0] == knots.shape[0] and dy.shape[0] == knots.shape[0]

    dy0 = dy[:-1] * differences
    dy1 = dy[1:] * differences
    y0 = y[:-1]
    y1 = y[1:]

    a = y0
    b = dy0
    c = 3 * (y1 - y0) - 2 * dy0 - dy1
    d = 2 * (y0 - y1) + dy0 + dy1

    if shift:
        x0 = knots[:-1]

        transformed_a = (
            a
            - b * (1 / differences) * x0
            + c * (1 / differences**2) * x0**2
            - d * (1 / differences**3) * x0**3
        )
        transformed_b = (
            b * (1 / differences)
            - 2 * c * (1 / differences**2) * x0
            + 3 * d * (1 / differences**3) * x0**2
        )
        transformed_c = c * (1 / differences**2) - 3 * d * (
            1 / differences**3
        ) * x0
        transformed_d = d * (1 / differences**3)

        return transformed_a, transformed_b, transformed_c, transformed_d

    transformed_a = a
    transformed_b = b * 1 / differences
    transformed_c = c * 1 / differences**2
    transformed_d = d * 1 / differences**3

    return transformed_a, transformed_b, transformed_c, transformed_d


def calc_2d_spline_component(
    x: torch.Tensor,
    knot_idx: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    def per_param(p: torch.Tensor) -> torch.Tensor:
        selected_idx = knot_idx.unsqueeze(-1)
        selected_params = torch.gather(p, dim=1, index=selected_idx)
        return selected_params.squeeze(-1)

    poly_params = torch.vmap(per_param, in_dims=-1, out_dims=-1)(params)

    poly_per_component = (
        poly_params[..., 0]
        + poly_params[..., 1] * x
        + poly_params[..., 2] * x**2
        + poly_params[..., 3] * x**3
    )

    return poly_per_component.prod(dim=-1)


def calc_log_mixture(
    x: torch.Tensor,
    knot_idx: torch.Tensor,
    m_params: torch.Tensor,
    m_weights_log: torch.Tensor,
    m_normalization: torch.Tensor,
    eps: float = -1,
) -> torch.Tensor:
    mixture_poly: torch.Tensor = torch.vmap(
        lambda p: calc_2d_spline_component(x, knot_idx, p)
    )(m_params)

    m_normalization_coeff = m_normalization.sum(dim=-1).sum(dim=-1)

    if eps != -1:
        m_normalization_coeff = m_normalization_coeff.clamp(min=eps)

    densities_components = mixture_poly**2 / m_normalization_coeff

    return torch.logsumexp(
        torch.log(densities_components) + m_weights_log,
        dim=0,
    )