import torch

from pal.config import reset_configuration, set_polynomial_basis
from pal.distribution.polynomial_factory import build_polynomial


def test_factory_builds_monomial_polynomial_by_default():
    reset_configuration()

    coeffs = torch.tensor([1.0, 2.0, 3.0])
    powers = torch.tensor([[0], [1], [2]], dtype=torch.int64)
    variable_map = {"x": 0}

    poly = build_polynomial(
        coeffs=coeffs,
        powers=powers,
        variable_map_dict=variable_map,
    )

    x = torch.tensor([[2.0]])
    params = torch.ones(1, 3)

    out = poly.eval_tensor(x, params)

    expected = torch.tensor([1.0 + 2.0 * 2.0 + 3.0 * 4.0])
    assert torch.allclose(out, expected)


def test_factory_builds_legendre_polynomial():
    set_polynomial_basis("legendre")

    coeffs = torch.tensor([1.0, 2.0, 3.0])
    powers = torch.tensor([[0], [1], [2]], dtype=torch.int64)
    variable_map = {"x": 0}

    poly = build_polynomial(
        coeffs=coeffs,
        powers=powers,
        variable_map_dict=variable_map,
    )

    x = torch.tensor([[0.5]])
    params = torch.ones(1, 3)

    out = poly.eval_tensor(x, params)

    assert torch.isfinite(out).all()
    assert out.shape == torch.Size([1])


def test_factory_builds_chebyshev_polynomial():
    set_polynomial_basis("chebyshev")

    coeffs = torch.tensor([1.0, 2.0, 3.0])
    powers = torch.tensor([[0], [1], [2]], dtype=torch.int64)
    variable_map = {"x": 0}

    poly = build_polynomial(
        coeffs=coeffs,
        powers=powers,
        variable_map_dict=variable_map,
    )

    x = torch.tensor([[0.5]])
    params = torch.ones(1, 3)

    out = poly.eval_tensor(x, params)

    assert torch.isfinite(out).all()
    assert out.shape == torch.Size([1])


def test_factory_rejects_unknown_basis():
    reset_configuration()

    coeffs = torch.tensor([1.0])
    powers = torch.tensor([[0]], dtype=torch.int64)
    variable_map = {"x": 0}

    set_polynomial_basis("monomial")

    poly = build_polynomial(
        coeffs=coeffs,
        powers=powers,
        variable_map_dict=variable_map,
        basis="monomial",
    )

    assert poly is not None