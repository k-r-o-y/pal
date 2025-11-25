###############################################
#
# This file severs as an example of how to
# train a MLP model using the SDD dataset using
# the PAL library and a spline-distribution.
#
###############################################

if __name__ == "__main__":
    import os
    import sys

    module_path = os.path.abspath(os.path.join("."))
    if module_path not in sys.path:
        sys.path.append(module_path)

import os
import pal.problem.sdd as csdd
import pal.distribution.spline_distribution as spline
from pal.wmi.compute_integral import integrate_distribution
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse
import time


class UnconditionalNetwork(nn.Module):
    def __init__(
        self,
        conditional_spline_dist: spline.ConditionalSplineSQ2D,
        init_positive: bool,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        shape_value, shape_derivative, shape_mixture_weights = (
            conditional_spline_dist.parameter_shape()
        )
        # register trainable parameters
        self.dens_value_tensor = nn.Parameter(torch.rand(shape_value, dtype=dtype))
        self.derivative_tensor = nn.Parameter(torch.randn(shape_derivative, dtype=dtype))
        self.mixture_weights_tensor = nn.Parameter(torch.ones(shape_mixture_weights, dtype=dtype))
        # mixture_weights_tensor = torch.ones(shape_mixture_weights, dtype=dtype)
        # self.register_buffer("mixture_weights_tensor", mixture_weights_tensor)

        if init_positive:
            with torch.no_grad():
                self.dens_value_tensor.data.copy_(
                    self.dens_value_tensor.data + 0.2
                )
                # make density values positive while preserving gradient flow
                self.derivative_tensor.data.copy_(0.1 * self.derivative_tensor.data)

        self.conditional_spline_dist = conditional_spline_dist

    def reparam(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        param_mixture_weights_log = self.mixture_weights_tensor.log_softmax(dim=-1)
        param_dens_value = self.dens_value_tensor
        param_dens_derivative = self.derivative_tensor

        return param_dens_value, param_dens_derivative, param_mixture_weights_log

    def forward(self) -> spline.SplineSQ2D:
        return self.conditional_spline_dist(*self.reparam())
    
    def __call__(self) -> spline.SplineSQ2D:
        return super().__call__()  # preserves hooks


def create_network(
    input_size: int,
    conditional_spline_dist: spline.ConditionalSplineSQ2D,
    init_positive: bool,
    device: torch.device,
) -> UnconditionalNetwork:
    model = UnconditionalNetwork(
        conditional_spline_dist=conditional_spline_dist,
        init_positive=init_positive,
    )
    model.to(device)
    return model


def get_spline_model(
    sdd: csdd.SDDSingleImageTrajectory,
    num_knots: int,
    num_mixtures: int,
    init_positive: bool,
    device: torch.device,
) -> UnconditionalNetwork:
    """
    Constructs and returns a spline-based neural network model for trajectory prediction.

    Args:
        sdd (csdd.SDDSingleImageTrajectory): An instance of SDDSingleImageTrajectory
            containing trajectory data and constraints.
        num_knots (int): The number of knots for the spline distribution.
        num_mixtures (int): The number of mixture components for the spline distribution.
        net_size (str): The size of the neural network. Options are "small", "medium", or "large".
        init_last_layer_positive (bool): If True, initializes the last layer of the network to be positive.
        device (torch.device): The device on which the model and computations will be performed (e.g., CPU or GPU).

    Returns:
        MarginalNetwork: A neural network model
            configured to output spline-based trajectory distributions.
    """
    # create the constraints
    lra_problem = sdd.create_constraints()

    input_size = int(np.prod(sdd.get_x_shape()))

    spline_distribution_builder = spline.SplineSQ2DBuilder(
        constraints=lra_problem,
        var_positions=sdd.get_y_vars(),
        num_knots=num_knots,
        num_mixtures=num_mixtures,
    )

    spline_distribution_builder.to(device)

    gasp_kwargs = {
        "batch_size": 256,
    }

    # create the distribution
    start_time = time.time()
    conditional_spline_dist = integrate_distribution(
        d=spline_distribution_builder,
        device=device,
        precision=torch.float64,
        gasp_kwargs=gasp_kwargs,
    )
    end_time = time.time()
    print(f"Time to integrate distribution: {end_time - start_time:.2f} seconds")

    # create the model
    model = create_network(
        input_size=input_size,
        conditional_spline_dist=conditional_spline_dist,
        init_positive=init_positive,
        device=device,
    )
    return model


def main(args: argparse.Namespace) -> None:
    # check if random seed is set
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    else:
        args.seed = int(time.time())
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to {args.seed}")

    sdd = csdd.SDDSingleImageTrajectory(
        img_id=args.img_id,
        path="./data/sdd",
        unconditional=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_spline_model(
        sdd=sdd,
        num_knots=args.num_knots,
        num_mixtures=args.num_mixtures,
        init_positive=args.init_positive,
        device=device,
    )

    print("Loading dataset")
    dataset = sdd.load_dataset()
    dataset_train = dataset.train
    dataset_val = dataset.val
    dataset_test = dataset.test
    len_train = len(dataset_train)  # type: ignore
    len_val = len(dataset_val)  # type: ignore
    len_test = len(dataset_test)  # type: ignore
    print(
        f"Train size: {len_train}, Val size: {len_val}, Test size: {len_test}"
    )

    model.to(device)
    precision = torch.float64 if args.use_float64 else torch.float32
    model.to(precision)

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    dataset_train_x: torch.Tensor = dataset_train.tensors[0].to(device).to(precision)  # type: ignore
    dataset_val_x: torch.Tensor = dataset_val.tensors[0].to(device).to(precision)  # type: ignore
    dataset_test_x: torch.Tensor = dataset_test.tensors[0].to(device).to(precision)  # type: ignore

    epochs = args.epochs

    # Create a unique directory for this run
    run_id = f"run_{int(time.time())}"
    checkpoint_dir = os.path.join("./data/sdd_checkpoints_unconditional", run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    best_val_ll = float("-inf")

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()

        # forward pass
        log_dens = model().log_dens(dataset_train_x)

        loss = -log_dens.mean()
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            # validation
            model.eval()
            with torch.no_grad():

                log_dens_val = model().log_dens(dataset_val_x)
                val_ll = log_dens_val.to("cpu").mean()
                print(f"Epoch {epoch}: Validation log-likelihood: {val_ll:.4f}")

                # Save the best model
                if val_ll > best_val_ll:
                    best_val_ll = val_ll
                    torch.save(model.state_dict(), best_model_path)
                    print(
                        f"Best model saved with validation log-likelihood: {best_val_ll:.4f}"
                    )

    # Load the best model for testing
    model.load_state_dict(torch.load(best_model_path))
    print(f"Best model restored for testing from {best_model_path}.")

    # validation
    with torch.no_grad():
        log_dens_val = model().log_dens(dataset_val_x)
        val_ll = log_dens_val.to("cpu").mean()
        print(f"Validation log-likelihood: {val_ll:.4f} (reference: {best_val_ll:.4f})")

    # test
    with torch.no_grad():
        log_dens_test = model().log_dens(dataset_test_x)
        test_ll = log_dens_test.to("cpu").mean()
        print(f"Test log-likelihood: {test_ll:.4f}")


def args():
    parser = argparse.ArgumentParser(
        description="Train a MLP model using the SDD dataset"
    )
    parser.add_argument(
        "--img_id",
        type=int,
        default=12,
        help="Image ID to use for the SDD dataset",
    )
    parser.add_argument(
        "--num_knots",
        type=int,
        default=14,
        help="Number of knots to use for the spline distribution",
    )
    parser.add_argument(
        "--num_mixtures",
        type=int,
        default=10,
        help="Number of mixtures to use for the spline distribution",
    )
    parser.add_argument(
        "--init_positive",
        action="store_true",
        help="Initialize the last layer of the network to be positive",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1500,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--use_float64",
        action="store_true",
        help="Use float64 precision instead of float32",
    )
    return parser.parse_args()


if __name__ == "__main__":
    targs = args()
    main(targs)
