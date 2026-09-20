"""Python 3.14 version of the Case-1 transient 1D heat-conduction PINN.

Original source:
    Code.ipynb

The original notebook uses TensorFlow 1.15, TensorFlow's static-graph
interface, SciPyOptimizerInterface, and pyDOE.lhs.  This file keeps the
original calculation flow and rewrites only the execution layer for a
Python 3.14 environment:

    data generation
        -> PINN prediction
        -> PDE residual
        -> boundary and initial losses
        -> L-BFGS training
        -> post-processing and model saving

The governing equation is:

    T_t - T_xx - exp(x + 2t) = 0

with analytical solution:

    T(x, t) = exp(x + 2t)

The training also uses an implicit heat-diffusion preconditioner on a
uniform auxiliary grid:

    (I / dt - alpha * Dxx) * delta_T = -residual_T

The resulting temperature correction is used to guide the PINN update.
"""

from __future__ import annotations

import math as m
import pickle
import random
import shutil
import time
import timeit
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn


DTYPE = torch.float32


def lhs(n_dim: int, samples: int) -> np.ndarray:
    """Small local replacement for pyDOE.lhs."""
    points = np.empty((samples, n_dim), dtype=np.float64)
    cut = np.linspace(0.0, 1.0, samples + 1)

    for dim in range(n_dim):
        u = np.random.rand(samples)
        points[:, dim] = cut[:-1] + u * (cut[1:] - cut[:-1])
        np.random.shuffle(points[:, dim])

    return points


def get_default_device() -> torch.device:
    """Use GPU 2 when available, matching the original notebook setting."""
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 2:
            return torch.device("cuda:2")
        return torch.device("cuda:0")
    return torch.device("cpu")


def configure_plot_style() -> None:
    """Use LaTeX only when a LaTeX executable is available."""
    plt.rc("text", usetex=shutil.which("latex") is not None)
    plt.rc("font", family="serif")


class Pinn(nn.Module):

    def __init__(self, data, params, exist_model=False, file_dir="",
                 device=None):
        super().__init__()

        self.device = device if device is not None else get_default_device()

        # Initialize & unpack input data
        self.unpack(data, params)
        self.initialize_variables()

        # Initialize neural-network weights and biases
        if exist_model:
            print("Loading NN parameters ...")
            self.weights, self.biases = self.load_model(file_dir)
        else:
            self.weights, self.biases = self.initialize_network()

        self.to(self.device)
        self.initialize_physical_preconditioner()

        # These methods are retained as named stages from the original code.
        # PyTorch uses eager execution, so placeholders and static graph
        # construction are replaced by tensors created in fit_newton().
        self.initialize_placeholders()
        self.graph_network()
        self.graph_loss()
        self.initialize_optimizers()
        self.initialize_session()

    def callback(self, loss_test, loss_total, loss_collo, loss_bound,
                 loss_init, loss_preconditioned=0.0):
        self.count += 1
        self.loss_test_log.append(loss_test)
        self.loss_total_log.append(loss_total)
        self.loss_collo_log.append(loss_collo)
        self.loss_bound_log.append(loss_bound)
        self.loss_initial_log.append(loss_init)
        self.loss_preconditioned_log.append(loss_preconditioned)

        if self.count % self.verboses_newton == 0:
            print(
                "iter: %d, Loss Test: %.4e, Loss Total: %.4e, "
                "Loss Collo: %.4e, Loss Boundary: %.4e, "
                "Loss Initial: %.4e, Loss Preconditioned: %.4e"
                % (
                    self.count,
                    loss_test,
                    loss_total,
                    loss_collo,
                    loss_bound,
                    loss_init,
                    loss_preconditioned,
                )
            )

    def normalize_input_data(self):
        # Normalize data
        x_c = self.normalize_data(data=self.x_c, axis="x")
        t_c = self.normalize_data(data=self.t_c, axis="t")
        x_left = self.normalize_data(data=self.x_left, axis="x")
        t_left = self.normalize_data(data=self.t_left, axis="t")
        x_right = self.normalize_data(data=self.x_right, axis="x")
        t_right = self.normalize_data(data=self.t_right, axis="t")
        x_initial = self.normalize_data(data=self.x_initial, axis="x")
        t_initial = self.normalize_data(data=self.t_initial, axis="t")
        x_test = self.normalize_data(data=self.x_test, axis="x")
        t_test = self.normalize_data(data=self.t_test, axis="t")

        return (
            x_c,
            t_c,
            x_left,
            t_left,
            x_right,
            t_right,
            x_initial,
            t_initial,
            x_test,
            t_test,
        )

    def _tensor(self, array, requires_grad=False):
        return torch.as_tensor(
            np.asarray(array),
            dtype=DTYPE,
            device=self.device,
        ).clone().detach().requires_grad_(requires_grad)

    def initialize_physical_preconditioner(self):
        """Build the implicit heat-diffusion operator on an auxiliary grid."""
        self.preconditioner_x = None
        self.preconditioner_t = None
        self.preconditioner_matrix = None

        if not self.preconditioner_enabled:
            return

        x_phys = np.linspace(
            self.lb[0],
            self.ub[0],
            num=self.preconditioner_n_x,
        )
        t_phys = np.linspace(
            self.lb[1],
            self.ub[1],
            num=self.preconditioner_n_t,
        )
        X_phys, T_phys = np.meshgrid(x_phys, t_phys)

        X_norm = self.normalize_data(X_phys.reshape(-1, 1), axis="x")
        T_norm = self.normalize_data(T_phys.reshape(-1, 1), axis="t")

        # Coordinates need gradients for the PDE residual.
        self.preconditioner_x = self._tensor(
            X_norm,
            requires_grad=True,
        )
        self.preconditioner_t = self._tensor(
            T_norm,
            requires_grad=True,
        )

        dx = float(x_phys[1] - x_phys[0])
        if self.preconditioner_dt is None:
            dt = float(t_phys[1] - t_phys[0])
        else:
            dt = float(self.preconditioner_dt)

        n_inner = self.preconditioner_n_x - 2
        identity = torch.eye(
            n_inner,
            dtype=DTYPE,
            device=self.device,
        )
        diffusion = torch.zeros(
            (n_inner, n_inner),
            dtype=DTYPE,
            device=self.device,
        )

        diagonal = -2.0 / (dx**2)
        off_diagonal = 1.0 / (dx**2)
        row = torch.arange(n_inner, device=self.device)
        diffusion[row, row] = diagonal
        if n_inner > 1:
            diffusion[row[:-1], row[1:]] = off_diagonal
            diffusion[row[1:], row[:-1]] = off_diagonal

        # M_T = I / dt - alpha * Dxx.
        # Dirichlet boundary corrections are excluded by solving only for
        # interior nodes.
        self.preconditioner_matrix = (
            identity / dt
            - self.preconditioner_diffusivity * diffusion
        )

    def preconditioned_physics_loss(self):
        """Calculate the heat-operator-preconditioned residual loss.

        The physical correction is detached before constructing the target.
        Thus, the backward pass follows the preconditioned correction direction
        without introducing a third-derivative residual gradient.
        """
        if not self.preconditioner_enabled:
            return next(self.parameters()).new_zeros(())

        temperature = self.net_dnn(
            self.preconditioner_x,
            self.preconditioner_t,
        )
        residual = self.physics_residual(
            temperature,
            self.preconditioner_x,
            self.preconditioner_t,
        )

        residual_grid = residual.reshape(
            self.preconditioner_n_t,
            self.preconditioner_n_x,
        )
        residual_inner = residual_grid[:, 1:-1].transpose(0, 1)

        # Solve M_T * delta_T = -residual_T at all auxiliary-grid time levels.
        delta_inner = torch.linalg.solve(
            self.preconditioner_matrix,
            -residual_inner,
        )

        temperature_grid = temperature.reshape(
            self.preconditioner_n_t,
            self.preconditioner_n_x,
        )
        target_grid = temperature_grid.detach().clone()
        target_grid[:, 1:-1] = (
            target_grid[:, 1:-1]
            + delta_inner.detach().transpose(0, 1)
        )

        # The correction is defined on the interior nodes.
        return torch.mean(
            torch.square(
                temperature_grid[:, 1:-1]
                - target_grid[:, 1:-1]
            )
        )

    def fit_newton(self, max_iter=100000):
        """Train with the PyTorch full-batch L-BFGS equivalent."""
        self.newton_started = True

        (
            x_c,
            t_c,
            x_left,
            t_left,
            x_right,
            t_right,
            x_initial,
            t_initial,
            x_test,
            t_test,
        ) = self.normalize_input_data()

        # Collocation coordinates must track gradients for the PDE residual.
        tensors = {
            "x_c": self._tensor(x_c, requires_grad=True),
            "t_c": self._tensor(t_c, requires_grad=True),
            "x_left": self._tensor(x_left),
            "t_left": self._tensor(t_left),
            "u_left": self._tensor(self.u_left),
            "x_right": self._tensor(x_right),
            "t_right": self._tensor(t_right),
            "u_right": self._tensor(self.u_right),
            "x_initial": self._tensor(x_initial),
            "t_initial": self._tensor(t_initial),
            "u_initial": self._tensor(self.u_initial),
            "x_test": self._tensor(x_test),
            "t_test": self._tensor(t_test),
            "u_test": self._tensor(self.u_test),
        }

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=1.0,
            max_iter=max_iter,
            max_eval=max_iter,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            losses = self.compute_losses(tensors)
            losses["total"].backward()

            self.callback(
                loss_test=float(losses["test"].detach().cpu()),
                loss_total=float(losses["total"].detach().cpu()),
                loss_collo=float(losses["collo"].detach().cpu()),
                loss_bound=float(losses["bound"].detach().cpu()),
                loss_init=float(losses["initial"].detach().cpu()),
                loss_preconditioned=float(
                    losses["preconditioned"].detach().cpu()
                ),
            )
            return losses["total"]

        optimizer.step(closure)

    def compute_losses(self, tensors):
        # Test
        u_test_pred = self.net_dnn(tensors["x_test"], tensors["t_test"])
        loss_test = torch.sqrt(
            torch.mean(torch.square(tensors["u_test"] - u_test_pred))
        )

        # Collocation points
        f_pred_u = self.net_physics(tensors["x_c"], tensors["t_c"])
        loss_collo = torch.mean(torch.square(f_pred_u))

        # Boundary
        u_left_pred = self.net_dnn(tensors["x_left"], tensors["t_left"])
        u_right_pred = self.net_dnn(tensors["x_right"], tensors["t_right"])
        u_initial_pred = self.net_dnn(
            tensors["x_initial"],
            tensors["t_initial"],
        )

        loss_left = torch.mean(
            torch.square(u_left_pred - tensors["u_left"])
        )
        loss_right = torch.mean(
            torch.square(u_right_pred - tensors["u_right"])
        )
        loss_initial = torch.mean(
            torch.square(u_initial_pred - tensors["u_initial"])
        )

        loss_bound = loss_left + loss_right
        loss_preconditioned = self.preconditioned_physics_loss()
        loss_total = (
            loss_collo
            + loss_bound
            + loss_initial
            + self.preconditioner_weight * loss_preconditioned
        )

        return {
            "test": loss_test,
            "collo": loss_collo,
            "left": loss_left,
            "right": loss_right,
            "initial": loss_initial,
            "bound": loss_bound,
            "preconditioned": loss_preconditioned,
            "total": loss_total,
        }

    def graph_loss(self):
        # Static-graph loss construction is represented by compute_losses()
        # in the eager PyTorch implementation.
        return None

    def graph_network(self):
        # Static-graph network construction is represented by net_dnn().
        return None

    def load_model(self, file_dir):
        weights = nn.ParameterList()
        biases = nn.ParameterList()
        num_layers = len(self.layers)

        with open(file_dir, "rb") as f:
            dnn_weights, dnn_biases = pickle.load(f)

        # Stored model must have the same layers
        assert num_layers == (len(dnn_weights) + 1)

        for num in range(0, num_layers - 1):
            weight = nn.Parameter(
                torch.as_tensor(dnn_weights[num], dtype=DTYPE)
            )
            bias = nn.Parameter(
                torch.as_tensor(dnn_biases[num], dtype=DTYPE)
            )
            weights.append(weight)
            biases.append(bias)

        print("Loaded NN parameters successfully ...")
        return weights, biases

    def initialize_network(self):
        weights = nn.ParameterList()
        biases = nn.ParameterList()
        num_layers = len(self.layers)

        # Create network
        for lyr in range(num_layers - 1):
            # Initialize weights from Xavier initialization
            np.random.seed(self.random_seed)
            weight = self.xavier_init(
                size=[self.layers[lyr], self.layers[lyr + 1]]
            )

            # Initialize biases = 0
            np.random.seed(self.random_seed)
            bias = nn.Parameter(
                torch.zeros(
                    (1, self.layers[lyr + 1]),
                    dtype=DTYPE,
                )
            )

            weights.append(weight)
            biases.append(bias)

        return weights, biases

    def initialize_optimizers(self):
        # The optimizer is constructed in fit_newton after all input tensors
        # have been prepared.
        return None

    def initialize_placeholders(self):
        # TensorFlow placeholders are not needed in eager PyTorch execution.
        return None

    def initialize_session(self):
        # TensorFlow session initialization is not needed in PyTorch.
        return None

    def initialize_variables(self):
        # For saving loss
        self.loss_total_log = []
        self.loss_collo_log = []
        self.loss_bound_log = []
        self.loss_initial_log = []
        self.loss_test_log = []
        self.loss_preconditioned_log = []
        self.count = 0
        self.newton_started = False

    def net_dnn(self, x, t):
        # Find results
        X = torch.cat([x, t], dim=1)
        results = self.net_forward(X)
        return results

    def net_forward(self, X):
        num_layers = len(self.weights) + 1
        H = X

        for lyr in range(num_layers - 2):
            weight = self.weights[lyr]
            bias = self.biases[lyr]
            H = torch.tanh(torch.add(torch.matmul(H, weight), bias))

        weight = self.weights[-1]
        bias = self.biases[-1]
        Y = torch.add(torch.matmul(H, weight), bias)

        return Y

    def net_physics(self, x, t):
        # Find results from DNN
        T = self.net_dnn(x, t)
        return self.physics_residual(T, x, t)

    def physics_residual(self, T, x, t):
        """Calculate the governing-equation residual."""

        # Temperature gradient
        T_x = torch.autograd.grad(
            T.sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0] / self.sigma_x
        T_xx = torch.autograd.grad(
            T_x.sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0] / self.sigma_x
        T_t = torch.autograd.grad(
            T.sum(),
            t,
            create_graph=True,
            retain_graph=True,
        )[0] / self.sigma_t

        # Physics error
        x_ = x * self.sigma_x + self.mu_x
        t_ = t * self.sigma_t + self.mu_t
        f = T_t - T_xx - torch.exp(x_ + 2.0 * t_)

        return f

    def normalize_data(self, data, axis):
        if axis == "x":
            normalized_data = (data - self.mu_x) / self.sigma_x
        elif axis == "t":
            normalized_data = (data - self.mu_t) / self.sigma_t
        else:
            raise ValueError("axis must be 'x' or 't'")

        return normalized_data

    @torch.no_grad()
    def predict(self, x_star, t_star):
        # Prepare the input
        x_star = self.normalize_data(x_star, axis="x")
        t_star = self.normalize_data(t_star, axis="t")

        x_tensor = self._tensor(x_star)
        t_tensor = self._tensor(t_star)

        was_training = self.training
        self.eval()
        u_star = self.net_dnn(x_tensor, t_tensor)
        if was_training:
            self.train()

        return u_star.detach().cpu().numpy()

    def save_loss(self, file_dir):
        loss_test = np.array(self.loss_test_log)
        loss_data = np.column_stack(
            (
                self.loss_total_log,
                self.loss_collo_log,
                self.loss_bound_log,
                self.loss_preconditioned_log,
                loss_test,
            )
        )

        try:
            import joblib
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "save_loss requires pandas and joblib. "
                "Install them in the Python 3.14 environment first."
            ) from exc

        loss_df = pd.DataFrame(
            loss_data,
            columns=[
                "total",
                "collo",
                "boundary",
                "preconditioned",
                "error_u",
            ],
        )
        joblib.dump(loss_df, file_dir)

    def save_model(self, file_dir):
        weights = [
            weight.detach().cpu().numpy()
            for weight in self.weights
        ]
        biases = [
            bias.detach().cpu().numpy()
            for bias in self.biases
        ]

        with open(file_dir, "wb") as f:
            pickle.dump([weights, biases], f)
            print("Save NN parameters successfully...")

    def unpack(self, data, params):
        # Initialize
        self.data = data
        self.params = params

        # Unpack parameters
        self.lb = params["data"]["lb"]
        self.ub = params["data"]["ub"]

        self.random_seed = data["train"]["random_seed"]

        # Data-collocation
        self.x_c = data["train"]["collo"][:, 0:1]
        self.t_c = data["train"]["collo"][:, 1:2]
        self.mu_x = data["train"]["mu_x"]
        self.mu_t = data["train"]["mu_t"]
        self.sigma_x = data["train"]["sigma_x"]
        self.sigma_t = data["train"]["sigma_t"]

        # Data-left
        self.x_left = data["train"]["left"][:, 0:1]
        self.t_left = data["train"]["left"][:, 1:2]
        self.u_left = data["train"]["left"][:, 2:3]

        # Data-right
        self.x_right = data["train"]["right"][:, 0:1]
        self.t_right = data["train"]["right"][:, 1:2]
        self.u_right = data["train"]["right"][:, 2:3]

        # Data-initial
        self.x_initial = data["train"]["initial"][:, 0:1]
        self.t_initial = data["train"]["initial"][:, 1:2]
        self.u_initial = data["train"]["initial"][:, 2:3]

        # Data-test
        self.x_test = data["test"][:, 0:1]
        self.t_test = data["test"][:, 1:2]
        self.u_test = data["test"][:, 2:3]

        # Network
        self.layers = params["network"]["layers"]
        self.verboses_newton = params["network"]["verboses_newton"]
        self.saver = params["network"]["saver"]

        # Physical preconditioner
        physic_params = params.get("physic", {})
        preconditioner_params = physic_params.get("preconditioner", {})
        self.preconditioner_enabled = preconditioner_params.get(
            "enabled",
            True,
        )
        self.preconditioner_n_x = int(
            preconditioner_params.get("n_x", 64)
        )
        self.preconditioner_n_t = int(
            preconditioner_params.get("n_t", 16)
        )
        self.preconditioner_dt = preconditioner_params.get(
            "dt",
            None,
        )
        self.preconditioner_diffusivity = float(
            preconditioner_params.get("diffusivity", 1.0)
        )
        self.preconditioner_weight = float(
            preconditioner_params.get("weight", 0.1)
        )

        if self.preconditioner_n_x < 3:
            raise ValueError("preconditioner n_x must be at least 3")
        if self.preconditioner_n_t < 2:
            raise ValueError("preconditioner n_t must be at least 2")
        if self.preconditioner_diffusivity <= 0.0:
            raise ValueError("preconditioner diffusivity must be positive")

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))

        torch.manual_seed(self.random_seed)
        weight = torch.empty(
            (in_dim, out_dim),
            dtype=DTYPE,
        )
        nn.init.trunc_normal_(
            weight,
            mean=0.0,
            std=xavier_stddev,
        )
        return nn.Parameter(weight)


class Conduction:

    def __init__(self, add_noise=False, save_fig=False):
        self.add_noise = add_noise
        self.save_fig = save_fig
        self.data = {}

    def analytics_solution(self, x, t):
        u_ = np.exp(x + 2 * t)
        return u_

    def generate_bc(self, loc):
        lb = self.lb
        ub = self.ub

        if loc == "left":
            n = self.n_left
            x = lb[0] * np.ones((n, 1))
            np.random.seed(self.random_seed)
            t = lb[1] + (ub[1] - lb[1]) * lhs(1, n)
            temperature = np.exp(2 * t)
            bc_datas = np.concatenate((x, t, temperature), axis=1)

        elif loc == "right":
            n = self.n_right
            x = self.ub[0] * np.ones((n, 1))
            np.random.seed(self.random_seed)
            t = lb[1] + (ub[1] - lb[1]) * lhs(1, n)
            temperature = np.exp(1 + 2 * t)
            bc_datas = np.concatenate((x, t, temperature), axis=1)

        else:
            n = self.n_initial
            np.random.seed(self.random_seed)
            x = lb[0] + (ub[0] - lb[0]) * lhs(1, n)
            t = lb[1] * lhs(1, n)
            temperature = np.exp(x)
            bc_datas = np.concatenate((x, t, temperature), axis=1)

        return bc_datas

    def generate_collo(self):
        # Unpack
        n = self.n_collo
        lb = self.lb
        ub = self.ub

        # Create points
        np.random.seed(self.random_seed)
        collo_pts = lb + (ub - lb) * lhs(2, n)

        self.mu_X = collo_pts.mean(0)
        self.sigma_X = collo_pts.std(0)
        self.mu_x = self.mu_X[0]
        self.sigma_x = self.sigma_X[0]
        self.mu_t = self.mu_X[1]
        self.sigma_t = self.sigma_X[1]

        return collo_pts

    def generate_train_data(self, param):
        # Unpack parameters
        self.unpack(param)

        # Generate boundary-condition data
        left_ = self.generate_bc("left")
        right_ = self.generate_bc("right")
        initial_ = self.generate_bc("initial")

        # Generate collocation-point data
        collo_ = self.generate_collo()

        # Pack data
        self.data["train"] = {}
        self.data["train"]["collo"] = collo_
        self.data["train"]["left"] = left_
        self.data["train"]["right"] = right_
        self.data["train"]["initial"] = initial_
        self.data["train"]["mu_x"] = self.mu_x
        self.data["train"]["sigma_x"] = self.sigma_x
        self.data["train"]["mu_t"] = self.mu_t
        self.data["train"]["sigma_t"] = self.sigma_t
        self.data["train"]["random_seed"] = self.random_seed

    def generate_test_data(self, param):
        # Unpack parameters
        self.unpack(param)

        x_ = np.linspace(self.lb[0], self.ub[0], num=self.n_test)
        t_ = np.linspace(self.lb[1], self.ub[1], num=self.n_test)
        X, T = np.meshgrid(x_, t_)
        X_flat = X.flatten()[:, None]
        T_flat = T.flatten()[:, None]
        u_ = self.analytics_solution(X_flat, T_flat)

        self.data["test"] = np.column_stack((X_flat, T_flat, u_))

    def plot(self):
        # Unpack data
        collo_ = self.data["train"]["collo"]
        left_ = self.data["train"]["left"]
        right_ = self.data["train"]["right"]
        initial_ = self.data["train"]["initial"]

        configure_plot_style()

        # Plot
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(5, 5),
            constrained_layout=True,
            dpi=300,
        )

        ax.plot(
            [self.lb[0], self.ub[0]],
            [self.lb[1], self.lb[1]],
            "k",
        )
        ax.plot(
            [self.lb[0], self.ub[0]],
            [self.ub[1], self.ub[1]],
            "k",
        )
        ax.plot(
            [self.lb[0], self.lb[0]],
            [self.ub[1], self.lb[1]],
            "k",
        )
        ax.plot(
            [self.ub[0], self.ub[0]],
            [self.ub[1], self.lb[1]],
            "k",
        )

        ax.scatter(
            collo_[:, 0:1],
            collo_[:, 1:2],
            marker=".",
            alpha=0.7,
            c="grey",
            label="Collo",
        )
        ax.scatter(
            left_[:, 0:1],
            left_[:, 1:2],
            marker=".",
            alpha=0.7,
            c="r",
            label="Left BC",
        )
        ax.scatter(
            right_[:, 0:1],
            right_[:, 1:2],
            marker=".",
            alpha=0.7,
            c="g",
            label="Right BC",
        )
        ax.scatter(
            initial_[:, 0:1],
            initial_[:, 1:2],
            marker=".",
            alpha=0.7,
            c="b",
            label="IC",
        )

        ax.set_title("Points distribution", fontsize=18)
        ax.set_xlabel("$x$ (m)", fontsize=20)
        ax.set_ylabel("$t$ (s)", fontsize=20)
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.legend(fontsize=10, loc=4)
        ax.grid(linestyle="--")
        ax.set_xlim(self.lb[0], self.ub[0])
        ax.set_ylim(self.lb[1], self.ub[1])

        if self.save_fig:
            fig.savefig("fig_point_distribution.eps", format="eps")
        plt.show()

        if self.save_fig:
            fig.savefig("fig_references.eps", format="eps")
        plt.show()

    def unpack(self, param):
        # Bound
        self.lb = param["data"]["lb"]
        self.ub = param["data"]["ub"]

        # Discretizations
        self.n_collo = param["data"]["n_collo"]
        self.n_left = param["data"]["n_left"]
        self.n_right = param["data"]["n_right"]
        self.n_initial = param["data"]["n_initial"]
        self.n_test = param["data"]["n_test"]
        self.random_seed = param["data"]["seed"]


class PostProcessing:

    def __init__(self, model, params, save_fig=False):
        self.save_fig = save_fig
        self.unpack(model, params)

    def calculate_pinn(self):
        self.u_pinn = self.model.predict(self.X_flat, self.T_flat)
        self.x_pinn_ = self.X_flat.reshape(self.n_t, self.n_x)
        self.t_pinn_ = self.T_flat.reshape(self.n_t, self.n_x)
        self.u_pinn_ = self.u_pinn.reshape(self.n_t, self.n_x)

    def calculate_error(self, n_data):
        x_test = self.model.x_test
        t_test = self.model.t_test

        # Predict
        u_pinn = self.model.predict(x_test, t_test)
        u_test = self.model.u_test
        u_analytic = self.model.u_test
        delta_u = np.abs(self.u_pinn - u_analytic)
        self.abs_err_u = np.sum(delta_u) / (self.n_x * self.n_t)
        rel_err_u_ij = delta_u / u_analytic
        self.rel_err_u = np.sum(rel_err_u_ij) / (self.n_x * self.n_t)

        delta_squared = delta_u**2
        self.rmse_u = np.sqrt(
            np.sum(delta_squared) / (self.n_x * self.n_t)
        )

        self.mean_u_test = np.mean(u_test)
        self.SST = np.sum((u_test - self.mean_u_test) ** 2)
        self.SSE = np.sum(delta_squared)
        self.R2 = 1 - self.SSE / self.SST

        print(f"- Absolute Error: {self.abs_err_u:5f}")
        print(f"- Relative Error (%): {self.rel_err_u * 100:5f}")
        print(f"- RMSE: {self.rmse_u:5f}")
        print(f"- R2: {self.R2:5f}")

        return x_test, t_test, u_pinn, u_test

    def create_test_data(self):
        # Find fraction
        length_ = self.ub - self.lb
        min_length_ = np.argmin(length_)
        frac_ = max(length_) / min(length_)
        n_test_1 = int(frac_ * self.n_test)

        # Create grid
        if min_length_ == 0:
            self.n_x = self.n_test
            self.n_t = n_test_1
        else:
            self.n_x = n_test_1
            self.n_t = self.n_test

        self.x = np.linspace(
            self.lb[0],
            self.ub[0],
            num=self.n_x,
        )
        self.t = np.linspace(
            self.lb[1],
            self.ub[1],
            num=self.n_t,
        )
        self.X, self.T = np.meshgrid(self.x, self.t)
        self.X_flat = self.X.flatten()[:, None]
        self.T_flat = self.T.flatten()[:, None]

    def display_loss(self):
        # Set
        configure_plot_style()

        # Extract data
        loss_total = np.sqrt(self.model.loss_total_log)
        loss_collo = np.sqrt(self.model.loss_collo_log)
        loss_initial = np.sqrt(self.model.loss_initial_log)
        loss_bound = np.sqrt(self.model.loss_bound_log)
        loss_preconditioned = np.sqrt(
            self.model.loss_preconditioned_log
        )
        loss_test = self.model.loss_test_log

        # Status
        if self.model.newton_started:
            run_status = "newton"
        else:
            run_status = "none"

        # Prepare
        iter_total = []

        if run_status == "newton":
            n_total = len(loss_total)
            for i in range(n_total):
                iter_total.append(i)
        else:
            n_total = 0

        # Plot history
        if run_status != "none":
            fig, ax = plt.subplots(
                nrows=1,
                ncols=1,
                figsize=(8, 6),
                constrained_layout=True,
                dpi=300,
            )
            ax.plot(
                iter_total,
                loss_total,
                "r",
                linestyle="solid",
                label="Physical Loss",
            )
            ax.plot(
                iter_total,
                loss_collo,
                "g",
                linestyle="dashdot",
                label="Collocation Loss",
            )
            ax.plot(
                iter_total,
                loss_bound,
                "b",
                linestyle="dotted",
                label="Boundary Loss",
            )
            ax.plot(
                iter_total,
                loss_initial,
                "purple",
                linestyle="solid",
                label="Initial Loss",
            )
            ax.plot(
                iter_total,
                loss_preconditioned,
                "orange",
                linestyle="dashed",
                label="Preconditioned Physics Loss",
            )
            ax.plot(
                iter_total,
                loss_test,
                "black",
                linestyle="dashed",
                label="Test Error",
            )

            ax.set_xlim(0, iter_total[-1])
            ax.set_yscale("log")
            ax.set_xlabel("Iterations", fontsize=25)
            ax.set_ylabel("RMSE", fontsize=25)
            ax.grid(linestyle="--")
            ax.legend(fontsize=13)
            ax.set_title("Loss History (Log Scale)", fontsize=35)
            ax.tick_params(axis="both", which="major", labelsize=11)

            if self.save_fig:
                fig.savefig("fig_loss_history.eps", format="eps")
            plt.show()

        print(f"- Last Iterations: {iter_total[-1]}")

    def display_contour(self):
        # Find PINN results
        self.create_test_data()
        self.calculate_pinn()

        # Calculate error
        x_, t_, u_pinn, u_test = self.calculate_error(
            n_data=self.n_test
        )

        # Plot comparison
        fig_h = 3
        fig_w = 10

        self.plot_comparison(
            hor_val=self.t,
            hor_type="t",
            phi_pinn=u_pinn.reshape(self.n_t, self.n_x),
            phi_analytics=u_test.reshape(self.n_t, self.n_x),
            fig_w=fig_w,
            fig_h=fig_h * 2,
        )
        self.plot_comparison(
            hor_val=self.x,
            hor_type="x",
            phi_pinn=u_pinn.reshape(self.n_t, self.n_x),
            phi_analytics=u_test.reshape(self.n_t, self.n_x),
            fig_w=fig_w,
            fig_h=fig_h * 2,
        )

        vmin = np.min(u_pinn)
        vmax = np.max(u_pinn)
        self.plot_countour(
            x=self.X,
            t=self.T,
            x_flat=self.X_flat,
            t_flat=self.T_flat,
            phi=self.u_pinn,
            phi_=self.u_pinn_,
            vmin=vmin,
            vmax=vmax,
            types="T",
            fig_w=fig_w,
            fig_h=fig_h,
        )

    def plot_comparison(
        self,
        hor_val,
        hor_type,
        phi_pinn,
        phi_analytics,
        fig_w,
        fig_h,
    ):
        fig, ax = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(fig_w, fig_h),
            dpi=100,
            constrained_layout=False,
        )

        if hor_type == "x":
            n = self.n_x
            ver_type = "t"
            hor_unit = "m"
            ver_unit = "s"

            ax[0].plot(
                hor_val,
                phi_pinn[n // 4, :].flatten(),
                "b",
                label="PINN solution",
                linewidth=4,
            )
            ax[0].plot(
                hor_val,
                phi_analytics[n // 4, :],
                "--r",
                label="Analytical solution",
                linewidth=4,
            )
            ax[1].plot(
                hor_val,
                phi_pinn[n // 2, :].flatten(),
                "b",
                label="PINN solution",
                linewidth=4,
            )
            ax[1].plot(
                hor_val,
                phi_analytics[n // 2, :],
                "--r",
                label="Analytical solution",
                linewidth=4,
            )
            ax[2].plot(
                hor_val,
                phi_pinn[n // 4 * 3 + 1, :].flatten(),
                "b",
                label="PINN solution",
                linewidth=4,
            )
            ax[2].plot(
                hor_val,
                phi_analytics[n // 4 * 3 + 1, :],
                "--r",
                label="Analytical solution",
                linewidth=4,
            )

        elif hor_type == "t":
            n = self.n_t
            ver_type = "x"
            hor_unit = "s"
            ver_unit = "m"

            ax[0].plot(
                hor_val,
                phi_pinn[:, n // 4].flatten(),
                "b",
                label="PINN solution",
                linewidth=4,
            )
            ax[0].plot(
                hor_val,
                phi_analytics[:, n // 4],
                "--r",
                label="Analytical solution",
                linewidth=4,
            )
            ax[1].plot(
                hor_val,
                phi_pinn[:, n // 2].flatten(),
                "b",
                label="PINN solution",
                linewidth=4,
            )
            ax[1].plot(
                hor_val,
                phi_analytics[:, n // 2],
                "--r",
                label="Analytical solution",
                linewidth=4,
            )
            ax[2].plot(
                hor_val,
                phi_pinn[:, n // 4 * 3 + 1].flatten(),
                "b",
                label="PINN solution",
                linewidth=4,
            )
            ax[2].plot(
                hor_val,
                phi_analytics[:, n // 4 * 3 + 1],
                "--r",
                label="Analytical solution",
                linewidth=4,
            )

        x_lim_min = min(
            np.min(phi_pinn),
            np.min(phi_analytics),
        )
        x_lim_max = max(
            np.max(phi_pinn),
            np.max(phi_analytics),
        )
        x_lim_min = 0
        x_lim_max = x_lim_max + 0.2 * abs(x_lim_max)

        for axis, position in zip(ax, ("0.25", "0.5", "0.75")):
            axis.set_ylim(x_lim_min, x_lim_max)
            axis.set_xlim(self.lb[1], self.ub[1])
            axis.set_xlabel(
                f"${hor_type}$ ({hor_unit}), "
                f"${ver_type}={position}$ {ver_unit}",
                fontsize=25,
            )
            axis.set_ylabel("$T$ (\u2103)", fontsize=25)
            axis.grid(linestyle="--")
            axis.legend(fontsize=13)
            axis.tick_params(
                axis="both",
                which="major",
                labelsize=23,
            )

        ax[1].set_title(
            "Temperature comparison",
            fontsize=45,
        )
        plt.tight_layout()

    def plot_countour(
        self,
        x,
        t,
        x_flat,
        t_flat,
        phi,
        phi_,
        vmin,
        vmax,
        types,
        fig_w,
        fig_h,
    ):
        # Create PINN-solution plot
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(fig_w, fig_h),
            dpi=300,
            constrained_layout=False,
        )

        cf = ax.scatter(
            x_flat,
            t_flat,
            c=phi,
            alpha=1.0,
            edgecolors="none",
            cmap="jet",
            marker=".",
            s=50,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlim(self.lb[0], self.ub[0])
        ax.set_ylim(self.lb[1], self.ub[1])
        ax.set_title(
            f"${types}$ $(x,t)$ PINN Solution",
            fontsize=30,
        )
        ax.set_xlabel("$x$ (m)", fontsize=25)
        ax.set_ylabel("$t$ (s)", fontsize=25)
        ax.tick_params(axis="both", which="major", labelsize=17)
        ax.contour(
            x,
            t,
            phi_,
            colors="k",
            linewidths=0.2,
            levels=50,
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.1)
        cb = fig.colorbar(cf, cax=cax)
        ticks = np.linspace(vmin, vmax, 3)
        ticks[1] = np.round(ticks[1], 2)
        ticks[-1] = m.floor(ticks[-1] * 100) / 100.0
        ticks[0] = m.ceil(ticks[0] * 100) / 100.0
        cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=17)

        # Analytical-solution contour
        u = np.exp(x_flat + 2 * t_flat)
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(fig_w, fig_h),
            dpi=300,
            constrained_layout=False,
        )
        cf = ax.scatter(
            x_flat,
            t_flat,
            c=u,
            alpha=1.0,
            edgecolors="none",
            cmap="jet",
            marker=".",
            s=50,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlim(self.lb[0], self.ub[0])
        ax.set_ylim(self.lb[1], self.ub[1])
        ax.set_title(
            f"${types}$ $(x,t)$ Analytical Solution",
            fontsize=30,
        )
        ax.set_xlabel("$x$ (m)", fontsize=25)
        ax.set_ylabel("$t$ (s)", fontsize=25)
        ax.tick_params(axis="both", which="major", labelsize=17)
        ax.contour(
            x,
            t,
            np.exp(x + 2 * t),
            colors="k",
            linewidths=0.2,
            levels=50,
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.1)
        cb = fig.colorbar(cf, cax=cax)
        ticks = np.linspace(vmin, vmax, 3)
        ticks[1] = np.round(ticks[1], 2)
        ticks[-1] = m.floor(ticks[-1] * 100) / 100.0
        ticks[0] = m.ceil(ticks[0] * 100) / 100.0
        cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=17)

        plt.show()

        if self.save_fig:
            fig.savefig(f"fig_{types}.eps", format="eps")
        plt.show()

    def unpack(self, model, params):
        self.model = model
        self.params = params

        self.lb = params["data"]["lb"]
        self.ub = params["data"]["ub"]
        self.n_test = params["data"]["n_test"]
        self.verboses_newton = params["network"]["verboses_newton"]


def generate_param():
    params = {}

    params["data"] = {}
    params["data"]["lb"] = np.array([0.0, 0.0])
    params["data"]["ub"] = np.array([1.0, 1.0])
    params["data"]["n_collo"] = 1000
    params["data"]["n_left"] = 101
    params["data"]["n_right"] = 101
    params["data"]["n_initial"] = 101
    params["data"]["n_test"] = 201
    seed = np.random.randint(1, 1000)
    params["data"]["seed"] = seed
    print(f"seed: {seed}")

    params["physic"] = {}
    params["physic"]["preconditioner"] = {
        "enabled": True,
        "n_x": 64,
        "n_t": 16,
        "dt": None,
        "diffusivity": 1.0,
        "weight": 0.1,
    }

    params["network"] = {}
    params["network"]["verboses_newton"] = 1000
    params["network"]["saver"] = 5000

    return params


def main():
    device = get_default_device()
    print(f"device: {device}")

    params = generate_param()
    params["network"]["layers"] = [2] + 1 * [5] + [1]

    case = Conduction()
    case.generate_train_data(param=params)
    case.generate_test_data(param=params)
    case.plot()

    # Create model instance
    model = Pinn(
        data=case.data,
        params=params,
        device=device,
    )

    # Fit using L-BFGS
    model.fit_newton()

    # Create results
    results = PostProcessing(
        model=model,
        params=params,
        save_fig=False,
    )

    results.display_loss()
    results.display_contour()

    # Save model
    model.save_model("Dirichlet_good.pickle")


if __name__ == "__main__":
    main()
