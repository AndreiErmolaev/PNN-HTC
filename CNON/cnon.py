"""
Coupled nonlinear oscillator network (CNON) used as a physical neural network.

This is a minimal port of the model classes `SparseLinear`, `Oscillator_RNN_dyn`
and `Oscillator_RNN_parallel` written by Anas Skalli
(https://github.com/ASkalli/nonlinear_oscillator_network, Utils/NN_utils.py).
The weight initialisation and the numerical integration are identical to the
original code that produced the CNON data of the paper; only the bookkeeping
was simplified (no stored trajectories, one class for both use cases).

Each hidden layer l = 1, ..., L evolves according to (Eq. 17 of the paper)

    dh^[l]/dt = -alpha h^[l] + sin(W_rec^[l] h^[l] + W_in^[l] h^[l-1] + b^[l]),

with h^[0] the input image, and the output is y = W_out h^[L] + b_out (Eq. 18).
The input is clamped while the network is integrated with explicit Euler steps
starting from h = 0, either until it reaches a steady state (ELM / reservoir
use, `adaptive=True`) or for a fixed number of steps (backpropagation training,
`adaptive=False`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def spectral_radius(W):
    return torch.linalg.eigvals(W).abs().max()


class SparseSymmetricLinear(nn.Module):
    """Recurrent coupling W_rec with a symmetric sparsity pattern.

    Port of `SparseLinear(sym=True)`. As in the original code:
      * the mask is symmetric but the weights are not (W_ij and W_ji are drawn
        independently from U(-1, 1));
      * `density` is the fraction of selected upper-triangular entries (diagonal
        included) and the identity is added to the mask, so diagonal entries
        that were also selected carry a mask value of 2;
      * the masked weights are scaled to `rho`, and the mask is applied again
        in the forward pass and to the gradients.
    """

    def __init__(self, n, density=0.1, rho=0.8):
        super().__init__()
        self.register_buffer("mask", self.symmetric_mask(n, density))
        W = (2 * torch.rand(n, n) - 1) * self.mask
        rho_W = spectral_radius(W)
        rho_W = rho_W if rho_W > 1e-6 else 1e-6
        self.weight = nn.Parameter(W * (rho / rho_W))
        bound = 1 / n**0.5
        self.bias = nn.Parameter(torch.empty(n).uniform_(-bound, bound))
        self.weight.register_hook(lambda grad: grad * self.mask)

    @staticmethod
    def symmetric_mask(n, density):
        mask = torch.zeros(n, n)
        n_active = int(density * (n * (n - 1) // 2 + n))
        triu = torch.triu_indices(n, n)
        idx = torch.randperm(triu.shape[1])[:n_active]
        rows, cols = triu[0, idx], triu[1, idx]
        mask[rows, cols] = 1
        mask[cols, rows] = 1
        return mask + torch.eye(n)

    def effective_weight(self):
        return self.weight * self.mask

    def forward(self, h):
        return F.linear(h, self.effective_weight(), self.bias)


class CNON(nn.Module):
    """Multilayer CNON (port of `Oscillator_RNN_dyn` / `Oscillator_RNN_parallel`).

    Parameters
    ----------
    n_in, n_neurons, n_layers, n_out : network dimensions.
    alpha : damping coefficient.
    dt : Euler time step.
    max_steps : maximum (adaptive) or fixed number of Euler steps.
    eps_int : steady-state threshold on max|dh/dt| over all layers and samples
        of the batch (adaptive mode only).
    adaptive : stop at the steady state (True) or after `max_steps` (False).
    density, rho : sparsity and spectral radius of W_rec.
    train_reservoir : whether W_rec (and its bias) receive gradients.
    """

    def __init__(self, n_in=784, n_neurons=600, n_layers=1, n_out=10, alpha=3.0,
                 dt=0.1, max_steps=100, eps_int=1e-4, adaptive=True,
                 density=0.1, rho=0.8, train_reservoir=False):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.alpha = alpha
        self.dt = dt
        self.max_steps = max_steps
        self.eps_int = eps_int
        self.adaptive = adaptive

        self.W_input = nn.Linear(n_in, n_neurons)
        self.W_in = nn.ModuleList([nn.Linear(n_neurons, n_neurons)
                                   for _ in range(n_layers - 1)])
        self.W_rec = nn.ModuleList([SparseSymmetricLinear(n_neurons, density, rho)
                                    for _ in range(n_layers)])
        self.W_out = nn.Linear(n_neurons, n_out)

        # `init_esn_weights` of the original code: Xavier input weights and dense
        # inter-layer weights drawn from U(-1, 1) and scaled to the spectral radius
        # rho (the biases keep their default PyTorch initialisation). As in the
        # original, a matrix is drawn for every layer and the one of the first
        # layer is unused, so that a given seed builds the same network.
        with torch.no_grad():
            nn.init.xavier_uniform_(self.W_input.weight)
            for l in range(n_layers):
                W = 2 * torch.rand(n_neurons, n_neurons) - 1
                rho_W = spectral_radius(W)
                rho_W = rho_W if rho_W > 1e-6 else 1e-6
                if l > 0:
                    self.W_in[l - 1].weight.copy_(W * (rho / rho_W))
        for rec in self.W_rec:
            rec.weight.requires_grad_(train_reservoir)
            rec.bias.requires_grad_(train_reservoir)

    def forward(self, x):
        """Integrate the network for the input batch x (batch, n_in) or images.

        Returns
        -------
        y : (batch, n_out) readout W_out h^[L] + b_out.
        h : list of the final hidden states of every layer, (batch, n_neurons).
        n_steps : number of Euler steps taken.
        max_delta : max|dh/dt| over all layers and samples at the last step.
        """
        x_in = self.W_input(x.reshape(x.shape[0], -1))
        h = [x_in.new_zeros(x.shape[0], self.n_neurons) for _ in range(self.n_layers)]
        dh = [None] * self.n_layers
        n_steps = 0
        while True:
            n_steps += 1
            for l in range(self.n_layers):
                drive = x_in if l == 0 else self.W_in[l - 1](h[l - 1])
                dh[l] = -self.alpha * h[l] + torch.sin(drive + self.W_rec[l](h[l]))
                h[l] = h[l] + dh[l] * self.dt
            if self.adaptive:
                with torch.no_grad():
                    max_delta = max(float(d.abs().max()) for d in dh)
                if max_delta < self.eps_int or n_steps >= self.max_steps:
                    break
            elif n_steps >= self.max_steps:
                with torch.no_grad():
                    max_delta = max(float(d.abs().max()) for d in dh)
                break
        return self.W_out(h[-1]), h, n_steps, max_delta
