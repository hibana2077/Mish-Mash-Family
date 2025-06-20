import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn

# Define activation functions
def softplus(x):
    if isinstance(x, torch.Tensor):
        return F.softplus(x)
    else:
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def mish(x):
    if isinstance(x, torch.Tensor):
        return x * torch.tanh(softplus(x))
    else:
        return x * np.tanh(softplus(x))

class SineMish(nn.Module):
    def __init__(self, alpha=0.2, beta=np.pi, gamma=1.0, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, x):
        """
        gSineMish variant:
        f(x) = x * tanh(softplus(x)) + alpha * sigmoid(-gamma * x) * sin(beta * x)
        """
        sigmoid = F.sigmoid(-self.gamma * x)
        return mish(x) + self.alpha * sigmoid * torch.sin(self.beta * x)
    
class ParamMish(nn.Module):
    def __init__(self, init_beta=1.0, **kwargs):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(init_beta))
    def forward(self, x):
        return x * torch.tanh(F.softplus(self.beta * x))

class AsymMish(nn.Module):
    """
    Asymmetric Mish activation function (Asym-Mish)
        y = { x * tanh(softplus(beta_p * x))                   , x >= 0
            { alpha * x * tanh(softplus(beta_n * x))           , x < 0

    Learnable parameters (all support autograd):
        alpha  ∈ (0, 1]    # Controls negative region scaling; <1 suppresses negative saturation
        beta_p > 0         # Slope of softplus in positive region
        beta_n > 0         # Slope of softplus in negative region (can set < beta_p for "softer" negative side)
    """
    def __init__(self, alpha_init=0.5, beta_p_init=1.0, beta_n_init=1.0, eps: float = 1e-6):
        """
        alpha_init, beta_p_init, beta_n_init are initial values; do not need to satisfy constraints,
        but reasonable values can speed up convergence.
        """
        super().__init__()
        self.eps = eps  # Avoid numerical instability
        # Re-parameterisation to map "any real number" → "constrained interval":
        #   alpha_raw via sigmoid → (0,1)
        #   beta_raw  via softplus → (0,+∞)
        self.alpha_raw = nn.Parameter(self._inv_sigmoid(alpha_init))
        self.beta_p_raw = nn.Parameter(self._inv_softplus(beta_p_init))
        self.beta_n_raw = nn.Parameter(self._inv_softplus(beta_n_init))

    # ---------- util: map real number back to initial value ----------
    @staticmethod
    def _inv_sigmoid(y):
        y = float(min(max(y, 1e-4), 1 - 1e-4))  # Prevent infinite values
        return torch.log(torch.tensor(y) / (1 - y))

    @staticmethod
    def _inv_softplus(y):
        # softplus^{-1}(y) = log(exp(y) - 1)
        y = float(max(y, 1e-4))
        return torch.log(torch.exp(torch.tensor(y)) - 1)

    # ---------- forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.alpha_raw)              # (0,1)
        beta_p = F.softplus(self.beta_p_raw) + self.eps    # >0
        beta_n = F.softplus(self.beta_n_raw) + self.eps    # >0

        pos = x >= 0
        x_pos = x[pos]
        x_neg = x[~pos]

        out = torch.empty_like(x)
        if x_pos.numel() > 0:
            out[pos] = x_pos * torch.tanh(F.softplus(beta_p * x_pos))
        if x_neg.numel() > 0:
            out[~pos] = alpha * x_neg * torch.tanh(F.softplus(beta_n * x_neg))
        return out

    # Conveniently read learnable parameters
    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw).detach()

    @property
    def beta_p(self):
        return F.softplus(self.beta_p_raw).detach()

    @property
    def beta_n(self):
        return F.softplus(self.beta_n_raw).detach()
