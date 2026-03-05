from __future__ import annotations

import torch
from torch import nn


class TimeConditionedMLP(nn.Module):
    """Predicts velocity field v_theta(x, t) for 2D flow matching."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, depth: int = 3):
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim + 1  # x plus scalar time t
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)


@torch.no_grad()
def sample_with_euler(
    model: nn.Module,
    n_samples: int,
    *,
    dim: int = 2,
    n_steps: int = 200,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Integrate dx/dt = v_theta(x,t), t in [0,1], with forward Euler."""

    model.eval()
    x = torch.randn(n_samples, dim, device=device)
    dt = 1.0 / float(n_steps)

    for step in range(n_steps):
        t_value = step / float(n_steps)
        t = torch.full((n_samples, 1), t_value, device=device)
        v = model(x, t)
        x = x + dt * v

    return x
