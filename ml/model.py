"""MLP actor and critic for fixed-window market observations.

The policy issues one of three commands: sell, do nothing, or buy. Commands
move inventory one step and positions are bounded to short, flat, or long.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


BUY_ACTION = 0
NOTHING_ACTION = 1
SELL_ACTION = 2
ACTION_DIM = 3
MAX_POSITION = 1
ACTION_NAMES = ("buy", "nothing", "sell")


def apply_action(previous_positions: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Apply sell/nothing/buy commands to bounded integer positions.

    Repeating buy while long or sell while short is an idempotent no-op.
    Reversing therefore takes two commands: close, then open the other side.
    """
    if actions.dtype not in (torch.int32, torch.int64):
        raise TypeError("actions must contain integer categorical indices")
    if previous_positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("previous_positions must contain integer inventory levels")
    if actions.numel() and ((actions < 0).any() or (actions >= ACTION_DIM).any()):
        raise ValueError(f"actions must be in [0, {ACTION_DIM - 1}]")
    if previous_positions.shape != actions.shape:
        raise ValueError("previous_positions and actions must have the same shape")
    if previous_positions.numel() and (
        (previous_positions < -MAX_POSITION).any() or (previous_positions > MAX_POSITION).any()
    ):
        raise ValueError(f"previous_positions must be in [-{MAX_POSITION}, {MAX_POSITION}]")
    delta = NOTHING_ACTION - actions.to(torch.long)
    return (previous_positions.to(torch.long) + delta).clamp(-MAX_POSITION, MAX_POSITION)


def init_orthogonal(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(module.bias)


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 4) -> nn.Sequential:
    """Spider-style flat-window MLP with configurable hidden depth."""
    if depth < 1:
        raise ValueError("depth must be at least one")
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth - 1):
        layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class WindowMLP(nn.Module):
    def __init__(self, window_size: int, feature_dim: int, hidden_dim: int, out_dim: int, depth: int):
        super().__init__()
        self.window_size = int(window_size)
        self.feature_dim = int(feature_dim)
        self.main = make_mlp(self.window_size * self.feature_dim, int(hidden_dim), out_dim, int(depth))
        self.apply(init_orthogonal)

    def forward_features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if inputs.ndim not in (3, 4):
            raise ValueError("inputs must have shape (batch, window, features) or (batch, time, window, features)")
        if inputs.shape[-2:] != (self.window_size, self.feature_dim):
            raise ValueError(
                f"expected trailing shape {(self.window_size, self.feature_dim)}, got {tuple(inputs.shape[-2:])}"
            )
        leading = tuple(inputs.shape[:-2])
        return inputs.reshape(-1, self.window_size * self.feature_dim), leading


class TradingActor(WindowMLP):
    """Categorical policy over sell, nothing, and buy commands."""

    def __init__(
        self,
        window_size: int,
        feature_dim: int = 5,
        hidden_dim: int = 256,
        depth: int = 4,
        action_dim: int = ACTION_DIM,
    ):
        if action_dim != ACTION_DIM:
            raise ValueError(f"buy/nothing/sell policy requires action_dim={ACTION_DIM}")
        super().__init__(window_size, feature_dim, hidden_dim, action_dim, depth)
        nn.init.orthogonal_(self.main[-1].weight, gain=0.01)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flat, leading = self.forward_features(inputs)
        return self.main(flat).reshape(*leading, ACTION_DIM)

    def distribution(self, inputs: torch.Tensor) -> Categorical:
        return Categorical(logits=self(inputs))

    @torch.no_grad()
    def play(self, inputs: torch.Tensor, sampling: str = "multinomial") -> torch.Tensor:
        logits = self(inputs)
        if sampling == "multinomial":
            return Categorical(logits=logits).sample()
        if sampling in ("argmax", "greedy"):
            return logits.argmax(dim=-1)
        raise ValueError(f"unknown sampling mode: {sampling}")


class TradingCritic(WindowMLP):
    def __init__(
        self,
        window_size: int,
        feature_dim: int = 5,
        hidden_dim: int = 256,
        depth: int = 4,
    ):
        super().__init__(window_size, feature_dim, hidden_dim, 1, depth)
        # Market windows can contain rare large moves. A small initial value
        # head keeps the first advantages reward-driven instead of dominated by
        # arbitrary critic predictions.
        nn.init.orthogonal_(self.main[-1].weight, gain=0.01)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flat, leading = self.forward_features(inputs)
        return self.main(flat).reshape(*leading)


# Compatibility aliases matching the Spider project's concise class names.
Agent = TradingActor
Critic = TradingCritic
