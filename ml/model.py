"""MLP actor and critic for fixed-window market observations.

The single-symbol policy issues one of three commands: sell, do nothing, or buy.
Commands move inventory one step and positions are bounded to short, flat, or
long. The pair policy issues one joint command per tick for two legs at once.
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
    """Flat-window MLP with an optional per-decision scalar side-input.

    Window-level statistics such as a hedge ratio are appended once to the
    flattened window instead of being broadcast across every tick, which would
    spend ``window_size`` inputs to carry a single number.
    """

    def __init__(
        self,
        window_size: int,
        feature_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        scalar_dim: int = 0,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.feature_dim = int(feature_dim)
        self.scalar_dim = int(scalar_dim)
        if self.scalar_dim < 0:
            raise ValueError("scalar_dim must be non-negative")
        in_dim = self.window_size * self.feature_dim + self.scalar_dim
        self.main = make_mlp(in_dim, int(hidden_dim), out_dim, int(depth))
        self.apply(init_orthogonal)

    def forward_features(
        self, inputs: torch.Tensor, scalars: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        if inputs.ndim not in (3, 4):
            raise ValueError("inputs must have shape (batch, window, features) or (batch, time, window, features)")
        if inputs.shape[-2:] != (self.window_size, self.feature_dim):
            raise ValueError(
                f"expected trailing shape {(self.window_size, self.feature_dim)}, got {tuple(inputs.shape[-2:])}"
            )
        leading = tuple(inputs.shape[:-2])
        flat = inputs.reshape(-1, self.window_size * self.feature_dim)
        if self.scalar_dim:
            if scalars is None:
                raise ValueError(f"this model expects scalars with trailing dim {self.scalar_dim}")
            if scalars.shape[:-1] != leading or scalars.shape[-1] != self.scalar_dim:
                raise ValueError(
                    f"expected scalars with shape {leading + (self.scalar_dim,)}, got {tuple(scalars.shape)}"
                )
            flat = torch.cat((flat, scalars.reshape(-1, self.scalar_dim)), dim=-1)
        elif scalars is not None:
            raise ValueError("this model was built with scalar_dim=0 and cannot accept scalars")
        return flat, leading


class TradingActor(WindowMLP):
    """Categorical policy over sell, nothing, and buy commands."""

    def __init__(
        self,
        window_size: int,
        feature_dim: int = 5,
        hidden_dim: int = 256,
        depth: int = 4,
        action_dim: int = ACTION_DIM,
        scalar_dim: int = 0,
    ):
        if action_dim != ACTION_DIM:
            raise ValueError(f"buy/nothing/sell policy requires action_dim={ACTION_DIM}")
        super().__init__(
            window_size, feature_dim, hidden_dim, action_dim, depth, scalar_dim
        )
        nn.init.orthogonal_(self.main[-1].weight, gain=0.01)

    def forward(
        self, inputs: torch.Tensor, scalars: torch.Tensor | None = None
    ) -> torch.Tensor:
        flat, leading = self.forward_features(inputs, scalars)
        return self.main(flat).reshape(*leading, ACTION_DIM)

    def distribution(
        self, inputs: torch.Tensor, scalars: torch.Tensor | None = None
    ) -> Categorical:
        return Categorical(logits=self(inputs, scalars))

    @torch.no_grad()
    def play(
        self,
        inputs: torch.Tensor,
        scalars: torch.Tensor | None = None,
        sampling: str = "multinomial",
    ) -> torch.Tensor:
        logits = self(inputs, scalars)
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
        scalar_dim: int = 0,
    ):
        super().__init__(window_size, feature_dim, hidden_dim, 1, depth, scalar_dim)
        # Market windows can contain rare large moves. A small initial value
        # head keeps the first advantages reward-driven instead of dominated by
        # arbitrary critic predictions.
        nn.init.orthogonal_(self.main[-1].weight, gain=0.01)

    def forward(
        self, inputs: torch.Tensor, scalars: torch.Tensor | None = None
    ) -> torch.Tensor:
        flat, leading = self.forward_features(inputs, scalars)
        return self.main(flat).reshape(*leading)


PAIR_ACTION_DIM = ACTION_DIM * ACTION_DIM
PAIR_ACTION_NAMES = tuple(f"{a}/{b}" for a in ACTION_NAMES for b in ACTION_NAMES)


def decode_pair_action(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a joint index into per-leg sell/nothing/buy commands.

    The joint index is ``3 * command_a + command_b`` so leg A varies slowest.
    """
    if actions.dtype not in (torch.int32, torch.int64):
        raise TypeError("actions must contain integer categorical indices")
    if actions.numel() and ((actions < 0).any() or (actions >= PAIR_ACTION_DIM).any()):
        raise ValueError(f"actions must be in [0, {PAIR_ACTION_DIM - 1}]")
    long_actions = actions.to(torch.long)
    return torch.div(long_actions, ACTION_DIM, rounding_mode="floor"), long_actions.remainder(ACTION_DIM)


def encode_pair_action(actions_a: torch.Tensor, actions_b: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`decode_pair_action`."""
    if actions_a.shape != actions_b.shape:
        raise ValueError("actions_a and actions_b must have the same shape")
    return ACTION_DIM * actions_a.to(torch.long) + actions_b.to(torch.long)


def apply_pair_action(
    previous_a: torch.Tensor, previous_b: torch.Tensor, actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move both bounded leg inventories by one joint command.

    Each leg keeps the single-symbol semantics, so the joint action space spans
    flat, one-leg, spread, and doubled-up directional inventories.
    """
    command_a, command_b = decode_pair_action(actions)
    return apply_action(previous_a, command_a), apply_action(previous_b, command_b)


class PairTradingActor(WindowMLP):
    """Joint categorical policy over both legs of a two-symbol pair.

    A single 9-way head is used instead of two independent 3-way heads because
    the value of a command on one leg depends on the command issued to the
    other; a factorized policy cannot represent that coupling.
    """

    def __init__(
        self,
        window_size: int,
        feature_dim: int = 11,
        hidden_dim: int = 256,
        depth: int = 4,
        scalar_dim: int = 0,
        action_dim: int = PAIR_ACTION_DIM,
    ):
        if action_dim != PAIR_ACTION_DIM:
            raise ValueError(f"joint pair policy requires action_dim={PAIR_ACTION_DIM}")
        super().__init__(window_size, feature_dim, hidden_dim, action_dim, depth, scalar_dim)
        nn.init.orthogonal_(self.main[-1].weight, gain=0.01)

    def forward(self, inputs: torch.Tensor, scalars: torch.Tensor | None = None) -> torch.Tensor:
        flat, leading = self.forward_features(inputs, scalars)
        return self.main(flat).reshape(*leading, PAIR_ACTION_DIM)

    def distribution(self, inputs: torch.Tensor, scalars: torch.Tensor | None = None) -> Categorical:
        return Categorical(logits=self(inputs, scalars))

    @torch.no_grad()
    def play(
        self,
        inputs: torch.Tensor,
        scalars: torch.Tensor | None = None,
        sampling: str = "multinomial",
    ) -> torch.Tensor:
        logits = self(inputs, scalars)
        if sampling == "multinomial":
            return Categorical(logits=logits).sample()
        if sampling in ("argmax", "greedy"):
            return logits.argmax(dim=-1)
        raise ValueError(f"unknown sampling mode: {sampling}")


class PairTradingCritic(WindowMLP):
    def __init__(
        self,
        window_size: int,
        feature_dim: int = 11,
        hidden_dim: int = 256,
        depth: int = 4,
        scalar_dim: int = 0,
    ):
        super().__init__(window_size, feature_dim, hidden_dim, 1, depth, scalar_dim)
        nn.init.orthogonal_(self.main[-1].weight, gain=0.01)

    def forward(self, inputs: torch.Tensor, scalars: torch.Tensor | None = None) -> torch.Tensor:
        flat, leading = self.forward_features(inputs, scalars)
        return self.main(flat).reshape(*leading)


# Compatibility aliases matching the Spider project's concise class names.
Agent = TradingActor
Critic = TradingCritic
