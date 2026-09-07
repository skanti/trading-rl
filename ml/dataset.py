"""Market-day loader for autoregressive trading rollouts."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime

import fsspec
import numpy as np
from trading_rl.market_data.schema import BAR_INDEX, validate_bar_columns
import pandas as pd
import torch
from omegaconf import DictConfig
from rich.logging import RichHandler
from torch.utils.data import DataLoader, Dataset


logging.basicConfig(level=logging.INFO, handlers=[RichHandler()], force=True)
logger = logging.getLogger("DATASET")

# Largest ``decay ** -block`` factor tolerated inside the blocked OU scan.
MAX_SCAN_BLOCK_GAIN = 1e3
EXTENDED_SESSION_BARS = 16 * 60  # 04:00 through 19:59 Eastern.


def market_context_window_size(context_days: int) -> int:
    """Ticks visible at the open after ``context_days`` extended sessions."""
    if context_days < 1:
        raise ValueError("context_days must be positive")
    return int(context_days) * EXTENDED_SESSION_BARS + 1


@dataclass(frozen=True)
class BezierToyBatch:
    prices: torch.Tensor
    volumes: torch.Tensor
    progress: torch.Tensor
    target_positions: torch.Tensor
    anchor_counts: torch.Tensor
    latent_returns: torch.Tensor


def bernstein_matrix(points: torch.Tensor, degree: int) -> torch.Tensor:
    columns = []
    for i in range(degree + 1):
        coefficient = float(math.comb(degree, i))
        columns.append(coefficient * points.pow(i) * (1.0 - points).pow(degree - i))
    return torch.stack(columns, dim=-1)


def sample_bezier_curve(
    anchor_count: int,
    batch_size: int,
    ticks: int,
    device: torch.device,
    generator: torch.Generator | None,
    anchor_step_std: float = 0.035,
) -> torch.Tensor:
    """Sample smooth log-price trends through random cumulative anchors."""
    degree = anchor_count - 1
    anchor_steps = torch.randn(
        batch_size, anchor_count - 1, device=device, generator=generator
    ) * anchor_step_std
    anchors = torch.cat(
        (torch.zeros(batch_size, 1, device=device), anchor_steps.cumsum(dim=1)), dim=1
    )
    anchor_u = torch.linspace(0.0, 1.0, anchor_count, device=device)
    controls = torch.linalg.solve(bernstein_matrix(anchor_u, degree), anchors.T).T
    sample_u = torch.linspace(0.0, 1.0, ticks, device=device)
    return controls @ bernstein_matrix(sample_u, degree).T


def sample_bezier_log_prices(
    batch_size: int,
    ticks: int,
    device: torch.device,
    generator: torch.Generator | None,
    anchor_step_std: float = 0.035,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return latent Bezier log prices and the anchor count that produced each."""
    anchor_counts = torch.randint(3, 5, (batch_size,), device=device, generator=generator)
    latent_log_prices = torch.empty(batch_size, ticks, device=device)
    for anchor_count in (3, 4):
        selected = anchor_counts.eq(anchor_count)
        count = int(selected.sum().item())
        if count:
            latent_log_prices[selected] = sample_bezier_curve(
                anchor_count, count, ticks, device, generator, anchor_step_std
            )
    return latent_log_prices, anchor_counts


class OnlineBezierToyProvider:
    """Generate fresh Bezier trends with integrated return noise in memory.

    Noise is added to log returns and cumulatively integrated into prices. IID
    innovations are therefore unpredictable from the current observation and
    do not create the mechanical mean reversion caused by IID price-level noise.
    """

    def __init__(
        self,
        window_size: int,
        rollout_size: int,
        return_noise_std: float = 3e-4,
        flat_return_threshold: float = 2.5e-4,
    ):
        if window_size < 2 or rollout_size < 1:
            raise ValueError("window_size must be >= 2 and rollout_size must be >= 1")
        if return_noise_std < 0 or flat_return_threshold < 0:
            raise ValueError("return_noise_std and flat_return_threshold must be non-negative")
        self.window_size = int(window_size)
        self.rollout_size = int(rollout_size)
        self.return_noise_std = float(return_noise_std)
        self.flat_return_threshold = float(flat_return_threshold)

    @property
    def ticks(self) -> int:
        return self.window_size + self.rollout_size

    _bernstein_matrix = staticmethod(bernstein_matrix)

    def _sample_curve(
        self,
        anchor_count: int,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        return sample_bezier_curve(anchor_count, batch_size, self.ticks, device, generator)

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> BezierToyBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        device = torch.device(device)
        latent_log_prices, anchor_counts = sample_bezier_log_prices(
            batch_size, self.ticks, device, generator
        )

        latent_returns = latent_log_prices[:, 1:] - latent_log_prices[:, :-1]
        return_innovations = torch.randn(
            batch_size, self.ticks - 1, device=device, generator=generator
        ) * self.return_noise_std
        observed_log_returns = latent_returns + return_innovations
        observed_log_prices = torch.cat(
            (
                latent_log_prices[:, :1],
                latent_log_prices[:, :1] + observed_log_returns.cumsum(dim=1),
            ),
            dim=1,
        )
        base_log_price = torch.empty(batch_size, 1, device=device).uniform_(
            np.log(8.0), np.log(800.0), generator=generator
        )
        prices = torch.exp(base_log_price + observed_log_prices)

        future_returns = latent_returns[
            :, self.window_size - 1 : self.window_size - 1 + self.rollout_size
        ]
        target_positions = torch.where(
            future_returns.abs() > self.flat_return_threshold,
            future_returns.sign(),
            torch.zeros_like(future_returns),
        ).to(torch.long)

        volume_noise = torch.randn(
            batch_size, self.ticks, device=device, generator=generator
        ) * 0.35
        padded_returns = torch.nn.functional.pad(observed_log_returns, (1, 0))
        volumes = torch.exp(6.0 + volume_noise + 10.0 * padded_returns.abs())
        progress = torch.linspace(-1.0, 1.0, self.ticks, device=device).expand(batch_size, -1)
        return BezierToyBatch(prices, volumes, progress, target_positions, anchor_counts, latent_returns)


@dataclass(frozen=True)
class PairToyBatch:
    """One batch of time-matched two-symbol paths with a known spread process."""

    prices_a: torch.Tensor
    prices_b: torch.Tensor
    volumes_a: torch.Tensor
    volumes_b: torch.Tensor
    progress: torch.Tensor
    target_positions_a: torch.Tensor
    target_positions_b: torch.Tensor
    spread: torch.Tensor
    spread_decay: torch.Tensor
    is_mean_reverting: torch.Tensor
    beta: torch.Tensor
    latent_returns: torch.Tensor


def sample_ornstein_uhlenbeck(
    batch_size: int,
    ticks: int,
    decay: torch.Tensor,
    step_std: torch.Tensor,
    initial: torch.Tensor,
    device: torch.device,
    generator: torch.Generator | None = None,
    block: int = 64,
) -> torch.Tensor:
    """Sample ``s[t+1] = decay * s[t] + step_std * eps`` for per-sample decays.

    The recursion is evaluated as a blocked scan so the Python loop runs once
    per block rather than once per tick. ``decay = 1`` yields a random walk,
    which is how a structurally broken pair is generated.
    """
    if batch_size < 1 or ticks < 1:
        raise ValueError("batch_size and ticks must be positive")
    if block < 1:
        raise ValueError("block must be positive")
    for name, tensor in (("decay", decay), ("step_std", step_std), ("initial", initial)):
        if tensor.shape != (batch_size,):
            raise ValueError(f"{name} must have shape ({batch_size},), got {tuple(tensor.shape)}")
    if (decay <= 0).any() or (decay > 1).any():
        raise ValueError("decay must lie in (0, 1]")

    # The within-block closed form divides by ``decay ** offset``; cap the block
    # length so that factor never grows large enough to lose float32 precision.
    smallest_decay = float(decay.min().item())
    if smallest_decay < 1.0:
        affordable = int(math.log(MAX_SCAN_BLOCK_GAIN) / -math.log(smallest_decay))
        block = max(1, min(block, affordable))
    padded_ticks = ticks + (-ticks) % block
    innovations = (
        torch.randn(batch_size, padded_ticks, device=device, generator=generator)
        * step_std.unsqueeze(-1)
    ).view(batch_size, -1, block)

    offsets = torch.arange(block, device=device, dtype=decay.dtype)
    decay_column = decay.view(batch_size, 1, 1)
    forward_powers = decay_column.pow(offsets)
    local = forward_powers * torch.cumsum(innovations * decay_column.pow(-offsets), dim=-1)

    # ``carry`` holds the value the process had at the tick before each block,
    # so within a block it is discounted by ``decay ** (offset + 1)``.
    block_decay = decay.pow(block)
    block_tails = local[:, :, -1]
    carry = initial
    carries = []
    for index in range(block_tails.shape[1]):
        carries.append(carry)
        carry = carry * block_decay + block_tails[:, index]
    stacked_carries = torch.stack(carries, dim=1).unsqueeze(-1)
    path = (local + stacked_carries * forward_powers * decay_column).reshape(batch_size, padded_ticks)
    return path[:, :ticks]


class OnlinePairToyProvider:
    """Generate two time-matched symbols sharing a factor plus a traded spread.

    ``log P_A = c + s/2`` and ``log P_B = beta * c - s/2`` where ``c`` is the
    same unpredictable Bezier-plus-integrated-noise path used by the
    single-symbol provider and ``s`` is an Ornstein-Uhlenbeck spread. Only
    ``s`` is forecastable, so a policy can only beat a single-leg trader by
    reading the relationship between the two legs. A configurable fraction of
    pairs draw a random-walk spread instead, which is untradeable and must be
    left flat.

    The spread is parameterized by its per-step volatility rather than its
    stationary width, so every pair - mean-reverting or broken, fast or slow -
    shows the same leg-to-leg return correlation. The regimes can then only be
    separated by the autocorrelation of the residual, never by its scale. The
    common factor keeps the single-symbol provider's noise level so leg A alone
    remains the same marginal process a single-symbol policy was trained on.
    """

    def __init__(
        self,
        window_size: int,
        rollout_size: int,
        return_noise_std: float = 3e-4,
        flat_return_threshold: float = 2.5e-6,
        spread_step_std: float = 2.2e-4,
        half_life_min: float = 20.0,
        half_life_max: float = 240.0,
        broken_fraction: float = 0.35,
        beta_min: float = 0.8,
        beta_max: float = 1.25,
        trend_std: float = 0.035,
    ):
        if window_size < 2 or rollout_size < 1:
            raise ValueError("window_size must be >= 2 and rollout_size must be >= 1")
        if return_noise_std < 0 or flat_return_threshold < 0 or spread_step_std < 0:
            raise ValueError("noise, threshold, and spread scales must be non-negative")
        if not 1.0 <= half_life_min <= half_life_max:
            raise ValueError("require 1 <= half_life_min <= half_life_max")
        if not 0.0 <= broken_fraction <= 1.0:
            raise ValueError("broken_fraction must lie in [0, 1]")
        if not 0.0 < beta_min <= beta_max:
            raise ValueError("require 0 < beta_min <= beta_max")
        if trend_std < 0:
            raise ValueError("trend_std must be non-negative")
        self.window_size = int(window_size)
        self.rollout_size = int(rollout_size)
        self.return_noise_std = float(return_noise_std)
        self.flat_return_threshold = float(flat_return_threshold)
        self.spread_step_std = float(spread_step_std)
        self.half_life_min = float(half_life_min)
        self.half_life_max = float(half_life_max)
        self.broken_fraction = float(broken_fraction)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        # Scale of the shared trend. Lowering it isolates the relative signal,
        # which is how the fast gate keeps the directional trade from drowning
        # out the spread at a short window.
        self.trend_std = float(trend_std)

    @property
    def ticks(self) -> int:
        return self.window_size + self.rollout_size

    @property
    def reference_decay(self) -> float:
        """Decay of the geometric-mean half-life, used to size broken pairs."""
        reference_half_life = math.sqrt(self.half_life_min * self.half_life_max)
        return math.exp(-math.log(2.0) / reference_half_life)

    def _sample_spread(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_half_life = torch.empty(batch_size, device=device).uniform_(
            math.log(self.half_life_min), math.log(self.half_life_max), generator=generator
        )
        decay = torch.exp(-math.log(2.0) / log_half_life.exp())
        is_mean_reverting = (
            torch.rand(batch_size, device=device, generator=generator) >= self.broken_fraction
        )
        decay = torch.where(is_mean_reverting, decay, torch.ones_like(decay))

        step_std = torch.full_like(decay, self.spread_step_std)
        # Start each mean-reverting pair at its own stationary width so the
        # rollout never opens inside a transient warm-up. Broken pairs have no
        # stationary law, so they borrow the reference half-life's width.
        stationary_decay = torch.where(
            is_mean_reverting, decay, torch.full_like(decay, self.reference_decay)
        )
        stationary_std = step_std / (1.0 - stationary_decay.square()).clamp_min(1e-12).sqrt()
        initial = torch.randn(batch_size, device=device, generator=generator) * stationary_std
        spread = sample_ornstein_uhlenbeck(
            batch_size, self.ticks, decay, step_std, initial, device, generator
        )
        return spread, decay, is_mean_reverting

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> PairToyBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        device = torch.device(device)
        latent_log_prices, _ = sample_bezier_log_prices(
            batch_size, self.ticks, device, generator, self.trend_std
        )
        latent_returns = latent_log_prices[:, 1:] - latent_log_prices[:, :-1]
        return_innovations = torch.randn(
            batch_size, self.ticks - 1, device=device, generator=generator
        ) * self.return_noise_std
        common = torch.cat(
            (
                latent_log_prices[:, :1],
                latent_log_prices[:, :1] + (latent_returns + return_innovations).cumsum(dim=1),
            ),
            dim=1,
        )

        spread, decay, is_mean_reverting = self._sample_spread(batch_size, device, generator)
        beta = torch.empty(batch_size, 1, device=device).uniform_(
            self.beta_min, self.beta_max, generator=generator
        )
        base_a = torch.empty(batch_size, 1, device=device).uniform_(
            np.log(8.0), np.log(800.0), generator=generator
        )
        base_b = torch.empty(batch_size, 1, device=device).uniform_(
            np.log(8.0), np.log(800.0), generator=generator
        )
        log_prices_a = base_a + common + 0.5 * spread
        log_prices_b = base_b + beta * common - 0.5 * spread
        prices_a, prices_b = torch.exp(log_prices_a), torch.exp(log_prices_b)

        # Myopic oracle: the predictable part of the next log return is the
        # Bezier drift plus the mean-reversion pull on the spread.
        window_slice = slice(self.window_size - 1, self.window_size - 1 + self.rollout_size)
        drift = latent_returns[:, window_slice]
        pull = (decay.unsqueeze(-1) - 1.0) * spread[:, window_slice]
        expected_a = drift + 0.5 * pull
        expected_b = beta * drift - 0.5 * pull
        target_positions_a = self._targets(expected_a)
        target_positions_b = self._targets(expected_b)

        volumes_a = self._volumes(log_prices_a, batch_size, device, generator)
        volumes_b = self._volumes(log_prices_b, batch_size, device, generator)
        progress = torch.linspace(-1.0, 1.0, self.ticks, device=device).expand(batch_size, -1)
        return PairToyBatch(
            prices_a=prices_a,
            prices_b=prices_b,
            volumes_a=volumes_a,
            volumes_b=volumes_b,
            progress=progress,
            target_positions_a=target_positions_a,
            target_positions_b=target_positions_b,
            spread=spread,
            spread_decay=decay,
            is_mean_reverting=is_mean_reverting,
            beta=beta.squeeze(-1),
            latent_returns=latent_returns,
        )

    def _targets(self, expected_returns: torch.Tensor) -> torch.Tensor:
        return torch.where(
            expected_returns.abs() > self.flat_return_threshold,
            expected_returns.sign(),
            torch.zeros_like(expected_returns),
        ).to(torch.long)

    def _volumes(
        self,
        log_prices: torch.Tensor,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        noise = torch.randn(batch_size, self.ticks, device=device, generator=generator) * 0.35
        returns = torch.nn.functional.pad(log_prices[:, 1:] - log_prices[:, :-1], (1, 0))
        return torch.exp(6.0 + noise + 10.0 * returns.abs())


class MarketDayDataset(Dataset):
    """Return an N-tick context followed by T tradable price intervals.

    Input ``.npy`` files use the eight-column OHLCV bar schema.
    Symbol identity and absolute price are intentionally not returned to the
    policy, preventing the easiest forms of symbol-specific memorization.
    ``eod_idx`` is treated as an inclusive index, as in the original dataset.
    """

    def __init__(
        self,
        days: pd.DataFrame,
        data_dir: str,
        window_size: int,
        rollout_size: int,
        should_augment: bool = False,
        require_full_session: bool = False,
        limit: int | None = None,
        calendar_days: pd.DataFrame | None = None,
    ):
        self.data_dir = data_dir
        self.window_size = int(window_size)
        self.rollout_size = int(rollout_size)
        self.should_augment = bool(should_augment)
        self.require_full_session = bool(require_full_session)
        if self.window_size < 2 or self.rollout_size < 1:
            raise ValueError("window_size must be >= 2 and rollout_size must be >= 1")

        schedule_source = calendar_days if calendar_days is not None else days
        self.session_schedule: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        if {"sod_sec", "eod_sec"}.issubset(schedule_source.columns):
            has_context_schedule = {
                "context_sod_sec",
                "context_eod_sec",
            }.issubset(schedule_source.columns)
            for sample_id, group in schedule_source.groupby("sample_id", sort=False):
                ordered = group.sort_values("sod_sec")
                session_sod = ordered["sod_sec"].to_numpy(dtype=np.int64)
                session_eod = ordered["eod_sec"].to_numpy(dtype=np.int64)
                if has_context_schedule:
                    context_sod = ordered["context_sod_sec"].to_numpy(dtype=np.int64)
                    context_eod = ordered["context_eod_sec"].to_numpy(dtype=np.int64)
                else:
                    # Backward compatibility for older metadata and small unit
                    # fixtures. Production 10-day indices carry explicit
                    # 04:00--19:59 boundaries.
                    context_sod, context_eod = session_sod, session_eod
                self.session_schedule[str(sample_id)] = (
                    session_sod,
                    session_eod,
                    context_sod,
                    context_eod,
                )

        filtered = days.copy()
        context_floor = (
            filtered["ctx_idx"].fillna(0).astype(np.int64)
            if "ctx_idx" in filtered
            else pd.Series(0, index=filtered.index, dtype=np.int64)
        )
        has_expected_times = {"sod_sec", "eod_sec"}.issubset(filtered.columns)
        if has_expected_times and self.session_schedule:
            has_context = pd.Series(False, index=filtered.index)
            for sample_id, group in filtered.groupby("sample_id", sort=False):
                schedule = self.session_schedule.get(str(sample_id))
                if schedule is None:
                    continue
                schedule_sod, _, context_sod, context_eod = schedule
                session_lengths = (context_eod - context_sod) // 60 + 1
                prefix_ticks = np.concatenate(([0], np.cumsum(session_lengths)))
                positions = np.searchsorted(
                    schedule_sod,
                    group["sod_sec"].to_numpy(dtype=np.int64),
                    side="left",
                )
                has_context.loc[group.index] = prefix_ticks[positions] >= self.window_size - 1
        else:
            has_context = filtered["sod_idx"].astype(np.int64) >= context_floor + self.window_size - 1
        if has_expected_times:
            session_intervals = (
                filtered["eod_sec"].astype(np.int64) - filtered["sod_sec"].astype(np.int64)
            ) // 60
        else:
            session_intervals = filtered["eod_idx"].astype(np.int64) - filtered["sod_idx"].astype(np.int64)
        has_rollout = session_intervals >= self.rollout_size
        if self.require_full_session:
            has_rollout &= session_intervals == self.rollout_size
        if "is_tradable" in filtered:
            is_tradable = filtered["is_tradable"].fillna(False).astype(bool)
        else:
            is_tradable = pd.Series(True, index=filtered.index)
        filtered = filtered[is_tradable & has_context & has_rollout]
        if limit is not None:
            filtered = filtered.iloc[:limit]
        self.days = filtered.reset_index(drop=True)
        removed = len(days) - len(self.days)
        if removed:
            logger.info(
                "Filtered unusable market days, removed=%d, retained=%d, full_session=%s",
                removed,
                len(self.days),
                self.require_full_session,
            )
        if not len(self.days):
            raise ValueError("no market days satisfy the configured context and rollout lengths")

    def __len__(self) -> int:
        return len(self.days)

    def _expected_context_secs(self, sample_id: str, sod_sec: int, count: int) -> np.ndarray:
        schedule = self.session_schedule.get(sample_id)
        if schedule is None:
            raise ValueError(f"sample {sample_id} has no market-session schedule")
        session_sod, _, context_sod, context_eod = schedule
        position = int(np.searchsorted(session_sod, sod_sec, side="left"))
        pieces = []
        remaining = count
        for index in range(position - 1, -1, -1):
            grid = np.arange(context_sod[index], context_eod[index] + 60, 60, dtype=np.int64)
            take = min(remaining, grid.size)
            pieces.append(grid[-take:])
            remaining -= take
            if remaining == 0:
                break
        if remaining:
            raise ValueError(f"sample {sample_id} lacks {count} scheduled context minutes")
        return np.concatenate(pieces[::-1])

    def __getitem__(self, idx: int) -> dict:
        sample = self.days.iloc[idx]
        sample_id = str(sample.sample_id)
        sample_path = f"{self.data_dir}/{sample_id}.npy"
        if "://" not in sample_path:
            data = np.load(sample_path, mmap_mode="r")
        else:
            with fsspec.open(sample_path, "rb") as f:
                data = np.load(f)
        validate_bar_columns(data, "1Min", str(sample_path))

        n, t = self.window_size, self.rollout_size
        sod_idx, eod_idx = int(sample.sod_idx), int(sample.eod_idx)
        context_floor = int(sample.ctx_idx) if "ctx_idx" in sample.index and not pd.isna(sample.ctx_idx) else 0

        has_expected_times = (
            "sod_sec" in sample.index
            and "eod_sec" in sample.index
            and not pd.isna(sample.sod_sec)
            and not pd.isna(sample.eod_sec)
        )
        if self.require_full_session and has_expected_times:
            sod_sec, eod_sec = int(sample.sod_sec), int(sample.eod_sec)
            session_secs = np.arange(sod_sec, eod_sec + 60, 60, dtype=np.int64)
            if session_secs.size != t + 1:
                raise ValueError(
                    f"sample {sample_id} expected {session_secs.size - 1} session intervals, not {t}"
                )
            context_secs = self._expected_context_secs(sample_id, sod_sec, n - 1)
            expected_secs = np.concatenate((context_secs, session_secs))

            source_secs = data[:, 0].astype(np.int64)
            # A missing bar means no new trade-derived observation. Carry the
            # most recent price forward and assign zero volume for that minute.
            price_source = np.searchsorted(source_secs, expected_secs, side="right") - 1
            if (price_source < 0).any():
                raise ValueError(f"sample {sample_id} cannot forward-fill its first session price")
            exact_source = np.searchsorted(source_secs, expected_secs, side="left")
            in_bounds = exact_source < source_secs.size
            exact = in_bounds.copy()
            exact[in_bounds] &= source_secs[exact_source[in_bounds]] == expected_secs[in_bounds]

            prices = data[price_source, 1].astype(np.float32) / 1000.0
            volumes = np.zeros(n + t, dtype=np.float32)
            volumes[exact] = data[exact_source[exact], BAR_INDEX["volume"]].astype(np.float32)
            secs = expected_secs
            if prices.size != n + t or volumes.size != n + t or secs.size != n + t:
                raise ValueError(f"sample {sample_id} completion produced an invalid segment length")
            if not np.isfinite(prices).all() or (prices <= 0).any():
                raise ValueError(f"sample {sample_id} contains non-positive or non-finite prices")
            if not np.isfinite(volumes).all() or (volumes < 0).any():
                raise ValueError(f"sample {sample_id} contains negative or non-finite volumes")
            return {"_id": sample_id, "prices": prices, "volumes": volumes, "secs": secs}

        # The action at last_context_idx receives the preceding N ticks. Every
        # selected position must have a subsequent price inside RTH.
        last_context_low = max(sod_idx, context_floor + n - 1)
        last_context_high = eod_idx - t
        if last_context_high < last_context_low:
            raise ValueError(
                f"sample {sample_id} is too short for window_size={n}, rollout_size={t}: "
                f"sod_idx={sod_idx}, eod_idx={eod_idx}, ctx_idx={context_floor}"
            )
        if self.require_full_session:
            if last_context_low != sod_idx or last_context_high != sod_idx:
                raise ValueError(f"sample {sample_id} does not describe one complete market session")
            last_context_idx = sod_idx
        elif self.should_augment:
            last_context_idx = int(np.random.randint(last_context_low, last_context_high + 1))
        else:
            last_context_idx = last_context_low

        start = last_context_idx - n + 1
        stop = last_context_idx + t + 1
        segment = data[start:stop]
        if segment.shape[0] != n + t:
            raise ValueError(f"sample {sample_id} ended unexpectedly at index {stop - 1}")

        prices = segment[:, 1].astype(np.float32) / 1000.0
        volumes = segment[:, BAR_INDEX["volume"]].astype(np.float32)
        secs = segment[:, 0].astype(np.int64)
        if self.require_full_session:
            session_secs = secs[n - 1 :]
            if not (np.all(session_secs % 60 == 0) and np.all(np.diff(session_secs) == 60)):
                raise ValueError(f"sample {sample_id} full-session ticks must be contiguous one-minute intervals")
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError(f"sample {sample_id} contains non-positive or non-finite prices")
        if not np.isfinite(volumes).all() or (volumes < 0).any():
            raise ValueError(f"sample {sample_id} contains negative or non-finite volumes")

        return {"_id": sample_id, "prices": prices, "volumes": volumes, "secs": secs}


# Preserve the old public name while changing its output to raw market arrays.
TokLoader = MarketDayDataset


def make_dataloader(cfg_data: DictConfig, cfg_split: DictConfig, seed: int) -> DataLoader:
    del seed  # DataLoader sampling uses PyTorch's process seed.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        torch.distributed.barrier()

    with fsspec.open(cfg_data.days_path, "r") as f:
        days = pd.read_csv(f)
    context_days = cfg_data.get("context_days", None)
    if context_days is not None:
        expected_window = market_context_window_size(int(context_days))
        if int(cfg_data.window_size) != expected_window:
            raise ValueError(
                f"window_size must be {expected_window} for context_days={int(context_days)} "
                f"(got {int(cfg_data.window_size)})"
            )
        required = {"context_sod_sec", "context_eod_sec"}
        missing = required.difference(days.columns)
        if missing:
            raise ValueError(
                f"day metadata lacks extended-session columns: {sorted(missing)}; "
                "rebuild it with scripts/split.py"
            )
    days.date = pd.to_datetime(days.date, format="%Y-%m-%d")
    calendar_days = days

    date_val = datetime.strptime(cfg_data.date_val, "%Y-%m-%d")
    if cfg_split.split == "val":
        # Validation remains bounded without exposing symbol identity to the
        # model or depending on the obsolete tokenizer mapping file.
        shortlist = set(sorted(days.sample_id.unique())[:32])
        mask = (days.date >= date_val) & days.sample_id.isin(shortlist)
    else:
        mask = days.date < date_val
    days = days[mask].reset_index(drop=True)
    rank_indices = np.array_split(np.arange(len(days)), world_size)[rank]
    days = days.iloc[rank_indices].reset_index(drop=True)
    logger.info("Samples found, samples_num=%d, split=%s", len(days), cfg_split.split)

    dataset = MarketDayDataset(
        days=days,
        data_dir=cfg_data.data_dir,
        window_size=int(cfg_data.window_size),
        rollout_size=int(cfg_data.rollout_size),
        limit=cfg_split.get("samples_num", None),
        should_augment=cfg_split.get("should_augment", False),
        require_full_session=cfg_data.get("require_full_session", False),
        calendar_days=calendar_days,
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg_split.batch_size),
        num_workers=int(cfg_split.workers_num),
        shuffle=bool(cfg_split.get("should_augment", False)),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(cfg_split.workers_num) > 0,
    )
