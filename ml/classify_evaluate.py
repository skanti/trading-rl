"""Backtest binary price-direction checkpoints."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from classify import (
    CLASSIFY_SCALAR_NAMES,
    RelativeDirectionClassifier,
    binary_metrics,
    build_classify_features,
    build_classify_scalars,
    classify_feature_names,
    classify_labels,
)
from classify_dataset import (
    EXTENDED_OPEN_MINUTE,
    RTH_CLOSE_MINUTE,
    RTH_OPEN_MINUTE,
    RelativeDirectionDataset,
)
from reference_dataset import trailing_validation_start
from week_dataset import forward_fill_positions, read_universe


@lru_cache(maxsize=256)
def _cached_price_source(data_dir: str, sample_id: str) -> np.ndarray:
    """One mmap per symbol and loader process for a large minute sweep."""
    source = np.load(f"{data_dir}/{sample_id}.npy", mmap_mode="r")
    if source.ndim != 2 or source.shape[1] < 3:
        raise ValueError(f"{sample_id} must contain [seconds, price_mills, volume]")
    return source


def _cached_forward_filled_prices(
    data_dir: str, sample_id: str, secs: np.ndarray
) -> np.ndarray:
    source = _cached_price_source(data_dir, sample_id)
    position = forward_fill_positions(source, secs, sample_id)
    prices = np.asarray(source[position, 1], dtype=np.float64) / 1000.0
    return prices


class RegularMinuteSweepDataset(Dataset):
    """Expand each symbol/session across 09:30..15:59 entry minutes."""

    def __init__(self, base: RelativeDirectionDataset):
        if base.should_augment or base.fixed_anchor_offset is not None:
            raise ValueError("minute sweep requires an unaugmented, unpinned base dataset")
        self.base = base
        self.anchor_minutes = np.arange(
            RTH_OPEN_MINUTE, RTH_CLOSE_MINUTE, dtype=np.int64
        )
        self._cached_sample_index: int | None = None
        self._cached_prices: np.ndarray | None = None
        self._cached_reference: np.ndarray | None = None

    def _minute_matrices(self, sample_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Vectorize all 390 windows for one symbol/session and cache the result."""
        if self._cached_sample_index == sample_index:
            if self._cached_prices is None or self._cached_reference is None:
                raise AssertionError("minute matrix cache is incomplete")
            return self._cached_prices, self._cached_reference

        row = self.base.samples.iloc[sample_index]
        session = int(row.session)
        session_minutes = self.base.day_ticks * self.base.tick_minutes
        first = session - self.base.context_days
        minute_grid = np.concatenate(
            [
                self.base.context_sod[day]
                + np.arange(session_minutes, dtype=np.int64) * 60
                for day in range(first, session + 1)
            ]
        )
        anchor_positions = (
            self.base.context_days * session_minutes
            + self.anchor_minutes
            - EXTENDED_OPEN_MINUTE
        )
        backwards = (
            np.arange(self.base.window_size - 1, -1, -1, dtype=np.int64)
            * self.base.tick_minutes
        )
        history_indices = anchor_positions[:, None] - backwards[None, :]
        if history_indices.min() < 0 or history_indices.max() >= minute_grid.size:
            raise AssertionError("minute sweep history extends outside its session grid")
        history_secs = minute_grid[history_indices]
        if self.base.fixed_target_offset is None:
            target_minutes = self.anchor_minutes
        else:
            target_minutes = np.full_like(
                self.anchor_minutes,
                EXTENDED_OPEN_MINUTE
                + self.base.fixed_target_offset * self.base.tick_minutes,
            )
        target_secs = self.base.context_sod[session + self.base.horizon_days] + (
            target_minutes - EXTENDED_OPEN_MINUTE
        ) * 60
        secs = np.concatenate((history_secs, target_secs[:, None]), axis=1)
        sample_id = str(row.sample_id)
        self._cached_sample_index = sample_index
        self._cached_prices = _cached_forward_filled_prices(
            self.base.data_dir, sample_id, secs
        ).astype(np.float32)
        self._cached_reference = _cached_forward_filled_prices(
            self.base.data_dir, self.base.reference_symbol, secs
        ).astype(np.float32)
        return self._cached_prices, self._cached_reference

    def __len__(self) -> int:
        return len(self.base) * len(self.anchor_minutes)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_index, minute_index = divmod(int(index), len(self.anchor_minutes))
        row = self.base.samples.iloc[sample_index]
        anchor_minute = int(self.anchor_minutes[minute_index])
        target_minute = (
            anchor_minute
            if self.base.fixed_target_offset is None
            else EXTENDED_OPEN_MINUTE
            + self.base.fixed_target_offset * self.base.tick_minutes
        )
        prices, reference = self._minute_matrices(sample_index)
        sample_id = str(row.sample_id)
        progress = (anchor_minute - RTH_OPEN_MINUTE) / (
            RTH_CLOSE_MINUTE - RTH_OPEN_MINUTE
        )
        return {
            "_id": sample_id,
            "date": str(pd.Timestamp(row.date).date()),
            "target_date": str(
                pd.Timestamp(
                    self.base.calendar_dates[
                        int(row.session) + self.base.horizon_days
                    ]
                ).date()
            ),
            "weekday": np.int64(pd.Timestamp(row.date).dayofweek),
            "anchor_time": f"{anchor_minute // 60:02d}:{anchor_minute % 60:02d}",
            "target_time": f"{target_minute // 60:02d}:{target_minute % 60:02d}",
            "prices": prices[minute_index],
            "reference_prices": reference[minute_index],
            "anchor_progress": np.float32(progress),
        }


def parse_anchor_time(value: str) -> int:
    """Parse an Eastern ``HH:MM`` clock into minutes after midnight."""
    try:
        hour_text, minute_text = value.split(":", maxsplit=1)
        hour, minute = int(hour_text), int(minute_text)
    except (AttributeError, TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("anchor time must use HH:MM") from error
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise argparse.ArgumentTypeError("anchor time must use a valid 24-hour clock")
    result = hour * 60 + minute
    if not RTH_OPEN_MINUTE <= result <= RTH_CLOSE_MINUTE:
        raise argparse.ArgumentTypeError("anchor time must be inside 09:30..16:00 Eastern")
    return result


def load_classifier(
    checkpoint_path: str, device: torch.device
) -> tuple[RelativeDirectionClassifier, dict]:
    """Load a classifier while refusing incompatible checkpoint layouts."""
    with fsspec.open(checkpoint_path, "rb") as checkpoint_file:
        state = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    checkpoint_config = OmegaConf.create(state.get("config", {}))
    if not checkpoint_config.get("model") or not checkpoint_config.model.get("mlp"):
        raise ValueError("checkpoint does not contain model.mlp configuration")
    model_config = OmegaConf.to_container(checkpoint_config.model.mlp, resolve=True)
    if not isinstance(model_config, dict):
        raise TypeError("checkpoint model.mlp must be a mapping")
    feature_mode = str(checkpoint_config.data.get("feature_mode", "relative"))
    if tuple(state.get("feature_names", ())) != classify_feature_names(feature_mode):
        raise ValueError("checkpoint feature layout does not match its configuration")
    if tuple(state.get("scalar_names", ())) != CLASSIFY_SCALAR_NAMES:
        raise ValueError("checkpoint does not use the clock and weekday scalar layout")
    classifier = RelativeDirectionClassifier(**model_config).to(device)
    classifier.load_state_dict(state["model"], strict=True)
    classifier.eval()
    return classifier, state


def make_evaluation_dataset(
    cfg: DictConfig,
    symbols: tuple[str, ...],
    anchor_minute: int | None,
    weeks: int = 4,
) -> tuple[Dataset, pd.Timestamp, pd.Timestamp]:
    """Build every eligible symbol/session sample in the trailing eval weeks."""
    days = pd.read_csv(str(cfg.data.days_path))
    days.date = pd.to_datetime(days.date, format="%Y-%m-%d")
    last_date = pd.Timestamp(days.date.max())
    validation_start = trailing_validation_start(last_date, int(weeks))
    reference_symbol = str(cfg.data.reference_symbol)
    targets = days.loc[
        days.date.ge(validation_start)
        & days.sample_id.isin(symbols)
        & ~days.sample_id.eq(reference_symbol),
        ["sample_id", "date"],
    ].drop_duplicates()
    if not len(targets):
        raise ValueError("no top-universe samples fall inside the evaluation period")
    base = RelativeDirectionDataset(
        days=days,
        data_dir=str(cfg.data.data_dir),
        reference_symbol=reference_symbol,
        context_days=int(cfg.data.context_days),
        tick_minutes=int(cfg.data.tick_minutes),
        window_size=int(cfg.data.window_size),
        horizon_days=int(cfg.data.horizon_days),
        should_augment=False,
        targets=targets,
        seed=0,
        anchor_minute=anchor_minute,
        target_minute=cfg.data.get("target_minute", None),
    )
    dataset: Dataset = base if anchor_minute is not None else RegularMinuteSweepDataset(base)
    return dataset, validation_start, last_date


def _profit_factor(returns: pd.Series) -> float:
    gains = float(returns.clip(lower=0.0).sum())
    losses = float(-returns.clip(upper=0.0).sum())
    if losses > 0.0:
        return gains / losses
    return float("inf") if gains > 0.0 else float("nan")


def summarize_trades(
    frame: pd.DataFrame,
    candidate_logits: torch.Tensor,
    candidate_labels: torch.Tensor,
    checkpoint_path: str,
    global_step: int,
    validation_start: pd.Timestamp,
    last_date: pd.Timestamp,
    anchor_time: str,
    horizon_days: int,
    confidence_threshold: float,
    position_mode: str,
    transaction_cost_bps: float,
    requested_symbols: tuple[str, ...],
    entry_minutes_per_stock_day: int = 1,
    selection_mode: str = "threshold",
    target_time: str | None = None,
) -> dict[str, object]:
    candidate = binary_metrics(candidate_logits, candidate_labels)
    entry_minutes = int(entry_minutes_per_stock_day)
    if entry_minutes < 1 or candidate_labels.numel() % entry_minutes:
        raise ValueError("candidate count does not match the entry-minute sweep")
    summary: dict[str, object] = {
        "checkpoint": checkpoint_path,
        "global_step": global_step,
        "validation_start": str(validation_start.date()),
        "data_last_date": str(last_date.date()),
        "anchor_time_eastern": anchor_time,
        "target_time_eastern": target_time or anchor_time,
        "horizon_trading_days": horizon_days,
        "selection_mode": selection_mode,
        "max_trades_per_minute": 3 if selection_mode == "top3" else None,
        "max_trades_per_day": 5 if selection_mode == "top5_daily" else None,
        "confidence_threshold": confidence_threshold,
        "position_mode": position_mode,
        "transaction_cost_bps_per_side": transaction_cost_bps,
        "requested_symbols": len(requested_symbols),
        "entry_minutes_per_stock_day": entry_minutes,
        "eligible_stock_days": int(candidate_labels.numel() // entry_minutes),
        "candidate_samples": int(candidate_labels.numel()),
        "trades": int(len(frame)),
        "coverage": float(len(frame) / candidate_labels.numel()),
        **{
            f"candidate_{key}": value
            for key, value in candidate.items()
            if key != "samples"
        },
    }
    if frame.empty:
        summary.update(
            {
                "symbols_traded": 0,
                "long_trades": 0,
                "short_trades": 0,
                "signal_win_rate": float("nan"),
                "trade_win_rate": float("nan"),
                "mean_confidence": float("nan"),
                "mean_net_return_on_gross_capital": float("nan"),
                "median_net_return_on_gross_capital": float("nan"),
                "profit_factor": float("nan"),
                "equal_weight_cohort_return": float("nan"),
                "equal_weight_cohort_win_rate": float("nan"),
                "equal_weight_cohort_max_drawdown": float("nan"),
            }
        )
        return summary

    returns = frame.net_return_on_gross_capital
    cohorts = frame.groupby("entry_date", sort=True).net_return_on_gross_capital.mean()
    equity = cohorts.cumsum()
    equity_with_origin = pd.concat((pd.Series([0.0]), equity), ignore_index=True)
    drawdown = equity_with_origin.cummax() - equity_with_origin
    summary.update(
        {
            "symbols_traded": int(frame.sample_id.nunique()),
            "entry_dates_traded": int(frame.entry_date.nunique()),
            "long_trades": int(frame.direction.gt(0).sum()),
            "short_trades": int(frame.direction.lt(0).sum()),
            "signal_win_rate": float(frame.signal_correct.mean()),
            "trade_win_rate": float(frame.trade_won.mean()),
            "mean_confidence": float(frame.confidence.mean()),
            "mean_net_return_on_gross_capital": float(returns.mean()),
            "median_net_return_on_gross_capital": float(returns.median()),
            "total_unit_net_pnl": float(frame.net_pnl_per_stock_leg.sum()),
            "profit_factor": _profit_factor(returns),
            "equal_weight_cohort_return": float(cohorts.mean()),
            "equal_weight_cohort_win_rate": float(cohorts.gt(0).mean()),
            "equal_weight_cohort_max_drawdown": float(drawdown.max()),
        }
    )
    return summary


@torch.no_grad()
def evaluate_bets(
    classifier: RelativeDirectionClassifier,
    loader,
    cfg: DictConfig,
    device: torch.device,
    min_confidence: float,
    position_mode: str,
    transaction_cost_bps: float,
    show_progress: bool = False,
    max_trades_per_minute: int | None = None,
    max_trades_per_day: int | None = None,
) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    """Score candidates and return one row for every thresholded unit bet.

    ``relative`` uses long-stock/short-SPY for a positive prediction and the
    reverse for a negative one. Its return is divided by two to express P&L on
    the two legs' gross capital. ``stock`` trades only the selected stock.
    """
    if not 0.5 < float(min_confidence) <= 1.0:
        raise ValueError("min_confidence must be in (0.5, 1.0]")
    if position_mode not in ("relative", "stock"):
        raise ValueError("position_mode must be 'relative' or 'stock'")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")
    if max_trades_per_minute is not None and int(max_trades_per_minute) < 1:
        raise ValueError("max_trades_per_minute must be positive")
    if max_trades_per_day is not None and int(max_trades_per_day) < 1:
        raise ValueError("max_trades_per_day must be positive")
    if max_trades_per_minute is not None and max_trades_per_day is not None:
        raise ValueError("minute and daily trade limits are mutually exclusive")

    rows: list[dict[str, object]] = []
    ranked_candidates: dict[tuple[str, ...], list[dict[str, object]]] = {}
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    cost_rate = float(transaction_cost_bps) / 10_000.0
    batches = tqdm(loader, desc="scoring minute candidates", unit="batch") if show_progress else loader
    for batch in batches:
        prices = batch["prices"].to(device=device, dtype=torch.float32)
        reference = batch["reference_prices"].to(device=device, dtype=torch.float32)
        anchor_progress = batch["anchor_progress"].to(device=device, dtype=torch.float32)
        weekday = batch["weekday"].to(device=device, dtype=torch.long)
        scalars = build_classify_scalars(anchor_progress, weekday)
        features = build_classify_features(
            prices,
            reference,
            str(cfg.data.get("feature_mode", "relative")),
            float(cfg.data.get("price_feature_scale", 100.0)),
        )
        logits = classifier(features, scalars)
        labels = classify_labels(
            prices,
            reference,
            str(cfg.data.get("target_mode", "relative_direction")),
        )
        probabilities = logits.sigmoid()
        confidence = torch.maximum(probabilities, 1.0 - probabilities)
        selected = confidence.ge(float(min_confidence))
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

        stock_returns = prices[:, -1] / prices[:, -2] - 1.0
        reference_returns = reference[:, -1] / reference[:, -2] - 1.0
        directions = torch.where(probabilities.ge(0.5), 1.0, -1.0)
        for index in selected.nonzero(as_tuple=False).flatten().tolist():
            direction = float(directions[index])
            stock_return = float(stock_returns[index])
            reference_return = float(reference_returns[index])
            if position_mode == "relative":
                gross_pnl = direction * (stock_return - reference_return)
                transaction_cost = 4.0 * cost_rate  # two legs, entry and exit
                gross_capital = 2.0
                side = "long_stock_short_spy" if direction > 0 else "short_stock_long_spy"
            else:
                gross_pnl = direction * stock_return
                transaction_cost = 2.0 * cost_rate  # one leg, entry and exit
                gross_capital = 1.0
                side = "long_stock" if direction > 0 else "short_stock"
            net_pnl = gross_pnl - transaction_cost
            row = {
                "sample_id": batch["_id"][index],
                "entry_date": batch["date"][index],
                "exit_date": batch["target_date"][index],
                "anchor_time": batch["anchor_time"][index],
                "target_time": batch.get("target_time", batch["anchor_time"])[index],
                "side": side,
                "direction": int(direction),
                "probability_higher": float(probabilities[index]),
                "probability_outperform": float(probabilities[index]),
                "confidence": float(confidence[index]),
                "actual_higher": int(labels[index]),
                "actual_outperform": int(labels[index]),
                "signal_correct": bool(directions[index].gt(0).eq(labels[index].bool())),
                "stock_entry_price": float(prices[index, -2]),
                "stock_exit_price": float(prices[index, -1]),
                "spy_entry_price": float(reference[index, -2]),
                "spy_exit_price": float(reference[index, -1]),
                "stock_return": stock_return,
                "spy_return": reference_return,
                "gross_pnl_per_stock_leg": gross_pnl,
                "transaction_cost": transaction_cost,
                "net_pnl_per_stock_leg": net_pnl,
                "net_return_on_gross_capital": net_pnl / gross_capital,
                "trade_won": net_pnl > 0.0,
            }
            if max_trades_per_minute is None and max_trades_per_day is None:
                rows.append(row)
            else:
                if max_trades_per_day is not None:
                    cohort_key = (str(row["entry_date"]),)
                    cohort_limit = int(max_trades_per_day)
                else:
                    cohort_key = (str(row["entry_date"]), str(row["anchor_time"]))
                    cohort_limit = int(max_trades_per_minute)
                cohort = ranked_candidates.setdefault(cohort_key, [])
                cohort.append(row)
                cohort.sort(
                    key=lambda candidate: (
                        -float(candidate["confidence"]),
                        str(candidate["sample_id"]),
                    )
                )
                del cohort[cohort_limit:]
    if not all_logits:
        raise ValueError("evaluation loader produced no candidate batches")
    if max_trades_per_minute is not None or max_trades_per_day is not None:
        rows = [
            candidate
            for cohort_key in sorted(ranked_candidates)
            for candidate in ranked_candidates[cohort_key]
        ]
    frame = pd.DataFrame(rows)
    return frame, torch.cat(all_logits), torch.cat(all_labels)


def evaluate(
    config_path: str,
    checkpoint_path: str,
    anchor_minute: int | None = None,
    min_confidence: float = 0.70,
    top: int = 50,
    universe_path: str | None = None,
    weeks: int = 4,
    device_name: str | None = None,
    batch_size: int = 390,
    workers: int = 4,
    position_mode: str | None = None,
    transaction_cost_bps: float = 0.0,
    selection_mode: str = "threshold",
) -> tuple[pd.DataFrame, dict[str, object]]:
    if selection_mode not in ("threshold", "top3", "top5_daily"):
        raise ValueError("selection_mode must be 'threshold', 'top3', or 'top5_daily'")
    cfg = OmegaConf.load(config_path)
    if anchor_minute is None and cfg.data.get("anchor_minute", None) is not None:
        anchor_minute = int(cfg.data.anchor_minute)
    target_mode = str(cfg.data.get("target_mode", "relative_direction"))
    if position_mode is None:
        position_mode = "stock" if target_mode == "stock_direction" else "relative"
    device = torch.device(device_name or str(cfg.model.device))
    classifier, state = load_classifier(checkpoint_path, device)
    checkpoint_horizon = int(state.get("horizon_days", cfg.data.horizon_days))
    if checkpoint_horizon != int(cfg.data.horizon_days):
        raise ValueError("checkpoint horizon_days does not match the evaluation config")
    if str(state.get("reference_symbol", cfg.data.reference_symbol)) != str(
        cfg.data.reference_symbol
    ):
        raise ValueError("checkpoint reference symbol does not match the evaluation config")
    if classifier.window_size != int(cfg.data.window_size):
        raise ValueError("checkpoint window size does not match the evaluation config")
    feature_mode = str(cfg.data.get("feature_mode", "relative"))
    if tuple(state.get("feature_names", ())) != classify_feature_names(feature_mode):
        raise ValueError("checkpoint feature layout does not match the evaluation config")
    if str(state.get("target_mode", "relative_direction")) != target_mode:
        raise ValueError("checkpoint target mode does not match the evaluation config")

    universe = cfg.data.get("val_universe", None)
    ranked_path = universe_path or (
        str(universe.path) if universe is not None else "../data/tickers_all.txt"
    )
    symbols = read_universe(ranked_path, int(top), str(cfg.data.reference_symbol))
    dataset, validation_start, last_date = make_evaluation_dataset(
        cfg, symbols, anchor_minute, weeks
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=int(workers),
        shuffle=False,
        drop_last=False,
        pin_memory=device.type == "cuda",
        persistent_workers=int(workers) > 0,
    )
    frame, logits, labels = evaluate_bets(
        classifier,
        loader,
        cfg,
        device,
        min_confidence,
        position_mode,
        transaction_cost_bps,
        show_progress=True,
        max_trades_per_minute=3 if selection_mode == "top3" else None,
        max_trades_per_day=5 if selection_mode == "top5_daily" else None,
    )
    anchor_text = (
        "every minute 09:30..15:59"
        if anchor_minute is None
        else f"{anchor_minute // 60:02d}:{anchor_minute % 60:02d}"
    )
    configured_target = cfg.data.get("target_minute", None)
    target_text = (
        anchor_text
        if configured_target is None
        else f"{int(configured_target) // 60:02d}:{int(configured_target) % 60:02d}"
    )
    summary = summarize_trades(
        frame,
        logits,
        labels,
        checkpoint_path,
        int(state.get("global_step", 0)),
        validation_start,
        last_date,
        anchor_text,
        checkpoint_horizon,
        min_confidence,
        position_mode,
        transaction_cost_bps,
        symbols,
        len(dataset.anchor_minutes)
        if isinstance(dataset, RegularMinuteSweepDataset)
        else 1,
        selection_mode=selection_mode,
        target_time=target_text,
    )
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest a binary price-direction checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="classifier .ckpt to evaluate")
    parser.add_argument("--config_path", default="main.yaml")
    parser.add_argument(
        "--selection_mode",
        choices=("threshold", "top3", "top5_daily"),
        default="threshold",
        help=(
            "threshold trades every qualifying signal; top3 keeps the three most-"
            "confident qualifying symbols per minute; top5_daily keeps the five "
            "most-confident qualifying symbols per entry date"
        ),
    )
    parser.add_argument(
        "--anchor_time",
        type=parse_anchor_time,
        default=None,
        help=(
            "optional entry HH:MM Eastern time; defaults to data.anchor_minute when "
            "configured, otherwise sweeps every regular minute"
        ),
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.70,
        help="trade when max(p, 1-p) reaches this threshold, default: 0.70",
    )
    parser.add_argument("--top", type=int, default=50, help="ranked universe size")
    parser.add_argument("--universe_path", default=None)
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--device", default=None, help="defaults to model.device in main.yaml")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=390,
        help="390 keeps one stock-session minute sweep in each batch",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--position_mode",
        choices=("relative", "stock"),
        default=None,
        help=(
            "relative hedges with SPY; stock trades only the stock; defaults to "
            "stock for stock-direction targets and relative otherwise"
        ),
    )
    parser.add_argument("--transaction_cost_bps", type=float, default=0.0)
    parser.add_argument("--output_csv", default=None, help="optional trade-level CSV path")
    parser.add_argument("--summary_json", default=None, help="optional summary JSON path")
    args = parser.parse_args()

    frame, summary = evaluate(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint,
        anchor_minute=args.anchor_time,
        min_confidence=args.min_confidence,
        top=args.top,
        universe_path=args.universe_path,
        weeks=args.weeks,
        device_name=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
        position_mode=args.position_mode,
        transaction_cost_bps=args.transaction_cost_bps,
        selection_mode=args.selection_mode,
    )
    print(json.dumps(summary, indent=2, default=str))
    if args.output_csv:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(f"wrote {len(frame)} trades to {output}")
    if args.summary_json:
        output = Path(args.summary_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        print(f"wrote summary to {output}")


if __name__ == "__main__":
    main()
