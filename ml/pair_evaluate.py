"""Score a pair checkpoint against every reference on identical price paths.

Each strategy trades the same two legs over the same ticks and is charged by
the same reward, so the numbers differ only in how the two legs are used. The
single-symbol policy is included twice: once on leg A alone, which reproduces
the published single-symbol KPI, and once run independently on both legs, which
is the benchmark that controls for simply deploying twice the capital.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch
from omegaconf import DictConfig, OmegaConf

from . import model, utils
from .dataset import OnlinePairToyProvider
from .pair import PAIR_SCALAR_DIM, collect_pair_rollout, pair_rollout_metrics
from .pair_eval import (
    current_residual_z,
    independent_leg_positions,
    oracle_positions,
    score_positions,
    spread_behaviour,
    zscore_rule_positions,
)
from .train import collect_rollout, performance_metrics


REPORTED_METRICS = (
    "return",
    "profit_factor",
    "max_drawdown",
    "trades",
    "gross_exposure",
    "abs_net_exposure",
    "spread_fraction",
    "directional_fraction",
    "single_leg_fraction",
    "flat_fraction",
    "basket_pnl",
    "spread_pnl",
)


@dataclass
class PairBatch:
    prices_a: torch.Tensor
    prices_b: torch.Tensor
    volumes_a: torch.Tensor
    volumes_b: torch.Tensor
    progress: torch.Tensor
    target_positions_a: torch.Tensor | None = None
    target_positions_b: torch.Tensor | None = None


def load_pair_actor(checkpoint_path: str, device: torch.device) -> tuple[model.PairTradingActor, DictConfig]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = OmegaConf.create(state["config"])
    actor = model.PairTradingActor(**OmegaConf.to_container(cfg.model.actor, resolve=True)).to(device)
    actor.load_state_dict(state["actor"], strict=True)
    actor.eval()
    return actor, cfg


def load_single_actor(checkpoint_path: str, device: torch.device) -> model.TradingActor:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = OmegaConf.create(state["config"])
    actor = model.TradingActor(**OmegaConf.to_container(cfg.model.actor, resolve=True)).to(device)
    actor.load_state_dict(state["actor"], strict=True)
    actor.eval()
    return actor


def toy_batches(cfg: DictConfig, batch_size: int, batches: int, device: torch.device, seed: int):
    toy = cfg.data.get("toy", {})
    provider = OnlinePairToyProvider(
        window_size=int(cfg.data.window_size),
        rollout_size=int(cfg.data.rollout_size),
        return_noise_std=float(toy.get("return_noise_std", 3e-4)),
        flat_return_threshold=float(toy.get("flat_return_threshold", 2.5e-6)),
        spread_step_std=float(toy.get("spread_step_std", 2.2e-4)),
        half_life_min=float(toy.get("half_life_min", 20.0)),
        half_life_max=float(toy.get("half_life_max", 240.0)),
        broken_fraction=float(toy.get("broken_fraction", 0.35)),
        beta_min=float(toy.get("beta_min", 0.8)),
        beta_max=float(toy.get("beta_max", 1.25)),
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(batches):
        sample = provider.sample(batch_size, device, generator)
        yield PairBatch(
            prices_a=sample.prices_a,
            prices_b=sample.prices_b,
            volumes_a=sample.volumes_a,
            volumes_b=sample.volumes_b,
            progress=sample.progress,
            target_positions_a=sample.target_positions_a,
            target_positions_b=sample.target_positions_b,
        ), sample


def market_batches(cfg: DictConfig, cfg_split: DictConfig, device: torch.device, max_batches: int):
    from pair_dataset import make_pair_dataloader
    from train import prepare_batch

    loader = make_pair_dataloader(cfg.data, cfg_split, 0)
    for index, raw in enumerate(loader):
        if index >= max_batches:
            break
        prices_a, volumes_a, _, progress = prepare_batch(
            {"prices": raw["prices_a"], "volumes": raw["volumes_a"], "secs": raw["secs"]},
            device,
            cfg.data.anno,
        )
        yield PairBatch(
            prices_a=prices_a,
            prices_b=raw["prices_b"].to(device=device, dtype=torch.float32),
            volumes_a=volumes_a,
            volumes_b=raw["volumes_b"].to(device=device, dtype=torch.float32),
            progress=progress,
        ), None


def aggregate(rewards: list[torch.Tensor]) -> dict[str, float]:
    return performance_metrics(torch.cat(rewards, dim=0))


@torch.no_grad()
def evaluate(
    pair_actor: model.PairTradingActor,
    single_actor: model.TradingActor | None,
    cfg: DictConfig,
    batches,
    entry_threshold: float,
    exit_threshold: float,
) -> dict[str, dict[str, float]]:
    window = int(cfg.data.window_size)
    steps = int(cfg.data.rollout_size)
    scale = float(cfg.data.get("price_feature_scale", 100.0))
    cost = float(cfg.model.transaction_cost)
    net_penalty = float(cfg.model.net_risk_penalty)
    gross_penalty = float(cfg.model.gross_risk_penalty)

    collected: dict[str, list[dict[str, float]]] = {}
    single_leg_rewards: list[torch.Tensor] = []

    def record(name: str, metrics: dict[str, float]) -> None:
        collected.setdefault(name, []).append(metrics)

    for batch, toy_sample in batches:
        common = dict(
            prices_a=batch.prices_a,
            prices_b=batch.prices_b,
            volumes_a=batch.volumes_a,
            volumes_b=batch.volumes_b,
            progress=batch.progress,
        )
        residual_z, hedge_ratio = current_residual_z(**common, window_size=window, rollout_size=steps,
                                                    price_feature_scale=scale)
        reference = dict(
            **common,
            window_size=window,
            rollout_size=steps,
            transaction_cost=cost,
            net_risk_penalty=net_penalty,
            gross_risk_penalty=gross_penalty,
            price_feature_scale=scale,
            hedge_ratio=hedge_ratio,
        )

        rollout = collect_pair_rollout(
            pair_actor, **common, rollout_size=steps, transaction_cost=cost,
            net_risk_penalty=net_penalty, gross_risk_penalty=gross_penalty,
            sampling="greedy", price_feature_scale=scale,
        )
        metrics = pair_rollout_metrics(rollout)
        if toy_sample is not None:
            # Only the toy knows which pairs actually mean revert, so this is
            # where standing aside on a broken pair can be verified.
            metrics.update(spread_behaviour(rollout, residual_z, toy_sample.is_mean_reverting))
        record("pair_policy", metrics)

        rule_a, rule_b = zscore_rule_positions(residual_z, entry_threshold, exit_threshold)
        record("zscore_rule", pair_rollout_metrics(score_positions(rule_a, rule_b, **reference)))

        if single_actor is not None:
            leg_a, leg_b = independent_leg_positions(
                single_actor, **common, rollout_size=steps, sampling="greedy",
                price_feature_scale=scale,
            )
            record("baseline_two_legs", pair_rollout_metrics(score_positions(leg_a, leg_b, **reference)))
            # Leg A on its own, scored exactly as the single-symbol trainer does.
            single = collect_rollout(
                single_actor, batch.prices_a, batch.volumes_a, batch.progress, steps,
                transaction_cost=cost, risk_penalty=net_penalty, sampling="greedy",
                price_feature_scale=scale,
            )
            single_leg_rewards.append(single.rewards)

        if toy_sample is not None:
            oracle_a, oracle_b = oracle_positions(
                toy_sample.target_positions_a, toy_sample.target_positions_b
            )
            record("oracle_myopic", pair_rollout_metrics(score_positions(oracle_a, oracle_b, **reference)))

    summary = {
        name: {key: float(sum(m[key] for m in runs) / len(runs)) for key in runs[0]}
        for name, runs in collected.items()
    }
    if single_leg_rewards:
        summary["baseline_single_leg"] = aggregate(single_leg_rewards)
    return summary


def scan_checkpoints(
    cfg: DictConfig,
    single_actor: model.TradingActor | None,
    device: torch.device,
    make_batches,
    entry_threshold: float,
    exit_threshold: float,
) -> list[tuple[str, dict[str, float]]]:
    """Score every checkpoint on the selection seed.

    Selection and the final number must not share a seed, or the reported
    result is the best of forty draws rather than an estimate of what the
    policy earns.
    """
    results = []
    for checkpoint in utils.sort_latest_checkpoints(cfg.general.experiment_dir):
        actor, checkpoint_cfg = load_pair_actor(checkpoint, device)
        summary = evaluate(actor, single_actor, checkpoint_cfg, make_batches(checkpoint_cfg),
                           entry_threshold, exit_threshold)
        results.append((checkpoint, summary["pair_policy"]))
        print(
            f"{checkpoint.rsplit('/', 1)[-1]:<18} return={summary['pair_policy']['return']:+.5f} "
            f"pf={summary['pair_policy']['profit_factor']:.4f} "
            f"spread={summary['pair_policy']['spread_fraction']:.3f} "
            f"dir={summary['pair_policy']['directional_fraction']:.3f}",
            flush=True,
        )
        del actor
        torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--scan_checkpoints", action="store_true",
                        help="score every checkpoint on --seed and report the best")
    parser.add_argument("--pair_checkpoint", default=None, help="defaults to the newest in the experiment dir")
    parser.add_argument("--single_checkpoint", default=None, help="single-symbol baseline to compare against")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=10_007)
    parser.add_argument("--entry_threshold", type=float, default=1.5)
    parser.add_argument("--exit_threshold", type=float, default=0.0)
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    device = torch.device(cfg.model.device)
    checkpoint = args.pair_checkpoint or utils.load_most_recent_checkpoint(cfg.general.experiment_dir)
    if not checkpoint:
        raise SystemExit(f"no pair checkpoint found in {cfg.general.experiment_dir}")
    pair_actor, checkpoint_cfg = load_pair_actor(checkpoint, device)
    if int(pair_actor.scalar_dim) != PAIR_SCALAR_DIM:
        raise SystemExit("checkpoint was trained with a different scalar layout")
    single_actor = load_single_actor(args.single_checkpoint, device) if args.single_checkpoint else None

    use_toy = bool(cfg.data.get("use_toy", False))

    def make_batches(active_cfg: DictConfig):
        if use_toy:
            return toy_batches(active_cfg, args.batch_size, args.batches, device, args.seed)
        split = OmegaConf.merge(cfg.val, {"batch_size": args.batch_size})
        return market_batches(cfg, split, device, args.batches)

    if args.scan_checkpoints:
        ranked = scan_checkpoints(
            cfg, single_actor, device, make_batches, args.entry_threshold, args.exit_threshold,
        )
        best = max(ranked, key=lambda row: row[1]["return"])
        print(f"\nbest on seed {args.seed}: {best[0]} return={best[1]['return']:+.5f}")
        checkpoint = best[0]
        pair_actor, checkpoint_cfg = load_pair_actor(checkpoint, device)

    batches = make_batches(checkpoint_cfg)

    summary = evaluate(
        pair_actor, single_actor, checkpoint_cfg, batches, args.entry_threshold, args.exit_threshold
    )
    report = {
        "pair_checkpoint": checkpoint,
        "single_checkpoint": args.single_checkpoint,
        "use_toy": use_toy,
        "seed": args.seed,
        "symbol_days": args.batch_size * args.batches,
        "strategies": summary,
    }
    print(json.dumps(report, indent=2))

    header = f"{'strategy':<22}" + "".join(f"{key[:12]:>14}" for key in REPORTED_METRICS)
    print("\n" + header)
    for name, metrics in summary.items():
        row = "".join(f"{metrics.get(key, float('nan')):>14.4f}" for key in REPORTED_METRICS)
        print(f"{name:<22}{row}")

    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
