import tempfile
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

import model
from dataset import (
    EXTENDED_SESSION_BARS,
    MarketDayDataset,
    OnlineBezierToyProvider,
    market_context_window_size,
)
from prepare_days import rows_for_file
from train import (
    FEATURE_DIM,
    build_market_features,
    collect_rollout,
    generalized_advantages,
    market_rewards,
    performance_metrics,
    ppo_update,
    regular_session_mask,
)


ANNO = "2010-01-01"


def et_seconds(ts: str) -> int:
    dt = pd.Timestamp(ts, tz="US/Eastern").tz_convert("UTC")
    return int((dt - pd.Timestamp(ANNO, tz="UTC")).total_seconds())


class ConstantActor(model.TradingActor):
    def __init__(self, window_size: int, action: int):
        super().__init__(window_size, hidden_dim=8, depth=1)
        self.constant_action = action

    @torch.no_grad()
    def play(self, inputs: torch.Tensor, sampling: str = "multinomial") -> torch.Tensor:
        return torch.full(inputs.shape[:-2], self.constant_action, dtype=torch.long, device=inputs.device)


class MarketPPOTest(unittest.TestCase):
    def test_ten_extended_sessions_produce_9601_tick_window(self):
        self.assertEqual(EXTENDED_SESSION_BARS, 960)
        self.assertEqual(market_context_window_size(10), 9601)

    def test_actions_move_bounded_inventory_one_step(self):
        previous = torch.tensor([0, 1, 1, 0, -1, -1, 1, -1])
        actions = torch.tensor(
            [
                model.BUY_ACTION,
                model.BUY_ACTION,
                model.SELL_ACTION,
                model.SELL_ACTION,
                model.SELL_ACTION,
                model.BUY_ACTION,
                model.NOTHING_ACTION,
                model.NOTHING_ACTION,
            ]
        )
        self.assertEqual(model.ACTION_DIM, 3)
        self.assertEqual(model.ACTION_NAMES, ("buy", "nothing", "sell"))
        self.assertEqual(model.apply_action(previous, actions).tolist(), [1, 1, 0, -1, -1, 0, 1, -1])

    def test_models_are_flat_window_mlps(self):
        actor = model.TradingActor(window_size=8, hidden_dim=16, depth=2)
        critic = model.TradingCritic(window_size=8, hidden_dim=16, depth=2)
        inputs = torch.randn(3, 4, 8, FEATURE_DIM)
        self.assertEqual(actor(inputs).shape, (3, 4, 3))
        self.assertEqual(critic(inputs).shape, (3, 4))
        self.assertFalse(any("transformer" in type(module).__name__.lower() for module in actor.modules()))

    def test_price_features_are_invariant_to_absolute_symbol_scale(self):
        prices = torch.tensor([[10.0, 10.1, 10.0, 10.2, 10.3, 10.4]])
        volumes = torch.tensor([[10.0, 20.0, 15.0, 30.0, 25.0, 40.0]])
        progress = torch.linspace(-1, 1, 6).reshape(1, -1)
        a = build_market_features(prices, volumes, progress, window_size=4, rollout_size=2)
        b = build_market_features(prices * 137.0, volumes, progress, window_size=4, rollout_size=2)
        self.assertTrue(torch.allclose(a, b, atol=2e-4))

    def test_online_bezier_provider_is_fresh_bounded_and_reproducible(self):
        provider = OnlineBezierToyProvider(window_size=8, rollout_size=4)
        first_generator = torch.Generator().manual_seed(123)
        repeated_generator = torch.Generator().manual_seed(123)
        first = provider.sample(32, "cpu", first_generator)
        repeated = provider.sample(32, "cpu", repeated_generator)
        fresh = provider.sample(32, "cpu")

        self.assertEqual(first.prices.shape, (32, 12))
        self.assertEqual(first.target_positions.shape, (32, 4))
        self.assertEqual(first.latent_returns.shape, (32, 11))
        self.assertTrue(first.prices.gt(0).all())
        self.assertTrue(first.anchor_counts.ge(3).logical_and(first.anchor_counts.le(4)).all())
        self.assertTrue(first.target_positions.ge(-1).logical_and(first.target_positions.le(1)).all())
        self.assertTrue(torch.equal(first.prices, repeated.prices))
        self.assertFalse(torch.equal(first.prices, fresh.prices))

    def test_bezier_noise_is_an_unpredictable_integrated_return_innovation(self):
        provider = OnlineBezierToyProvider(window_size=8, rollout_size=4, return_noise_std=3e-4)
        batch = provider.sample(2048, "cpu", torch.Generator().manual_seed(321))
        observed_returns = torch.log(batch.prices[:, 1:] / batch.prices[:, :-1])
        innovations = observed_returns - batch.latent_returns

        variance = innovations.square().mean()
        lag_one_correlation = (innovations[:, 1:] * innovations[:, :-1]).mean() / variance
        self.assertAlmostEqual(variance.sqrt().item(), 3e-4, delta=1e-5)
        self.assertLess(abs(lag_one_correlation.item()), 0.03)

        clean = OnlineBezierToyProvider(window_size=8, rollout_size=4, return_noise_std=0.0)
        clean_batch = clean.sample(32, "cpu", torch.Generator().manual_seed(123))
        clean_returns = torch.log(clean_batch.prices[:, 1:] / clean_batch.prices[:, :-1])
        self.assertTrue(torch.allclose(clean_returns, clean_batch.latent_returns, atol=2e-6))

    def test_rollout_is_autoregressive_over_selected_positions(self):
        # Buy opens a unit long. Repeating buy at the upper bound is a no-op,
        # and the held position appears at the newest tick of the next state.
        actor = ConstantActor(window_size=4, action=model.BUY_ACTION)
        prices = torch.arange(100.0, 106.0).reshape(1, -1)
        volumes = torch.ones_like(prices)
        progress = torch.linspace(-1, 1, 6).reshape(1, -1)
        rollout = collect_rollout(actor, prices, volumes, progress, rollout_size=2)
        self.assertTrue(rollout.states[:, 0, :, -1].eq(0).all())
        self.assertEqual(rollout.states[0, 1, -1, -1].item(), 1.0)
        self.assertEqual(rollout.positions.tolist(), [[1, 1]])
        self.assertEqual(rollout.trades.tolist(), [[True, False]])

    def test_rewards_charge_position_changes_and_force_close(self):
        positions = torch.tensor([[1, 0, 0], [-1, -1, -1]])
        now = torch.tensor([[100.0, 101.0, 102.0], [100.0, 99.0, 98.0]])
        nxt = torch.tensor([[101.0, 102.0, 103.0], [99.0, 98.0, 97.0]])
        rewards, info = market_rewards(positions, now, nxt, transaction_cost=0.001)
        self.assertGreater(rewards.sum().item(), 0.0)
        self.assertEqual(info["forced_closes"].tolist(), [False, True])
        self.assertEqual(info["end_positions"].tolist(), [0.0, 0.0])
        self.assertAlmostEqual(info["costs"][0].sum().item(), 0.002, places=6)

        flat_reward, _ = market_rewards(torch.zeros_like(positions), now, nxt, risk_penalty=0.1)
        exposed_reward, exposed_info = market_rewards(positions, now, nxt, risk_penalty=0.1)
        self.assertTrue(exposed_info["risk_costs"].ge(0).all())
        self.assertGreater(flat_reward.sum().item(), exposed_reward.sum().item())

    def test_performance_metrics_track_return_profit_factor_and_drawdown(self):
        rewards = torch.tensor([[0.10, -0.04, 0.02], [-0.02, 0.01, -0.03]])
        metrics = performance_metrics(rewards)
        self.assertAlmostEqual(metrics["return"], 0.02, places=6)
        # Rollouts net to +0.08 and -0.04 before the ratio is taken.
        self.assertAlmostEqual(metrics["profit_factor"], 0.08 / 0.04, places=5)
        self.assertAlmostEqual(metrics["profit_factor_timestep"], 0.13 / 0.09, places=5)
        self.assertAlmostEqual(metrics["max_drawdown"], 0.04, places=6)

    def test_profit_factor_names_its_degenerate_cases(self):
        # A policy that never traded risked nothing; reporting 0.0 would read as
        # a total loss, which is the opposite of the truth.
        flat = performance_metrics(torch.zeros(3, 4))
        self.assertTrue(math.isnan(flat["profit_factor"]))
        self.assertTrue(math.isnan(flat["profit_factor_timestep"]))
        self.assertEqual(flat["return"], 0.0)
        winners = performance_metrics(torch.tensor([[0.01, 0.02], [0.03, 0.01]]))
        self.assertEqual(winners["profit_factor"], float("inf"))
        losers = performance_metrics(torch.tensor([[-0.01, -0.02]]))
        self.assertEqual(losers["profit_factor"], 0.0)

    def test_profit_factor_is_pooled_not_averaged_over_batches(self):
        rewards = torch.tensor([[0.10, -0.04, 0.02], [-0.02, 0.01, -0.03]])
        pooled = performance_metrics(rewards)["profit_factor"]
        halves = [performance_metrics(rewards[i : i + 1])["profit_factor"] for i in (0, 1)]
        self.assertEqual(halves[0], float("inf"))
        self.assertNotAlmostEqual(pooled, halves[1], places=3)

    def test_dataset_returns_raw_window_without_symbol_or_absolute_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            secs = np.array([et_seconds(f"2025-01-02 09:{27 + i:02d}:00") for i in range(8)], dtype=np.int64)
            raw = np.stack((secs, np.arange(100000, 100800, 100), np.arange(1, 9), np.arange(8)), axis=1)
            np.save(path / "ABC.npy", raw)
            days = pd.DataFrame([{"sample_id": "ABC", "sod_idx": 3, "eod_idx": 7}])
            dataset = MarketDayDataset(days, str(path), window_size=4, rollout_size=2)
            item = dataset[0]
            self.assertEqual(set(item), {"_id", "prices", "volumes", "secs"})
            self.assertEqual(item["prices"].shape, (6,))
            self.assertAlmostEqual(item["prices"][0], 100.0)

    def test_full_session_dataset_filters_incomplete_days_and_starts_at_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            raw = np.stack(
                (
                    np.arange(12, dtype=np.int64) * 60,
                    np.arange(100000, 101200, 100),
                    np.arange(1, 13),
                ),
                axis=1,
            )
            np.save(path / "FULL.npy", raw)
            np.save(path / "SHORT.npy", raw)
            days = pd.DataFrame(
                [
                    {"sample_id": "FULL", "ctx_idx": 0, "sod_idx": 3, "eod_idx": 7},
                    {"sample_id": "SHORT", "ctx_idx": 0, "sod_idx": 3, "eod_idx": 6},
                ]
            )
            dataset = MarketDayDataset(
                days,
                str(path),
                window_size=4,
                rollout_size=4,
                should_augment=True,
                require_full_session=True,
            )
            self.assertEqual(len(dataset), 1)
            item = dataset[0]
            self.assertEqual(item["_id"], "FULL")
            self.assertEqual(item["prices"].shape, (8,))
            self.assertAlmostEqual(item["prices"][3], 100.3, places=4)
            self.assertAlmostEqual(item["prices"][-1], 100.7, places=4)

    def test_prepare_days_rejects_missing_and_count_preserving_tick_gaps(self):
        session = pd.date_range(
            "2025-01-02 09:30:00",
            "2025-01-02 16:00:00",
            freq="min",
            tz="US/Eastern",
        )
        session_secs = np.array(
            [int((ts.tz_convert("UTC") - pd.Timestamp(ANNO, tz="UTC")).total_seconds()) for ts in session],
            dtype=np.int64,
        )
        context_secs = session_secs[0] - np.arange(3, 0, -1, dtype=np.int64) * 60

        def write_bars(path: Path, regular_secs: np.ndarray) -> None:
            secs = np.concatenate((context_secs, regular_secs))
            raw = np.stack(
                (
                    secs,
                    np.arange(len(secs), dtype=np.int64) + 100_000,
                    np.ones(len(secs), dtype=np.int64),
                ),
                axis=1,
            )
            np.save(path, raw)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            full_path = path / "FULL.npy"
            sparse_path = path / "SPARSE.npy"
            malformed_path = path / "MALFORMED.npy"
            write_bars(full_path, session_secs)
            sparse_secs = np.delete(session_secs, np.arange(30, 51))
            self.assertEqual(sparse_secs.size, 370)
            write_bars(sparse_path, sparse_secs)
            malformed = np.sort(np.concatenate((np.delete(session_secs, 30), [session_secs[29] + 30])))
            write_bars(malformed_path, malformed)

            self.assertEqual(len(rows_for_file(full_path, ANNO, window_size=4, rollout_size=390)), 1)
            self.assertEqual(rows_for_file(sparse_path, ANNO, window_size=4, rollout_size=390), [])
            self.assertEqual(rows_for_file(malformed_path, ANNO, window_size=4, rollout_size=390), [])

    def test_full_session_loader_revalidates_one_minute_cadence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            secs = np.array([0, 60, 120, 180, 240, 330, 360, 420], dtype=np.int64)
            raw = np.stack(
                (secs, np.arange(100_000, 100_800, 100), np.ones(8, dtype=np.int64)), axis=1
            )
            np.save(path / "GAP.npy", raw)
            days = pd.DataFrame([{"sample_id": "GAP", "ctx_idx": 0, "sod_idx": 3, "eod_idx": 7}])
            dataset = MarketDayDataset(
                days,
                str(path),
                window_size=4,
                rollout_size=4,
                require_full_session=True,
            )
            with self.assertRaisesRegex(ValueError, "contiguous one-minute intervals"):
                dataset[0]

    def test_full_session_loader_completes_sparse_ticks_on_the_fly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            secs = np.array([0, 60, 120, 180, 240, 360, 420], dtype=np.int64)
            prices = np.array([100_000, 100_100, 100_200, 100_300, 100_400, 100_600, 100_700])
            volumes = np.arange(1, 8, dtype=np.int64)
            np.save(path / "SPARSE.npy", np.stack((secs, prices, volumes), axis=1))
            days = pd.DataFrame(
                [
                    {
                        "sample_id": "SPARSE",
                        "ctx_idx": 0,
                        "sod_idx": 3,
                        "eod_idx": 6,
                        "sod_sec": 180,
                        "eod_sec": 420,
                        "context_sod_sec": 180,
                        "context_eod_sec": 420,
                        "missing_ticks": 1,
                    }
                ]
            )
            calendar_days = pd.concat(
                (
                    pd.DataFrame(
                        [
                            {
                                "sample_id": "SPARSE",
                                "ctx_idx": 0,
                                "sod_idx": 0,
                                "eod_idx": 2,
                                "sod_sec": 0,
                                "eod_sec": 120,
                                "context_sod_sec": 0,
                                "context_eod_sec": 120,
                                "missing_ticks": 0,
                            }
                        ]
                    ),
                    days,
                ),
                ignore_index=True,
            )
            dataset = MarketDayDataset(
                days,
                str(path),
                window_size=4,
                rollout_size=4,
                require_full_session=True,
                calendar_days=calendar_days,
            )
            item = dataset[0]

            self.assertEqual(item["secs"].tolist(), [0, 60, 120, 180, 240, 300, 360, 420])
            self.assertAlmostEqual(item["prices"][5], 100.4, places=4)
            self.assertEqual(item["volumes"][5], 0.0)
            self.assertEqual(item["volumes"][6], 6.0)

    def test_completely_missing_working_day_is_context_but_not_a_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            secs = np.array([0, 60, 120, 360, 420, 540, 600], dtype=np.int64)
            prices = np.array([100_000, 100_100, 100_200, 100_600, 100_700, 100_900, 101_000])
            np.save(path / "MISSING_DAY.npy", np.stack((secs, prices, np.ones(7)), axis=1))
            calendar_days = pd.DataFrame(
                [
                    {
                        "sample_id": "MISSING_DAY",
                        "sod_idx": 0,
                        "eod_idx": 2,
                        "sod_sec": 0,
                        "eod_sec": 120,
                        "context_sod_sec": 0,
                        "context_eod_sec": 120,
                        "is_tradable": True,
                    },
                    {
                        "sample_id": "MISSING_DAY",
                        "sod_idx": -1,
                        "eod_idx": -1,
                        "sod_sec": 180,
                        "eod_sec": 300,
                        "context_sod_sec": 180,
                        "context_eod_sec": 300,
                        "is_tradable": False,
                    },
                    {
                        "sample_id": "MISSING_DAY",
                        "sod_idx": 3,
                        "eod_idx": 6,
                        "sod_sec": 360,
                        "eod_sec": 600,
                        "context_sod_sec": 360,
                        "context_eod_sec": 600,
                        "is_tradable": True,
                    },
                ]
            )
            target = calendar_days.iloc[[2]]
            dataset = MarketDayDataset(
                target,
                str(path),
                window_size=4,
                rollout_size=4,
                require_full_session=True,
                calendar_days=calendar_days,
            )
            item = dataset[0]

            self.assertEqual(item["secs"].tolist(), [180, 240, 300, 360, 420, 480, 540, 600])
            self.assertEqual(item["volumes"][:3].tolist(), [0.0, 0.0, 0.0])
            self.assertTrue(np.allclose(item["prices"][:3], 100.2))

    def test_market_hours(self):
        secs = torch.tensor(
            [
                et_seconds("2025-01-02 09:29:00"),
                et_seconds("2025-01-02 09:30:00"),
                et_seconds("2025-01-02 16:00:00"),
                et_seconds("2025-01-02 16:01:00"),
            ]
        )
        self.assertEqual(regular_session_mask(secs, ANNO).tolist(), [False, True, True, False])

    def test_ppo_update_cpu_sanity(self):
        torch.manual_seed(0)
        actor = model.TradingActor(window_size=4, hidden_dim=16, depth=1)
        critic = model.TradingCritic(window_size=4, hidden_dim=16, depth=1)
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
        prices = torch.tensor([[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]]).repeat(2, 1)
        volumes = torch.ones_like(prices)
        progress = torch.linspace(-1, 1, 6).repeat(2, 1)
        rollout = collect_rollout(actor, prices, volumes, progress, rollout_size=2)
        with torch.no_grad():
            old_dist = actor.distribution(rollout.states)
            old_logprobs = old_dist.log_prob(rollout.actions)
            old_values = critic(rollout.states)
            advantages, returns = generalized_advantages(rollout.rewards, old_values, 0.99, 0.95)
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        cfg = OmegaConf.create(
            {
                "model": {
                    "loss_weights": {"policy": 1.0, "value": 0.5, "entropy": 0.01},
                    "ppo_clip": 0.2,
                    "ppo_value_clip": 0.2,
                    "max_grad_norm": 1.0,
                },
                "train": {"ppo_epochs": 1},
            }
        )
        before = [p.detach().clone() for p in actor.parameters()]
        losses = ppo_update(actor, critic, optimizer, rollout, old_logprobs, old_values, returns, advantages, cfg)
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
        self.assertLess(losses["loss_entropy"], 0.0)
        self.assertAlmostEqual(losses["loss_entropy"], -losses["entropy"], places=6)
        self.assertLessEqual(losses["entropy"], np.log(model.ACTION_DIM) + 1e-6)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, actor.parameters())))


if __name__ == "__main__":
    unittest.main()
