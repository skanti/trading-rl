import tempfile
import unittest
from pathlib import Path

import numpy as np
from scripts.tests.bar_fixtures import ohlcv_fixture
import pandas as pd
import torch
from omegaconf import OmegaConf

from ml import model
from ml.dataset import OnlinePairToyProvider, sample_ornstein_uhlenbeck
from ml.pair import (
    PAIR_FEATURE_DIM,
    PAIR_SCALAR_DIM,
    build_pair_features,
    collect_pair_rollout,
    pair_market_rewards,
    pair_train_step,
    pair_window_statistics,
)
from ml.pair_dataset import MarketPairDataset, make_pair_dataloader
from ml.pair_eval import step_limited_positions, zscore_rule_positions
from ml.prepare_pairs import scan_symbol, session_table


ANNO = "2010-01-01"
RTH_INTERVALS = 390


def et_seconds(ts: str) -> int:
    dt = pd.Timestamp(ts, tz="US/Eastern").tz_convert("UTC")
    return int((dt - pd.Timestamp(ANNO, tz="UTC")).total_seconds())


class ConstantPairActor(model.PairTradingActor):
    def __init__(self, window_size: int, action: int):
        super().__init__(window_size, PAIR_FEATURE_DIM, hidden_dim=8, depth=1, scalar_dim=PAIR_SCALAR_DIM)
        self.constant_action = action

    @torch.no_grad()
    def play(self, inputs, scalars=None, sampling: str = "multinomial") -> torch.Tensor:
        return torch.full(inputs.shape[:-2], self.constant_action, dtype=torch.long, device=inputs.device)


class PairActionTest(unittest.TestCase):
    def test_joint_action_decodes_to_two_independent_leg_commands(self):
        self.assertEqual(model.PAIR_ACTION_DIM, 9)
        actions = torch.arange(9)
        command_a, command_b = model.decode_pair_action(actions)
        self.assertEqual(command_a.tolist(), [0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertEqual(command_b.tolist(), [0, 1, 2, 0, 1, 2, 0, 1, 2])
        self.assertTrue(torch.equal(model.encode_pair_action(command_a, command_b), actions))

    def test_joint_action_spans_flat_single_leg_spread_and_directional(self):
        flat = torch.zeros(1, dtype=torch.long)
        buy_sell = model.encode_pair_action(
            torch.tensor([model.BUY_ACTION]), torch.tensor([model.SELL_ACTION])
        )
        buy_buy = model.encode_pair_action(
            torch.tensor([model.BUY_ACTION]), torch.tensor([model.BUY_ACTION])
        )
        buy_hold = model.encode_pair_action(
            torch.tensor([model.BUY_ACTION]), torch.tensor([model.NOTHING_ACTION])
        )
        for action, expected in ((buy_sell, (1, -1)), (buy_buy, (1, 1)), (buy_hold, (1, 0))):
            position_a, position_b = model.apply_pair_action(flat, flat, action)
            self.assertEqual((int(position_a), int(position_b)), expected)

    def test_reversal_still_costs_two_commands_on_each_leg(self):
        position_a = torch.tensor([1])
        position_b = torch.tensor([-1])
        action = model.encode_pair_action(
            torch.tensor([model.SELL_ACTION]), torch.tensor([model.BUY_ACTION])
        )
        position_a, position_b = model.apply_pair_action(position_a, position_b, action)
        self.assertEqual((int(position_a), int(position_b)), (0, 0))
        position_a, position_b = model.apply_pair_action(position_a, position_b, action)
        self.assertEqual((int(position_a), int(position_b)), (-1, 1))

    def test_rejects_out_of_range_joint_actions(self):
        with self.assertRaises(ValueError):
            model.decode_pair_action(torch.tensor([9]))


class PairModelTest(unittest.TestCase):
    def test_pair_models_accept_window_and_scalar_inputs(self):
        actor = model.PairTradingActor(8, PAIR_FEATURE_DIM, 16, 2, PAIR_SCALAR_DIM)
        critic = model.PairTradingCritic(8, PAIR_FEATURE_DIM, 16, 2, PAIR_SCALAR_DIM)
        inputs = torch.randn(3, 4, 8, PAIR_FEATURE_DIM)
        scalars = torch.randn(3, 4, PAIR_SCALAR_DIM)
        self.assertEqual(actor(inputs, scalars).shape, (3, 4, 9))
        self.assertEqual(critic(inputs, scalars).shape, (3, 4))

    def test_scalar_side_input_is_required_and_shape_checked(self):
        actor = model.PairTradingActor(8, PAIR_FEATURE_DIM, 16, 2, PAIR_SCALAR_DIM)
        inputs = torch.randn(2, 8, PAIR_FEATURE_DIM)
        with self.assertRaises(ValueError):
            actor(inputs)
        with self.assertRaises(ValueError):
            actor(inputs, torch.randn(2, PAIR_SCALAR_DIM + 1))

    def test_single_symbol_models_still_reject_scalars(self):
        actor = model.TradingActor(window_size=8, hidden_dim=16, depth=2)
        self.assertEqual(actor.scalar_dim, 0)
        with self.assertRaises(ValueError):
            actor.forward_features(torch.randn(2, 8, 5), torch.randn(2, 1))


class OrnsteinUhlenbeckTest(unittest.TestCase):
    def test_blocked_scan_matches_the_naive_recursion(self):
        batch, ticks, block = 4, 200, 32
        decay = torch.tensor([0.9, 0.97, 0.999, 1.0])
        step_std = torch.full((batch,), 1e-3)
        initial = torch.tensor([1e-3, -2e-3, 5e-4, 0.0])
        fast = sample_ornstein_uhlenbeck(
            batch, ticks, decay, step_std, initial, torch.device("cpu"),
            torch.Generator().manual_seed(11), block=block,
        )
        # Rebuild the innovation stream the scan consumed, then iterate.
        generator = torch.Generator().manual_seed(11)
        padded = ticks + (-ticks) % block
        innovations = torch.randn(batch, padded, generator=generator) * step_std.unsqueeze(-1)
        state = initial.clone()
        expected = []
        for i in range(ticks):
            state = decay * state + innovations[:, i]
            expected.append(state)
        self.assertTrue(torch.allclose(fast, torch.stack(expected, dim=1), atol=1e-6))

    def test_unit_decay_produces_a_random_walk(self):
        batch, ticks = 6, 500
        path = sample_ornstein_uhlenbeck(
            batch, ticks, torch.ones(batch), torch.full((batch,), 1e-3), torch.zeros(batch),
            torch.device("cpu"), torch.Generator().manual_seed(3),
        )
        # A random walk's variance grows with time; an OU's does not.
        early = path[:, :100].var().item()
        late = path[:, -100:].var().item()
        self.assertGreater(late, 2.0 * early)

    def test_rejects_invalid_decays(self):
        with self.assertRaises(ValueError):
            sample_ornstein_uhlenbeck(
                2, 10, torch.tensor([1.2, 0.9]), torch.ones(2), torch.zeros(2), torch.device("cpu")
            )


class PairToyProviderTest(unittest.TestCase):
    def test_provider_is_reproducible_and_produces_valid_prices(self):
        provider = OnlinePairToyProvider(64, 16)
        first = provider.sample(4, "cpu", torch.Generator().manual_seed(5))
        second = provider.sample(4, "cpu", torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(first.prices_a, second.prices_a))
        self.assertTrue(torch.equal(first.prices_b, second.prices_b))
        for prices in (first.prices_a, first.prices_b):
            self.assertTrue(torch.isfinite(prices).all())
            self.assertTrue((prices > 0).all())
        self.assertEqual(first.prices_a.shape, (4, provider.ticks))
        self.assertEqual(first.target_positions_a.shape, (4, provider.rollout_size))

    def test_legs_share_a_common_factor(self):
        provider = OnlinePairToyProvider(2048, 64)
        batch = provider.sample(24, "cpu", torch.Generator().manual_seed(9))
        returns_a = batch.prices_a.log().diff(dim=1)
        returns_b = batch.prices_b.log().diff(dim=1)
        correlations = torch.stack(
            [torch.corrcoef(torch.stack((returns_a[i], returns_b[i])))[0, 1] for i in range(24)]
        )
        self.assertGreater(correlations.mean().item(), 0.5)

    def test_broken_pairs_are_generated_and_do_not_mean_revert(self):
        provider = OnlinePairToyProvider(2048, 64, broken_fraction=0.5)
        batch = provider.sample(64, "cpu", torch.Generator().manual_seed(13))
        self.assertTrue(batch.is_mean_reverting.any())
        self.assertTrue((~batch.is_mean_reverting).any())
        # Broken pairs are random walks, so their decay is exactly one.
        self.assertTrue(torch.equal(batch.spread_decay[~batch.is_mean_reverting].unique(), torch.tensor([1.0])))
        self.assertTrue((batch.spread_decay[batch.is_mean_reverting] < 1.0).all())

    def test_absolute_price_levels_are_independent_between_legs(self):
        provider = OnlinePairToyProvider(64, 16)
        batch = provider.sample(64, "cpu", torch.Generator().manual_seed(21))
        ratio = (batch.prices_a[:, 0] / batch.prices_b[:, 0]).log()
        self.assertGreater(ratio.std().item(), 1.0)


class PairFeatureTest(unittest.TestCase):
    def setUp(self):
        self.provider = OnlinePairToyProvider(1024, 32)
        self.batch = self.provider.sample(16, "cpu", torch.Generator().manual_seed(17))

    def test_window_statistics_recover_the_generating_hedge_ratio(self):
        market, scalars, hedge = build_pair_features(
            self.batch.prices_a, self.batch.prices_b, self.batch.volumes_a,
            self.batch.volumes_b, self.batch.progress, 1024, 32,
        )
        error = (hedge[:, 0] - self.batch.beta).abs().median().item()
        self.assertLess(error, 0.25)

    def test_residual_tracks_the_true_spread(self):
        _, _, hedge = build_pair_features(
            self.batch.prices_a, self.batch.prices_b, self.batch.volumes_a,
            self.batch.volumes_b, self.batch.progress, 1024, 32,
        )
        residual = hedge[:, :1] * self.batch.prices_a[:, :1024].log() - self.batch.prices_b[:, :1024].log()
        spread = self.batch.spread[:, :1024]
        correlations = torch.stack(
            [torch.corrcoef(torch.stack((residual[i], spread[i])))[0, 1] for i in range(16)]
        )
        self.assertGreater(correlations.median().item(), 0.5)

    def test_features_are_invariant_to_each_leg_s_absolute_price(self):
        """Rescaling either leg must not change what the policy observes.

        Agreement is checked against each channel's own dynamic range rather
        than with a flat tolerance: the channels differ in scale by orders of
        magnitude, and what matters is that no channel shifts by enough to
        carry information about the price level. Exact equality is not
        available in float32, since rescaling changes the rounding of every
        logarithm, so the bar is that any residual drift stays two hundred
        times smaller than the signal the channel carries.
        """
        tolerance = 5e-3
        args = (self.batch.volumes_a, self.batch.volumes_b, self.batch.progress, 1024, 32)
        base, base_scalars, base_hedge = build_pair_features(
            self.batch.prices_a, self.batch.prices_b, *args
        )
        scaled, scaled_scalars, scaled_hedge = build_pair_features(
            self.batch.prices_a * 137.0, self.batch.prices_b * 0.05, *args
        )
        for channel in range(base.shape[-1]):
            spread = base[..., channel].abs().max().item()
            drift = (base[..., channel] - scaled[..., channel]).abs().max().item()
            self.assertLess(drift, tolerance * max(spread, 1e-6), f"channel {channel} moved with price")
        for channel in range(base_scalars.shape[-1]):
            spread = base_scalars[..., channel].abs().max().item()
            drift = (base_scalars[..., channel] - scaled_scalars[..., channel]).abs().max().item()
            self.assertLess(drift, tolerance * max(spread, 1e-6), f"scalar {channel} moved with price")
        self.assertLess((base_hedge - scaled_hedge).abs().max().item(), 1e-3)

    def test_feature_layout_matches_the_declared_names(self):
        market, scalars, _ = build_pair_features(
            self.batch.prices_a, self.batch.prices_b, self.batch.volumes_a,
            self.batch.volumes_b, self.batch.progress, 1024, 32,
        )
        # Nine market channels; the two position channels are appended online.
        self.assertEqual(market.shape[-1], PAIR_FEATURE_DIM - 2)
        self.assertEqual(scalars.shape[-1], PAIR_SCALAR_DIM)

    def test_statistics_use_only_the_observation_window(self):
        window, steps = 1024, 32
        truncated = self.batch.prices_a.clone()
        # Corrupting ticks after the last decision must not move any statistic.
        truncated[:, window + steps - 1 :] *= 3.0
        first = build_pair_features(
            self.batch.prices_a, self.batch.prices_b, self.batch.volumes_a,
            self.batch.volumes_b, self.batch.progress, window, steps,
        )[1]
        second = build_pair_features(
            truncated, self.batch.prices_b, self.batch.volumes_a,
            self.batch.volumes_b, self.batch.progress, window, steps,
        )[1]
        self.assertTrue(torch.equal(first, second))

    def test_no_decision_can_see_its_own_future(self):
        """Perturbing tick ``k`` must leave every decision before it untouched.

        This is the property the whole backtest rests on. The observation for
        decision ``t`` covers ticks ``[t, t + window - 1]`` and its reward comes
        from the move into tick ``t + window``, so corrupting from ``t + window``
        onwards must change nothing at or before ``t``.
        """
        window, steps = 256, 12
        provider = OnlinePairToyProvider(window, steps)
        batch = provider.sample(3, "cpu", torch.Generator().manual_seed(29))
        args = (batch.volumes_a, batch.volumes_b, batch.progress, window, steps)
        base_market, base_scalars, base_hedge = build_pair_features(
            batch.prices_a, batch.prices_b, *args
        )
        for decision in (0, 1, steps // 2, steps - 1):
            corrupted_a = batch.prices_a.clone()
            corrupted_b = batch.prices_b.clone()
            corrupted_a[:, window + decision :] *= 1.5
            corrupted_b[:, window + decision :] *= 0.7
            market, scalars, hedge = build_pair_features(corrupted_a, corrupted_b, *args)
            upto = slice(None, decision + 1)
            self.assertTrue(
                torch.equal(market[:, upto], base_market[:, upto]),
                f"decision {decision} observed a later tick",
            )
            self.assertTrue(torch.equal(scalars[:, upto], base_scalars[:, upto]))
            self.assertTrue(torch.equal(hedge[:, upto], base_hedge[:, upto]))
            # The perturbation has to actually reach later decisions, or the
            # test would pass on a feature builder that ignores prices entirely.
            if decision + 1 < steps:
                self.assertFalse(torch.equal(market[:, decision + 1 :], base_market[:, decision + 1 :]))

    def test_hedge_ratio_of_an_exactly_proportional_pair_is_the_slope(self):
        ticks = 64
        log_a = torch.linspace(0.0, 0.2, ticks).reshape(1, 1, -1) + torch.randn(1, 1, ticks) * 1e-3
        log_b = 1.5 * log_a
        returns_a = log_a.diff(dim=-1)
        returns_b = log_b.diff(dim=-1)
        stats = pair_window_statistics(
            torch.nn.functional.pad(returns_a, (1, 0)),
            torch.nn.functional.pad(returns_b, (1, 0)),
            log_a,
            log_b,
        )
        self.assertAlmostEqual(stats.hedge_ratio.item(), 1.5, places=4)
        self.assertLess(stats.residual_scale.item(), 1e-5)


class PairRewardTest(unittest.TestCase):
    def _prices(self, batch, steps):
        return torch.ones(batch, steps), torch.ones(batch, steps)

    def test_turnover_is_charged_on_both_legs_and_liquidation_is_forced(self):
        positions_a = torch.tensor([[1, 1, 1]])
        positions_b = torch.tensor([[-1, -1, -1]])
        now_a, next_a = self._prices(1, 3)
        rewards, info = pair_market_rewards(
            positions_a, positions_b, now_a, next_a, now_a, next_a,
            hedge_ratio=torch.ones(1, 3), transaction_cost=1e-3,
        )
        # Opening both legs costs two units of turnover, closing costs two more.
        self.assertAlmostEqual(info["costs"][0, 0].item(), 2e-3, places=9)
        self.assertAlmostEqual(info["costs"][0, 1].item(), 0.0, places=9)
        self.assertAlmostEqual(info["costs"][0, 2].item(), 2e-3, places=9)
        self.assertAlmostEqual(rewards.sum().item(), -4e-3, places=9)

    def test_net_penalty_is_zero_when_hedged_and_largest_when_doubled_up(self):
        now, nxt = self._prices(1, 1)
        def risk(position_a, position_b, hedge):
            _, info = pair_market_rewards(
                torch.tensor([[position_a]]), torch.tensor([[position_b]]),
                now, nxt, now, nxt, hedge_ratio=torch.tensor([[hedge]]),
                net_risk_penalty=1.0,
            )
            return info["risk_costs"][0, 0].item()

        self.assertAlmostEqual(risk(1, -1, 1.0), 0.0, places=9)
        self.assertAlmostEqual(risk(1, 1, 1.0), 4.0, places=9)
        self.assertAlmostEqual(risk(1, 0, 1.0), 1.0, places=9)
        # An unequal hedge ratio leaves residual exposure on a unit spread.
        self.assertAlmostEqual(risk(1, -1, 0.6), (1 - 0.6) ** 2, places=6)

    def test_gross_penalty_prices_deployed_capital(self):
        now, nxt = self._prices(1, 1)
        _, info = pair_market_rewards(
            torch.tensor([[1]]), torch.tensor([[-1]]), now, nxt, now, nxt,
            hedge_ratio=torch.ones(1, 1), gross_risk_penalty=1.0,
        )
        self.assertAlmostEqual(info["risk_costs"][0, 0].item(), 1.0, places=9)

    def test_both_legs_are_marked_to_market(self):
        rewards, _ = pair_market_rewards(
            torch.tensor([[1]]), torch.tensor([[-1]]),
            torch.tensor([[100.0]]), torch.tensor([[101.0]]),
            torch.tensor([[50.0]]), torch.tensor([[50.5]]),
            hedge_ratio=torch.ones(1, 1),
        )
        # Long A gains 1%, short B loses 1%; the spread nets to zero.
        self.assertAlmostEqual(rewards[0, 0].item(), 0.0, places=6)

    def test_pnl_splits_exactly_into_basket_and_spread(self):
        """The attribution must be an identity, not an approximation.

        With ``exposure_a = basket + spread`` and ``exposure_b = basket -
        spread``, the marked-to-market P&L is exactly
        ``basket * (r_a + r_b) + spread * (r_a - r_b)``. That identity is what
        makes ``spread_pnl`` a trustworthy answer to how much of the return came
        from the relationship between the legs rather than their direction.
        """
        torch.manual_seed(3)
        positions_a = torch.randint(-1, 2, (4, 20))
        positions_b = torch.randint(-1, 2, (4, 20))
        now_a = 100.0 + torch.randn(4, 20)
        next_a = now_a + torch.randn(4, 20)
        now_b = 50.0 + torch.randn(4, 20)
        next_b = now_b + torch.randn(4, 20)
        _, info = pair_market_rewards(
            positions_a, positions_b, now_a, next_a, now_b, next_b,
            hedge_ratio=torch.ones(4, 20),
        )
        gross = info["exposure_a"] * info["returns_a"] + info["exposure_b"] * info["returns_b"]
        self.assertTrue(torch.allclose(gross, info["basket_pnl"] + info["spread_pnl"], atol=1e-6))

    def test_a_hedged_spread_earns_only_spread_pnl(self):
        _, info = pair_market_rewards(
            torch.tensor([[1]]), torch.tensor([[-1]]),
            torch.tensor([[100.0]]), torch.tensor([[101.0]]),
            torch.tensor([[50.0]]), torch.tensor([[50.1]]),
            hedge_ratio=torch.ones(1, 1),
        )
        # Long one unit of A against one short unit of B earns the full
        # difference of the two returns, not half of it.
        self.assertAlmostEqual(info["basket_pnl"].item(), 0.0, places=6)
        self.assertAlmostEqual(info["spread_pnl"].item(), 0.01 - 0.002, places=6)

    def test_a_doubled_up_bet_earns_only_basket_pnl(self):
        _, info = pair_market_rewards(
            torch.tensor([[1]]), torch.tensor([[1]]),
            torch.tensor([[100.0]]), torch.tensor([[101.0]]),
            torch.tensor([[50.0]]), torch.tensor([[50.1]]),
            hedge_ratio=torch.ones(1, 1),
        )
        self.assertAlmostEqual(info["spread_pnl"].item(), 0.0, places=6)
        self.assertAlmostEqual(info["basket_pnl"].item(), 0.01 + 0.002, places=6)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            pair_market_rewards(
                torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 2, dtype=torch.long),
                *self._prices(1, 3), *self._prices(1, 3), hedge_ratio=torch.ones(1, 3),
            )


class PairRolloutTest(unittest.TestCase):
    def test_rollout_feeds_both_legs_positions_into_later_observations(self):
        window, steps, batch = 16, 5, 2
        provider = OnlinePairToyProvider(window, steps)
        sample = provider.sample(batch, "cpu", torch.Generator().manual_seed(4))
        buy_sell = int(
            model.encode_pair_action(
                torch.tensor([model.BUY_ACTION]), torch.tensor([model.SELL_ACTION])
            )
        )
        actor = ConstantPairActor(window, buy_sell)
        rollout = collect_pair_rollout(
            actor, sample.prices_a, sample.prices_b, sample.volumes_a,
            sample.volumes_b, sample.progress, steps,
        )
        self.assertTrue(torch.equal(rollout.positions_a, torch.ones(batch, steps, dtype=torch.long)))
        self.assertTrue(torch.equal(rollout.positions_b, -torch.ones(batch, steps, dtype=torch.long)))
        # The last two feature channels carry each leg's realized position.
        for t in range(1, steps):
            self.assertAlmostEqual(rollout.states[0, t, -1, -2].item(), 1.0, places=6)
            self.assertAlmostEqual(rollout.states[0, t, -1, -1].item(), -1.0, places=6)
        self.assertAlmostEqual(rollout.states[0, 0, -1, -2].item(), 0.0, places=6)

    def test_ppo_update_changes_parameters_on_cpu(self):
        window, steps = 32, 8
        provider = OnlinePairToyProvider(window, steps)
        sample = provider.sample(4, "cpu", torch.Generator().manual_seed(6))
        actor = model.PairTradingActor(window, PAIR_FEATURE_DIM, 16, 2, PAIR_SCALAR_DIM)
        critic = model.PairTradingCritic(window, PAIR_FEATURE_DIM, 16, 2, PAIR_SCALAR_DIM)
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
        cfg = OmegaConf.create(
            {
                "model": {
                    "gamma": 0.99, "gae_lambda": 0.95, "transaction_cost": 1e-4,
                    "net_risk_penalty": 1e-5, "gross_risk_penalty": 1e-6,
                    "ppo_clip": 0.2, "ppo_value_clip": 0.2, "max_grad_norm": 1.0,
                    "loss_weights": {"policy": 1.0, "value": 0.5, "entropy": 0.01},
                },
                "data": {"rollout_size": steps, "price_feature_scale": 100.0},
                "train": {"ppo_epochs": 2},
            }
        )
        before = actor.main[0].weight.detach().clone()
        rollout, losses = pair_train_step(
            actor, critic, optimizer, sample.prices_a, sample.prices_b,
            sample.volumes_a, sample.volumes_b, sample.progress, cfg,
        )
        self.assertFalse(torch.equal(before, actor.main[0].weight))
        self.assertTrue(np.isfinite(losses["loss"]))
        self.assertEqual(rollout.actions.shape, (4, steps))
        self.assertTrue((rollout.actions >= 0).all() and (rollout.actions < 9).all())


class PairBenchmarkTest(unittest.TestCase):
    def test_step_limiting_never_moves_more_than_one_unit_per_tick(self):
        targets = torch.tensor([[1, -1, -1, 1, 0, -1]])
        reachable = step_limited_positions(targets)
        deltas = reachable.diff(dim=1).abs()
        self.assertTrue((deltas <= 1).all())
        self.assertTrue((reachable.abs() <= 1).all())

    def test_zscore_rule_fades_a_stretched_residual(self):
        residual = torch.tensor([[0.0, 2.0, 2.0, 2.0, 0.0, -2.0, -2.0, -2.0]])
        positions_a, positions_b = zscore_rule_positions(residual, 1.5, 0.3)
        # A rich leg A is sold and the hedge leg is bought, and vice versa.
        self.assertEqual(positions_a[0, 3].item(), -1)
        self.assertEqual(positions_b[0, 3].item(), 1)
        self.assertEqual(positions_a[0, 4].item(), 0)
        self.assertEqual(positions_a[0, 7].item(), 1)
        self.assertEqual(positions_b[0, 7].item(), -1)

    def test_zscore_rule_rejects_inverted_thresholds(self):
        with self.assertRaises(ValueError):
            zscore_rule_positions(torch.zeros(1, 4), 0.3, 1.5)


class MarketPairDataTest(unittest.TestCase):
    def _write_symbol(self, directory: Path, name: str, seconds: np.ndarray, base: float) -> None:
        prices = (base * (1.0 + 1e-4 * np.arange(seconds.size))) * 1000.0
        volumes = np.full(seconds.size, 500.0)
        np.save(directory / f"{name}.npy", ohlcv_fixture(np.stack((seconds, prices, volumes), axis=1)))

    def _session_seconds(self, day: str) -> np.ndarray:
        start = et_seconds(f"{day} 09:30:00")
        return start + 60 * np.arange(RTH_INTERVALS + 1)

    def test_join_drops_ticks_only_one_leg_has(self):
        window, rollout = 8, RTH_INTERVALS
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            session = self._session_seconds("2024-03-05")
            previous = self._session_seconds("2024-03-04")
            # Leg A also carries pre-market ticks that leg B does not have.
            premarket = et_seconds("2024-03-05 08:00:00") + 60 * np.arange(20)
            self._write_symbol(directory, "AAA", np.concatenate((previous, premarket, session)), 100.0)
            self._write_symbol(directory, "BBB", np.concatenate((previous, session)), 40.0)

            pairs = pd.DataFrame(
                [("AAA", "BBB", "2024-03-05", int(session[0]), int(session[-1]), 0.9, "correlated")],
                columns=["sample_id_a", "sample_id_b", "date", "sod_sec", "eod_sec",
                         "context_correlation", "selection"],
            )
            dataset = MarketPairDataset(pairs, str(directory), window, rollout, anno=ANNO)
            item = dataset[0]
            self.assertEqual(item["prices_a"].shape, (window + rollout,))
            self.assertEqual(item["prices_b"].shape, (window + rollout,))
            np.testing.assert_array_equal(item["volumes_a"], np.full(window + rollout, 500.0))
            np.testing.assert_array_equal(item["volumes_b"], np.full(window + rollout, 500.0))
            # Every returned tick exists in both legs, so no pre-market survives.
            self.assertEqual(np.intersect1d(item["secs"], premarket).size, 0)
            self.assertEqual(item["secs"][window - 1], session[0])
            self.assertEqual(item["secs"][-1], session[-1])
            self.assertTrue(np.all(np.diff(item["secs"][window - 1 :]) == 60))
            self.assertNotIn("AAA", str(list(item.keys())))

    def test_missing_shared_history_is_rejected(self):
        window, rollout = 64, RTH_INTERVALS
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            session = self._session_seconds("2024-03-05")
            self._write_symbol(directory, "AAA", session, 100.0)
            self._write_symbol(directory, "BBB", session, 40.0)
            pairs = pd.DataFrame(
                [("AAA", "BBB", "2024-03-05", int(session[0]), int(session[-1]), 0.9, "correlated")],
                columns=["sample_id_a", "sample_id_b", "date", "sod_sec", "eod_sec",
                         "context_correlation", "selection"],
            )
            dataset = MarketPairDataset(pairs, str(directory), window, rollout, anno=ANNO)
            with self.assertRaises(ValueError):
                dataset[0]

    def test_splits_share_no_symbol_and_no_date(self):
        window, rollout = 8, RTH_INTERVALS
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            days = ["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07"]
            grid = np.concatenate([self._session_seconds(day) for day in days])
            names = [f"S{i}" for i in range(6)]
            for index, name in enumerate(names):
                self._write_symbol(directory, name, grid, 20.0 + index)

            rows = []
            for day in days[1:]:
                session = self._session_seconds(day)
                for universe, legs in (("train", names[:4]), ("holdout", names[4:])):
                    for i in range(len(legs)):
                        for j in range(i + 1, len(legs)):
                            rows.append((legs[i], legs[j], day, int(session[0]), int(session[-1]),
                                         0.9, "correlated", universe))
            pairs_path = directory / "pairs.csv"
            pd.DataFrame(rows, columns=[
                "sample_id_a", "sample_id_b", "date", "sod_sec", "eod_sec",
                "context_correlation", "selection", "universe",
            ]).to_csv(pairs_path, index=False)

            cfg_data = OmegaConf.create({
                "pairs_path": str(pairs_path), "data_dir": str(directory),
                "window_size": window, "rollout_size": rollout,
                "date_val": "2024-03-06", "anno": ANNO,
            })
            loaded = {}
            for split in ("train", "val"):
                cfg_split = OmegaConf.create({"split": split, "batch_size": 1, "workers_num": 0})
                frame = make_pair_dataloader(cfg_data, cfg_split, 0).dataset.pairs
                loaded[split] = frame
                self.assertGreater(len(frame), 0)

            train_symbols = set(loaded["train"].sample_id_a) | set(loaded["train"].sample_id_b)
            val_symbols = set(loaded["val"].sample_id_a) | set(loaded["val"].sample_id_b)
            self.assertEqual(train_symbols & val_symbols, set())
            self.assertLess(loaded["train"].date.max(), loaded["val"].date.min())
            self.assertEqual(set(loaded["train"].universe), {"train"})
            self.assertEqual(set(loaded["val"].universe), {"holdout"})

    def test_pairs_file_without_a_universe_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs_path = Path(tmp) / "pairs.csv"
            pd.DataFrame([("A", "B", "2024-03-05", 1, 2, 0.9, "correlated")], columns=[
                "sample_id_a", "sample_id_b", "date", "sod_sec", "eod_sec",
                "context_correlation", "selection",
            ]).to_csv(pairs_path, index=False)
            cfg_data = OmegaConf.create({
                "pairs_path": str(pairs_path), "data_dir": tmp, "window_size": 8,
                "rollout_size": RTH_INTERVALS, "date_val": "2024-03-06", "anno": ANNO,
            })
            with self.assertRaises(ValueError):
                make_pair_dataloader(cfg_data, OmegaConf.create({"split": "train", "batch_size": 1,
                                                                "workers_num": 0}), 0)

    def test_session_table_keeps_only_complete_regular_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            complete = self._session_seconds("2024-03-05")
            short = self._session_seconds("2024-03-06")[:-5]
            self._write_symbol(directory, "AAA", np.concatenate((complete, short)), 100.0)
            table = session_table(directory / "AAA.npy", ANNO)
            self.assertEqual(table.date.tolist(), ["2024-03-05"])
            self.assertEqual(int(table.sod_sec.iloc[0]), int(complete[0]))
            self.assertEqual(int(table.eod_sec.iloc[0]), int(complete[-1]))


if __name__ == "__main__":
    unittest.main()
