import tempfile
import unittest
import unittest.mock
from datetime import date, datetime, time, timedelta
from itertools import chain
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

import model
from week import (
    WEEK_FEATURE_DIM,
    WEEK_SCALAR_DIM,
    build_week_scalars,
    collect_week_rollout,
    overnight_metrics,
    validate_week_hours,
    week_train_step,
)
from week_dataset import (
    MarketWeekDataset,
    WeekReferenceDataset,
    ticks_per_context_day,
    ticks_per_session,
    week_context_ticks,
    week_rollout_size,
)
from reference_dataset import trailing_validation_start
from week_train import MAX_MLP_PARAMETERS, build_week_models


ANNO = datetime(2010, 1, 1, tzinfo=ZoneInfo("UTC"))
EASTERN = ZoneInfo("US/Eastern")


def to_seconds(day: date, hour: int, minute: int) -> int:
    stamp = datetime.combine(day, time(hour, minute), tzinfo=EASTERN)
    return int((stamp.astimezone(ZoneInfo("UTC")) - ANNO).total_seconds())


def build_day_frame(sample_ids: tuple[str, ...], sessions: list[date]) -> pd.DataFrame:
    rows = []
    for sample_id in sample_ids:
        for day in sessions:
            rows.append(
                {
                    "sample_id": sample_id,
                    "date": day.isoformat(),
                    "is_tradable": True,
                    "sod_sec": to_seconds(day, 9, 30),
                    "eod_sec": to_seconds(day, 16, 0),
                    "context_sod_sec": to_seconds(day, 4, 0),
                    "context_eod_sec": to_seconds(day, 4, 0) + 959 * 60,
                }
            )
    return pd.DataFrame(rows)


def weekday_sessions(first_monday: date, weeks: int) -> list[date]:
    days = []
    for week in range(weeks):
        monday = first_monday + timedelta(days=7 * week)
        days.extend(monday + timedelta(days=offset) for offset in range(5))
    return days


def write_minute_npy(directory: Path, sample_id: str, sessions: list[date], seed: int) -> None:
    rng = np.random.default_rng(seed)
    secs = []
    for day in sessions:
        start = to_seconds(day, 4, 0)
        secs.append(np.arange(start, start + 960 * 60, 60, dtype=np.int64))
    secs = np.concatenate(secs)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 1e-3, secs.size)))
    volumes = rng.integers(1, 100, secs.size)
    array = np.stack(
        (secs.astype(np.float64), np.round(prices * 1000.0), volumes.astype(np.float64)),
        axis=1,
    )
    np.save(directory / f"{sample_id}.npy", array)


def tiny_actor(window_size: int = 4) -> model.TradingActor:
    return model.TradingActor(
        window_size=window_size,
        feature_dim=WEEK_FEATURE_DIM,
        hidden_dim=16,
        depth=2,
        scalar_dim=WEEK_SCALAR_DIM,
    )


class WeekGridTest(unittest.TestCase):
    def test_ten_minute_grid_sizes(self):
        self.assertEqual(ticks_per_context_day(10), 96)
        self.assertEqual(ticks_per_session(10), 40)
        self.assertEqual(week_context_ticks(10, 10), 960)
        self.assertEqual(week_rollout_size(10), 199)

    def test_resolution_must_divide_both_sessions(self):
        with self.assertRaises(ValueError):
            ticks_per_session(7)
        with self.assertRaises(ValueError):
            ticks_per_context_day(7)

    def test_trailing_four_week_start_is_inclusive(self):
        self.assertEqual(
            trailing_validation_start(pd.Timestamp("2026-08-21"), 4),
            pd.Timestamp("2026-07-25"),
        )

    def test_week_config_matches_the_ten_minute_grid(self):
        cfg = OmegaConf.load(Path(__file__).parents[1] / "main.yaml")
        actor, critic = build_week_models(cfg, torch.device("cpu"))
        parameters = sum(
            parameter.numel()
            for parameter in chain(actor.parameters(), critic.parameters())
        )
        self.assertEqual(str(cfg.data.rollout_mode), "week")
        self.assertEqual(int(cfg.data.tick_minutes), 10)
        self.assertEqual(
            int(cfg.data.context_ticks),
            week_context_ticks(int(cfg.data.context_days), int(cfg.data.tick_minutes)),
        )
        self.assertEqual(
            int(cfg.data.rollout_size), week_rollout_size(int(cfg.data.tick_minutes))
        )
        self.assertLessEqual(actor.window_size, int(cfg.data.context_ticks) + 1)
        self.assertEqual(actor.scalar_dim, WEEK_SCALAR_DIM)
        self.assertLess(parameters, MAX_MLP_PARAMETERS)


class WeekScalarTest(unittest.TestCase):
    def test_clocks_and_monday_anchors(self):
        steps, session_ticks = 7, 4
        prices = torch.tensor([[100.0] * 3 + [100.0, 110.0, 90.0, 100.0, 100.0, 100.0, 100.0, 100.0]])
        reference = torch.full_like(prices, 200.0)
        scalars = build_week_scalars(
            prices, reference, context_ticks=3, rollout_size=steps, session_ticks=session_ticks
        )

        self.assertEqual(scalars.shape, (1, steps, WEEK_SCALAR_DIM))
        self.assertAlmostEqual(scalars[0, 0, 0].item(), 1.0)
        self.assertAlmostEqual(scalars[0, -1, 0].item(), 1.0 / steps, places=6)
        # The day clock restarts at every session boundary and reaches zero on
        # the decision that chooses whether to carry inventory overnight.
        self.assertAlmostEqual(scalars[0, 3, 1].item(), 0.0)
        self.assertAlmostEqual(scalars[0, 4, 1].item(), 1.0)
        # Monday's opening tick anchors the week-to-date return at zero.
        self.assertAlmostEqual(scalars[0, 0, 2].item(), 0.0)
        self.assertAlmostEqual(
            scalars[0, 1, 2].item(), 100.0 * float(np.log(110.0 / 100.0)), places=4
        )
        self.assertTrue(torch.allclose(scalars[0, :, 3], torch.zeros(steps), atol=1e-5))

    def test_week_anchor_is_scale_invariant(self):
        prices = torch.tensor([[100.0, 101.0, 99.0, 103.0, 102.0, 104.0]])
        reference = torch.tensor([[50.0, 51.0, 49.0, 52.0, 53.0, 54.0]])
        base = build_week_scalars(prices, reference, 2, 3, 2)
        scaled = build_week_scalars(prices * 31.0, reference * 7.0, 2, 3, 2)
        self.assertTrue(torch.allclose(base, scaled, atol=2e-4))


class WeekRolloutTest(unittest.TestCase):
    def test_rollout_shapes_and_scalar_wiring(self):
        actor = tiny_actor()
        torch.nn.init.zeros_(actor.main[-1].weight)
        asset = torch.arange(100.0, 112.0).unsqueeze(0)
        spy = torch.arange(500.0, 512.0).unsqueeze(0)
        rollout = collect_week_rollout(
            actor, asset, spy, context_ticks=5, rollout_size=6, session_ticks=3, sampling="greedy"
        )
        self.assertEqual(rollout.states.shape, (1, 6, 4, WEEK_FEATURE_DIM))
        self.assertEqual(rollout.scalars.shape, (1, 6, WEEK_SCALAR_DIM))
        # Context carries prices only; the sampled command and the resulting
        # inventory first appear at the newest slot of the following window.
        self.assertTrue(torch.equal(rollout.states[0, 0, :, 4:], torch.zeros(4, 4)))
        self.assertEqual(rollout.states[0, 1, -1, 4:].tolist(), [1.0, 1.0, 0.0, 0.0])

    def test_position_can_persist_overnight_but_is_liquidated_on_friday(self):
        actor = tiny_actor()
        torch.nn.init.zeros_(actor.main[-1].weight)  # greedy buy at every step
        asset = torch.arange(100.0, 112.0).unsqueeze(0)
        spy = torch.arange(500.0, 512.0).unsqueeze(0)
        rollout = collect_week_rollout(
            actor,
            asset,
            spy,
            context_ticks=5,
            rollout_size=6,
            session_ticks=3,
            transaction_cost=1e-3,
            sampling="greedy",
        )
        self.assertEqual(rollout.positions.tolist(), [[1] * 6])
        # Inventory survives the two intra-week session boundaries.
        self.assertAlmostEqual(overnight_metrics(rollout.positions, 3)["overnight_fraction"], 1.0)
        # The final interval pays both the mark to market and the liquidation.
        self.assertTrue(rollout.forced_closes.all())
        self.assertAlmostEqual(rollout.costs[0, -1].item(), 1e-3, places=9)
        self.assertAlmostEqual(rollout.costs[0, 1].item(), 0.0, places=9)

    def test_actor_dimensions_are_checked(self):
        wrong = model.TradingActor(
            window_size=4, feature_dim=WEEK_FEATURE_DIM, hidden_dim=8, depth=1, scalar_dim=1
        )
        asset = torch.arange(100.0, 112.0).unsqueeze(0)
        with self.assertRaises(ValueError):
            collect_week_rollout(wrong, asset, asset, 5, 6, 3)

    def test_ppo_update_changes_parameters(self):
        torch.manual_seed(11)
        actor = tiny_actor()
        critic = model.TradingCritic(
            window_size=4,
            feature_dim=WEEK_FEATURE_DIM,
            hidden_dim=16,
            depth=2,
            scalar_dim=WEEK_SCALAR_DIM,
        )
        optimizer = torch.optim.AdamW(
            list(actor.parameters()) + list(critic.parameters()), lr=1e-3
        )
        asset = torch.arange(100.0, 112.0).repeat(2, 1)
        spy = torch.arange(500.0, 512.0).repeat(2, 1)
        cfg = OmegaConf.create(
            {
                "data": {
                    "context_ticks": 5,
                    "rollout_size": 6,
                    "tick_minutes": 10,
                    "price_feature_scale": 100.0,
                },
                "model": {
                    "gamma": 0.999,
                    "gae_lambda": 0.95,
                    "transaction_cost": 1e-4,
                    "risk_penalty": 0.0,
                    "ppo_clip": 0.2,
                    "ppo_value_clip": 0.2,
                    "max_grad_norm": 1.0,
                    "loss_weights": {"policy": 1.0, "value": 0.5, "entropy": 0.01},
                },
                "train": {"ppo_epochs": 1},
            }
        )
        # ``week_train_step`` reads the real 10-minute session length, so line
        # the toy rollout up with it rather than with the tiny fixture above.
        cfg.data.rollout_size = 6
        before = actor.main[-1].weight.detach().clone()
        with unittest.mock.patch("week.ticks_per_session", return_value=3):
            rollout, losses = week_train_step(actor, critic, optimizer, asset, spy, cfg)
        self.assertEqual(rollout.actions.shape, (2, 6))
        self.assertFalse(torch.equal(before, actor.main[-1].weight))
        self.assertTrue(np.isfinite(losses["loss"]))


class WeekHoursTest(unittest.TestCase):
    def _week_secs(self, first_monday: date) -> np.ndarray:
        grids = []
        for offset in range(5):
            day = first_monday + timedelta(days=offset)
            start = to_seconds(day, 9, 30)
            grids.append(np.arange(start, start + 391 * 60, 600, dtype=np.int64))
        return np.concatenate(grids)[None, :]

    def test_valid_week_passes(self):
        secs = self._week_secs(date(2026, 8, 17))
        validate_week_hours(secs, 0, 199, 10, 40)

    def test_weekend_tick_is_rejected(self):
        secs = self._week_secs(date(2026, 8, 17)).copy()
        secs[0, -1] += 2 * 86_400  # push Friday's close into Sunday
        with self.assertRaises(ValueError):
            validate_week_hours(secs, 0, 199, 10, 40)

    def test_week_must_start_on_monday(self):
        secs = self._week_secs(date(2026, 8, 18))  # Tuesday through Saturday
        with self.assertRaises(ValueError):
            validate_week_hours(secs, 0, 199, 10, 40)

    def test_off_grid_tick_is_rejected(self):
        secs = self._week_secs(date(2026, 8, 17)).copy()
        secs[0, 5] += 120
        with self.assertRaises(ValueError):
            validate_week_hours(secs, 0, 199, 10, 40)


class WeekDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.sessions = weekday_sessions(date(2026, 3, 2), 5)
        self.days = build_day_frame(("ST-AAA", "ST-SPY"), self.sessions)
        for seed, sample_id in enumerate(("ST-AAA", "ST-SPY")):
            write_minute_npy(self.directory, sample_id, self.sessions, seed)

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_weeks_with_ten_preceding_sessions_are_offered(self):
        dataset = MarketWeekDataset(
            self.days[self.days.sample_id.eq("ST-AAA")], str(self.directory)
        )
        # Five weeks are present; the first two supply context for the rest.
        self.assertEqual(len(dataset), 3)
        self.assertEqual(
            [str(week.date()) for week in dataset.weeks.week_start],
            ["2026-03-16", "2026-03-23", "2026-03-30"],
        )

    def test_sample_grid_covers_monday_open_through_friday_close(self):
        dataset = MarketWeekDataset(
            self.days[self.days.sample_id.eq("ST-AAA")], str(self.directory)
        )
        sample = dataset[0]
        self.assertEqual(sample["prices"].shape, (dataset.context_ticks + dataset.rollout_size + 1,))
        self.assertEqual(sample["secs"].shape, sample["prices"].shape)
        validate_week_hours(
            sample["secs"][None, :],
            dataset.context_ticks,
            dataset.rollout_size,
            dataset.tick_minutes,
            dataset.session_ticks,
        )
        # Context ends at 19:50 on the Friday before the traded week.
        context_end = pd.to_datetime(
            sample["secs"][dataset.context_ticks - 1], unit="s", origin="2010-01-01", utc=True
        ).tz_convert("US/Eastern")
        self.assertEqual((context_end.dayofweek, context_end.hour, context_end.minute), (4, 19, 50))

    def test_incomplete_week_is_dropped(self):
        days = self.days[~self.days.date.eq("2026-03-18")]
        dataset = MarketWeekDataset(days[days.sample_id.eq("ST-AAA")], str(self.directory))
        self.assertNotIn(pd.Timestamp("2026-03-16"), set(dataset.weeks.week_start))

    def test_untradable_session_drops_its_week(self):
        days = self.days.copy()
        days.loc[days.date.eq("2026-03-25"), "is_tradable"] = False
        dataset = MarketWeekDataset(days[days.sample_id.eq("ST-AAA")], str(self.directory))
        self.assertNotIn(pd.Timestamp("2026-03-23"), set(dataset.weeks.week_start))

    def test_reference_is_time_matched_and_never_tradable(self):
        dataset = WeekReferenceDataset(self.days, str(self.directory), "ST-SPY")
        sample = dataset[0]
        self.assertEqual(set(dataset.assets.weeks.sample_id), {"ST-AAA"})
        self.assertEqual(sample["prices"].shape, sample["reference_prices"].shape)
        self.assertFalse(np.array_equal(sample["prices"], sample["reference_prices"]))


if __name__ == "__main__":
    unittest.main()
