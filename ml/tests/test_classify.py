import math
import tempfile
import unittest
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scripts.tests.bar_fixtures import ohlcv_fixture
import pandas as pd
import torch
from omegaconf import OmegaConf

from ml.classify import (
    CLASSIFY_FEATURE_DIM,
    CLASSIFY_SCALAR_DIM,
    DUAL_PRICE_FEATURE_NAMES,
    RelativeDirectionClassifier,
    binary_metrics,
    build_dual_price_features,
    build_classify_scalars,
    build_relative_features,
    relative_labels,
    relative_log_prices,
    stock_direction_labels,
)
from ml.classify_dataset import (
    RelativeDirectionDataset,
    classification_split_mask,
    regular_hours_offsets,
    session_tick_count,
)
from ml.classify_train import MAX_PARAMETERS, MIN_PARAMETERS, build_classifier
from ml.classify_evaluate import RegularMinuteSweepDataset
from ml.week_dataset import forward_filled_prices


ANNO = datetime(2010, 1, 1, tzinfo=ZoneInfo("UTC"))
EASTERN = ZoneInfo("US/Eastern")


def to_seconds(day: date, hour: int, minute: int) -> int:
    stamp = datetime.combine(day, time(hour, minute), tzinfo=EASTERN)
    return int((stamp.astimezone(ZoneInfo("UTC")) - ANNO).total_seconds())


def weekday_sessions(first_monday: date, weeks: int) -> list[date]:
    days = []
    for week in range(weeks):
        monday = first_monday + timedelta(days=7 * week)
        days.extend(monday + timedelta(days=offset) for offset in range(5))
    return days


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


def write_minute_npy(directory: Path, sample_id: str, sessions: list[date], seed: int) -> None:
    rng = np.random.default_rng(seed)
    secs = np.concatenate(
        [np.arange(to_seconds(d, 4, 0), to_seconds(d, 4, 0) + 960 * 60, 60, dtype=np.int64)
         for d in sessions]
    )
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 1e-3, secs.size)))
    array = np.stack(
        (secs.astype(np.float64), np.round(prices * 1000.0), np.ones(secs.size)), axis=1
    )
    np.save(directory / f"{sample_id}.npy", ohlcv_fixture(array))


class RelativeFeatureTest(unittest.TestCase):
    def test_only_the_ratio_survives(self):
        stock = torch.tensor([[100.0, 101.0, 102.0, 103.0, 110.0]])
        spy = torch.tensor([[50.0, 51.0, 50.0, 52.0, 49.0]])
        base = build_relative_features(stock, spy)
        # Scaling either leg by a constant shifts the log ratio by a constant,
        # which the anchoring removes: absolute price cannot reach the model.
        scaled = build_relative_features(stock * 17.0, spy * 0.003)
        self.assertEqual(base.shape, (1, 4, CLASSIFY_FEATURE_DIM))
        self.assertTrue(torch.allclose(base, scaled, atol=2e-3))
        # The newest window value is the anchor and is therefore exactly zero.
        self.assertAlmostEqual(base[0, -1, 0].item(), 0.0, places=6)

    def test_features_exclude_the_labelled_tick(self):
        stock = torch.tensor([[100.0, 101.0, 102.0, 103.0, 110.0]])
        spy = torch.tensor([[50.0, 51.0, 50.0, 52.0, 49.0]])
        moved = stock.clone()
        moved[:, -1] *= 3.0  # only the future tick changes
        self.assertTrue(
            torch.equal(build_relative_features(stock, spy), build_relative_features(moved, spy))
        )
        self.assertNotEqual(
            relative_labels(stock, spy).item(), relative_labels(stock / 3.0, spy).item() * 0 + 0.0
        )

    def test_dual_features_keep_stock_and_spy_as_separate_normalized_channels(self):
        stock = torch.tensor([[100.0, 102.0, 101.0, 104.0, 999.0]])
        spy = torch.tensor([[400.0, 399.0, 403.0, 402.0, 1.0]])
        features = build_dual_price_features(stock, spy)
        rescaled = build_dual_price_features(stock * 17.0, spy * 0.003)
        self.assertEqual(features.shape, (1, 4, len(DUAL_PRICE_FEATURE_NAMES)))
        self.assertTrue(torch.allclose(features, rescaled, atol=2e-3))
        self.assertTrue(torch.equal(features[:, -1], torch.zeros(1, 2)))
        moved_targets = build_dual_price_features(
            torch.cat((stock[:, :-1], stock[:, -1:] * 2), dim=1),
            torch.cat((spy[:, :-1], spy[:, -1:] * 3), dim=1),
        )
        self.assertTrue(torch.equal(features, moved_targets))

class RelativeLabelTest(unittest.TestCase):
    def test_worked_example_from_the_specification(self):
        # Monday 1pm both at 100%; Wednesday 1pm stock 105%, SPY 94% -> True.
        stock = torch.tensor([[100.0, 100.0, 105.0]])
        spy = torch.tensor([[100.0, 100.0, 94.0]])
        self.assertEqual(relative_labels(stock, spy).item(), 1.0)
        # Reverse the two moves and the outcome flips.
        self.assertEqual(relative_labels(spy, stock).item(), 0.0)

    def test_label_matches_the_normalized_price_difference(self):
        torch.manual_seed(3)
        stock = 100.0 * torch.rand(64, 3).add(0.5)
        spy = 400.0 * torch.rand(64, 3).add(0.5)
        normalized = stock[:, -1] / stock[:, -2] - spy[:, -1] / spy[:, -2]
        self.assertTrue(torch.equal(relative_labels(stock, spy), normalized.gt(0).float()))

    def test_absolute_level_of_either_leg_is_irrelevant(self):
        stock = torch.tensor([[100.0, 100.0, 105.0]])
        spy = torch.tensor([[100.0, 100.0, 94.0]])
        self.assertEqual(relative_labels(stock * 31.0, spy * 0.07).item(), 1.0)

    def test_non_positive_prices_are_rejected(self):
        with self.assertRaises(ValueError):
            relative_log_prices(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 1.0]]))

    def test_stock_direction_label_compares_target_with_entry_price_only(self):
        prices = torch.tensor([[99.0, 100.0, 101.0], [101.0, 100.0, 99.0]])
        self.assertEqual(stock_direction_labels(prices).tolist(), [1.0, 0.0])


class BinaryMetricTest(unittest.TestCase):
    def test_win_rate_is_reported_beside_the_trivial_baseline(self):
        logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
        labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
        metrics = binary_metrics(logits, labels)
        self.assertAlmostEqual(metrics["win_rate"], 0.75)
        self.assertAlmostEqual(metrics["base_rate"], 0.25)
        self.assertAlmostEqual(metrics["majority_rate"], 0.75)
        self.assertAlmostEqual(metrics["auc"], 1.0)
        self.assertAlmostEqual(metrics["predicted_positive_rate"], 0.5)

    def test_confident_subset_uses_the_largest_absolute_logits(self):
        logits = torch.tensor([9.0, 0.1, -0.2, 0.3, -0.1, 0.2, 0.05, -0.4, 0.15, -0.25])
        labels = torch.tensor([1.0] + [0.0] * 9)
        metrics = binary_metrics(logits, labels)
        self.assertAlmostEqual(metrics["win_rate_confident"], 1.0)

    def test_auc_is_nan_when_one_class_is_absent(self):
        metrics = binary_metrics(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]))
        self.assertTrue(math.isnan(metrics["auc"]))
        self.assertAlmostEqual(metrics["win_rate"], 1.0)


class ClassifierConfigTest(unittest.TestCase):
    def test_main_config_selects_the_overnight_dual_price_classifier(self):
        cfg = OmegaConf.load(Path(__file__).parents[1] / "main.yaml")
        classifier = build_classifier(cfg, torch.device("cpu"))
        self.assertEqual(str(cfg.general.task), "classification")
        self.assertEqual(str(cfg.data.feature_mode), "dual_normalized")
        self.assertEqual(str(cfg.data.target_mode), "stock_direction")
        self.assertEqual(int(cfg.data.horizon_days), 1)
        self.assertEqual(int(cfg.data.anchor_minute), 15 * 60 + 55)
        self.assertEqual(int(cfg.data.target_minute), 9 * 60 + 45)
        self.assertEqual(int(cfg.data.tick_minutes), 1)
        self.assertEqual(int(cfg.data.window_size), 9_600)
        self.assertEqual(classifier.window_size, int(cfg.data.window_size))
        self.assertEqual(classifier.feature_dim, len(DUAL_PRICE_FEATURE_NAMES))
        self.assertEqual(classifier.scalar_dim, CLASSIFY_SCALAR_DIM)
        parameters = sum(p.numel() for p in classifier.parameters())
        self.assertGreaterEqual(parameters, MIN_PARAMETERS)
        self.assertLessEqual(parameters, MAX_PARAMETERS)
        self.assertEqual(parameters, 1_241_793)

    def test_classifier_emits_one_logit_per_window(self):
        classifier = RelativeDirectionClassifier(window_size=4, hidden_dim=8, depth=2)
        logits = classifier(
            torch.randn(5, 4, CLASSIFY_FEATURE_DIM),
            torch.rand(5, CLASSIFY_SCALAR_DIM),
        )
        self.assertEqual(logits.shape, (5,))

    def test_scalars_contain_clock_and_weekday_one_hot(self):
        scalars = build_classify_scalars(
            torch.tensor([0.0, 0.5, 1.0]), torch.tensor([0, 2, 4])
        )
        self.assertEqual(scalars.shape, (3, CLASSIFY_SCALAR_DIM))
        self.assertTrue(torch.equal(scalars[:, 0], torch.tensor([0.0, 0.5, 1.0])))
        self.assertTrue(
            torch.equal(
                scalars[:, 1:],
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            )
        )


class ClassifyDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.sessions = weekday_sessions(date(2026, 3, 2), 4)
        self.days = build_day_frame(("AAA", "SPY"), self.sessions)
        for seed, sample_id in enumerate(("AAA", "SPY")):
            write_minute_npy(self.directory, sample_id, self.sessions, seed)

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self, **kwargs):
        options = dict(
            days=self.days, data_dir=str(self.directory), reference_symbol="SPY",
            context_days=2, tick_minutes=10, window_size=100, horizon_days=2,
        )
        options.update(kwargs)
        return RelativeDirectionDataset(**options)

    def test_grid_constants(self):
        self.assertEqual(session_tick_count(10), 96)
        self.assertEqual(regular_hours_offsets(10), (33, 72))

    def test_anchors_need_context_behind_and_a_labelled_session_ahead(self):
        dataset = self.dataset()
        # 20 sessions; 2 are consumed by context and 2 by the horizon.
        self.assertEqual(len(dataset), 20 - 2 - 2)
        self.assertNotIn("SPY", set(dataset.samples.sample_id))

    def test_two_session_labels_are_purged_before_validation(self):
        dates = pd.Series(pd.to_datetime(self.sessions))
        validation_start = dates.iloc[12]
        train = classification_split_mask(dates, "train", validation_start, 2)
        validation = classification_split_mask(dates, "val", validation_start, 2)
        train_positions = np.flatnonzero(train.to_numpy())

        self.assertTrue(np.all(train_positions + 2 < 12))
        self.assertEqual(np.flatnonzero(train.to_numpy())[-1], 9)
        self.assertEqual(np.flatnonzero(validation.to_numpy())[0], 12)
        self.assertFalse((train & validation).any())

    def test_next_session_opening_label_is_purged_before_validation(self):
        dates = pd.Series(pd.to_datetime(self.sessions))
        validation_start = dates.iloc[12]
        train = classification_split_mask(dates, "train", validation_start, 1)
        train_positions = np.flatnonzero(train.to_numpy())

        self.assertEqual(train_positions[-1], 10)
        self.assertTrue(np.all(train_positions + 1 < 12))

    def test_anchor_always_lands_inside_regular_hours(self):
        dataset = self.dataset()
        offsets = {dataset.anchor_offset(i) for i in range(len(dataset))}
        self.assertTrue(offsets)
        self.assertGreaterEqual(min(offsets), 33)
        self.assertLessEqual(max(offsets), 72)

    def test_evaluation_can_pin_every_sample_to_one_minute(self):
        dataset = self.dataset(anchor_minute=13 * 60)
        self.assertEqual({dataset.anchor_offset(i) for i in range(len(dataset))}, {54})
        sample = dataset[0]
        self.assertEqual(sample["target_date"], str(pd.Timestamp(self.sessions[4]).date()))
        stamps = pd.to_datetime(
            [sample["secs"][-2], sample["secs"][-1]],
            unit="s",
            origin="2010-01-01",
            utc=True,
        ).tz_convert("US/Eastern")
        self.assertEqual([(stamp.hour, stamp.minute) for stamp in stamps], [(13, 0), (13, 0)])

    def test_fixed_1555_entry_targets_0945_on_the_next_trading_session(self):
        dataset = self.dataset(
            tick_minutes=5,
            horizon_days=1,
            anchor_minute=15 * 60 + 55,
            target_minute=9 * 60 + 45,
        )
        sample = dataset[0]
        stamps = pd.to_datetime(
            [sample["secs"][-2], sample["secs"][-1]],
            unit="s",
            origin="2010-01-01",
            utc=True,
        ).tz_convert("US/Eastern")
        self.assertEqual(sample["anchor_time"], "15:55")
        self.assertEqual(sample["target_time"], "09:45")
        self.assertEqual([(stamp.hour, stamp.minute) for stamp in stamps], [(15, 55), (9, 45)])
        self.assertEqual(stamps[1].date(), pd.Timestamp(self.sessions[3]).date())
        self.assertAlmostEqual(float(sample["anchor_progress"]), 385 / 390, places=6)

    def test_evaluation_can_sweep_every_regular_minute(self):
        base = self.dataset()
        sweep = RegularMinuteSweepDataset(base)
        self.assertEqual(len(sweep), len(base) * 390)
        first, second, last = sweep[0], sweep[1], sweep[389]
        self.assertEqual(
            [first["anchor_time"], second["anchor_time"], last["anchor_time"]],
            ["09:30", "09:31", "15:59"],
        )
        for sample in (first, second, last):
            self.assertEqual(sample["target_date"], str(pd.Timestamp(self.sessions[4]).date()))
        # 13:01 shifts the complete 10-minute lattice by one minute, while
        # preserving the same model input length and two-session target clock.
        at_1301 = sweep[211]
        row = base.samples.iloc[0]
        expected_secs = base.sample_seconds(int(row.session), 54, minute_shift=1)
        expected = forward_filled_prices(
            str(self.directory), str(row.sample_id), expected_secs
        )
        self.assertTrue(np.allclose(at_1301["prices"], expected))

    def test_label_tick_is_the_same_time_of_day_two_sessions_later(self):
        dataset = self.dataset()
        sample = dataset[0]
        secs = sample["secs"]
        self.assertEqual(secs.size, dataset.window_size + 1)
        stamps = pd.to_datetime(
            [secs[-2], secs[-1]], unit="s", origin="2010-01-01", utc=True
        ).tz_convert("US/Eastern")
        anchor, target = stamps[0], stamps[1]
        self.assertEqual((anchor.hour, anchor.minute), (target.hour, target.minute))
        self.assertEqual((target - anchor).days, 2)
        self.assertGreaterEqual(anchor.hour * 60 + anchor.minute, 9 * 60 + 30)
        self.assertLessEqual(anchor.hour * 60 + anchor.minute, 16 * 60)

    def test_horizon_skips_the_weekend(self):
        dataset = self.dataset()
        rows = dataset.samples
        thursday = rows[pd.to_datetime(rows.date).dt.dayofweek.eq(3)].index[0]
        secs = dataset[int(thursday)]["secs"]
        target = pd.to_datetime(secs[-1], unit="s", origin="2010-01-01", utc=True).tz_convert(
            "US/Eastern"
        )
        # Thursday + 2 sessions is Monday, not Saturday.
        self.assertEqual(target.dayofweek, 0)

    def test_validation_anchors_are_deterministic_and_training_ones_are_not(self):
        fixed = self.dataset(should_augment=False)
        self.assertEqual(
            [fixed.anchor_offset(i) for i in range(8)],
            [fixed.anchor_offset(i) for i in range(8)],
        )
        np.random.seed(0)
        drawn = self.dataset(should_augment=True)
        first = [drawn.anchor_offset(0) for _ in range(24)]
        self.assertGreater(len(set(first)), 1)

    def test_sample_is_finite_and_labelable(self):
        dataset = self.dataset()
        sample = dataset[3]
        prices = torch.from_numpy(sample["prices"]).unsqueeze(0)
        reference = torch.from_numpy(sample["reference_prices"]).unsqueeze(0)
        features = build_relative_features(prices, reference)
        self.assertEqual(features.shape, (1, dataset.window_size, CLASSIFY_FEATURE_DIM))
        self.assertTrue(torch.isfinite(features).all())
        self.assertIn(relative_labels(prices, reference).item(), (0.0, 1.0))
        self.assertGreaterEqual(float(sample["anchor_progress"]), 0.0)
        self.assertLessEqual(float(sample["anchor_progress"]), 1.0)
        self.assertEqual(int(sample["weekday"]), pd.Timestamp(sample["date"]).dayofweek)


if __name__ == "__main__":
    unittest.main()
