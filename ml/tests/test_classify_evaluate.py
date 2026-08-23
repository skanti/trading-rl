import argparse
import unittest

import pandas as pd
import torch
from omegaconf import OmegaConf

from classify_evaluate import (
    RegularMinuteSweepDataset,
    evaluate_bets,
    evaluate_top_bottom,
    parse_anchor_time,
    summarize_trades,
)


class FixedClassifier(torch.nn.Module):
    def __init__(self, logits: list[float]):
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor(logits))

    def forward(self, inputs: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        self.last_inputs = inputs
        self.last_scalars = scalars
        return self.fixed_logits.to(inputs.device)


class ClassifyEvaluateTest(unittest.TestCase):
    def test_anchor_time_parser(self):
        self.assertEqual(parse_anchor_time("09:30"), 9 * 60 + 30)
        self.assertEqual(parse_anchor_time("13:00"), 13 * 60)
        self.assertEqual(parse_anchor_time("16:00"), 16 * 60)
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_anchor_time("09:20")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_anchor_time("lunch")

    def test_confident_signals_open_opposite_spreads_and_close_after_two_days(self):
        probability_logits = [
            2.1972246,   # p=0.90: long stock / short SPY
            -2.1972246,  # p=0.10: short stock / long SPY
            0.2006707,   # p=0.55: below threshold, no trade
        ]
        classifier = FixedClassifier(probability_logits)
        prices = torch.tensor(
            [
                [98.0, 99.0, 100.0, 100.0, 105.0],
                [102.0, 101.0, 100.0, 100.0, 90.0],
                [99.0, 100.0, 101.0, 100.0, 103.0],
            ]
        )
        reference = torch.tensor(
            [
                [97.0, 98.0, 99.0, 100.0, 94.0],
                [98.0, 99.0, 100.0, 100.0, 101.0],
                [98.0, 99.0, 100.0, 100.0, 100.0],
            ]
        )
        batch = {
            "_id": ["ST-A", "ST-B", "ST-C"],
            "date": ["2026-08-03"] * 3,
            "target_date": ["2026-08-05"] * 3,
            "anchor_time": ["13:00"] * 3,
            "weekday": torch.tensor([0, 0, 0]),
            "prices": prices,
            "reference_prices": reference,
            "anchor_progress": torch.full((3,), 0.5),
        }
        cfg = OmegaConf.create({"data": {"price_feature_scale": 100.0}})

        frame, logits, labels = evaluate_bets(
            classifier,
            [batch],
            cfg,
            torch.device("cpu"),
            min_confidence=0.80,
            position_mode="relative",
            transaction_cost_bps=10.0,
        )

        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.side.tolist(), [
            "long_stock_short_spy", "short_stock_long_spy"
        ])
        self.assertEqual(frame.exit_date.tolist(), ["2026-08-05", "2026-08-05"])
        self.assertEqual(labels.tolist(), [1.0, 0.0, 1.0])
        self.assertTrue(frame.signal_correct.all())
        # Both spreads make 11% per $1 stock leg before four 10bp side costs;
        # divided by $2 gross capital, each net return is 5.3%.
        self.assertTrue((frame.gross_pnl_per_stock_leg - 0.11).abs().lt(1e-6).all())
        self.assertTrue((frame.transaction_cost - 0.004).abs().lt(1e-9).all())
        self.assertTrue((frame.net_return_on_gross_capital - 0.053).abs().lt(1e-6).all())
        self.assertEqual(classifier.last_inputs.shape, (3, 4, 1))
        self.assertEqual(classifier.last_scalars.shape, (3, 6))
        self.assertTrue(classifier.last_scalars[:, 1].eq(1).all())

        summary = summarize_trades(
            frame,
            logits,
            labels,
            "model.ckpt",
            500,
            pd.Timestamp("2026-07-25"),
            pd.Timestamp("2026-08-21"),
            "13:00",
            2,
            0.80,
            "relative",
            10.0,
            ("ST-A", "ST-B", "ST-C"),
        )
        self.assertEqual(summary["trades"], 2)
        self.assertAlmostEqual(summary["coverage"], 2 / 3)
        self.assertEqual(summary["long_trades"], 1)
        self.assertEqual(summary["short_trades"], 1)
        self.assertAlmostEqual(summary["trade_win_rate"], 1.0)
        self.assertAlmostEqual(summary["signal_win_rate"], 1.0)

    def test_stock_mode_charges_one_leg_round_trip(self):
        classifier = FixedClassifier([2.1972246])
        batch = {
            "_id": ["ST-A"],
            "date": ["2026-08-03"],
            "target_date": ["2026-08-05"],
            "anchor_time": ["13:00"],
            "weekday": torch.tensor([0]),
            "prices": torch.tensor([[98.0, 99.0, 100.0, 100.0, 105.0]]),
            "reference_prices": torch.tensor([[98.0, 99.0, 100.0, 100.0, 94.0]]),
            "anchor_progress": torch.tensor([0.5]),
        }
        cfg = OmegaConf.create({"data": {"price_feature_scale": 100.0}})
        frame, _, _ = evaluate_bets(
            classifier,
            [batch],
            cfg,
            torch.device("cpu"),
            0.80,
            "stock",
            10.0,
        )
        self.assertEqual(frame.side.item(), "long_stock")
        self.assertAlmostEqual(frame.transaction_cost.item(), 0.002)
        self.assertAlmostEqual(frame.net_return_on_gross_capital.item(), 0.048, places=6)

    def test_top_bottom_ranks_each_day_and_does_not_reopen_active_symbols(self):
        symbols = ["A", "B", "C", "D", "E", "F"]
        classifier = FixedClassifier(
            [3.0, 2.0, 1.0, -1.0, -2.0, -3.0] * 2
        )
        exit_prices = torch.tensor(
            [105.0, 104.0, 103.0, 99.0, 98.0, 97.0] * 2
        )
        prices = torch.full((12, 5), 100.0)
        prices[:, -1] = exit_prices
        reference = torch.full((12, 5), 100.0)
        batch = {
            "_id": symbols * 2,
            "date": ["2026-08-03"] * 6 + ["2026-08-04"] * 6,
            "target_date": ["2026-08-05"] * 6 + ["2026-08-06"] * 6,
            "anchor_time": ["13:00"] * 12,
            "weekday": torch.tensor([0] * 6 + [1] * 6),
            "prices": prices,
            "reference_prices": reference,
            "anchor_progress": torch.full((12,), 0.5),
        }
        cfg = OmegaConf.create({"data": {"price_feature_scale": 100.0}})

        frame, logits, labels = evaluate_top_bottom(
            classifier,
            [batch],
            cfg,
            torch.device("cpu"),
            top_k=1,
            min_score_spread=0.0,
            position_mode="stock",
            transaction_cost_bps=10.0,
        )

        self.assertEqual(frame.sample_id.tolist(), ["A", "F", "B", "E"])
        self.assertEqual(frame.selection_bucket.tolist(), ["top", "bottom"] * 2)
        self.assertEqual(frame.direction.tolist(), [1, -1, 1, -1])
        self.assertEqual(frame.entry_date.tolist(), [
            "2026-08-03", "2026-08-03", "2026-08-04", "2026-08-04"
        ])
        self.assertTrue(frame.signal_correct.all())
        self.assertTrue(frame.trade_won.all())
        self.assertTrue(frame.transaction_cost.eq(0.002).all())
        self.assertTrue(frame.score_spread.gt(0).all())
        self.assertEqual(logits.numel(), 12)
        self.assertEqual(labels.numel(), 12)


if __name__ == "__main__":
    unittest.main()
