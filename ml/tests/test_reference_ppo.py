import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

import model
from reference import (
    REFERENCE_FEATURE_DIM,
    build_reference_features,
    collect_reference_rollout,
    reference_train_step,
)
from reference_dataset import MarketReferenceDataset, trailing_validation_start


class ConstantReferenceActor(model.TradingActor):
    def __init__(self, window_size: int, action: int):
        super().__init__(window_size, REFERENCE_FEATURE_DIM, hidden_dim=8, depth=1)
        self.action = int(action)

    @torch.no_grad()
    def play(self, inputs: torch.Tensor, sampling: str = "multinomial") -> torch.Tensor:
        return torch.full(
            inputs.shape[:-2], self.action, dtype=torch.long, device=inputs.device
        )


class ReferencePPOTest(unittest.TestCase):
    def test_trailing_four_week_start_is_inclusive(self):
        self.assertEqual(
            trailing_validation_start(pd.Timestamp("2026-08-21"), 4),
            pd.Timestamp("2026-07-25"),
        )

    def test_reference_features_are_scale_invariant_and_causal(self):
        prices = torch.tensor([[100.0, 101.0, 100.5, 102.0, 103.0, 102.5]])
        spy = torch.tensor([[500.0, 501.0, 502.0, 501.5, 503.0, 504.0]])
        volume = torch.tensor([[10.0, 12.0, 11.0, 20.0, 18.0, 25.0]])
        spy_volume = torch.tensor([[100.0, 120.0, 130.0, 90.0, 140.0, 160.0]])
        progress = torch.linspace(-1.0, 1.0, 6).reshape(1, -1)
        base = build_reference_features(
            prices, spy, volume, spy_volume, progress, window_size=4, rollout_size=2
        )
        scaled = build_reference_features(
            prices * 17.0,
            spy * 3.0,
            volume,
            spy_volume,
            progress,
            window_size=4,
            rollout_size=2,
        )
        self.assertEqual(base.shape, (1, 2, 4, REFERENCE_FEATURE_DIM - 1))
        self.assertTrue(torch.allclose(base, scaled, atol=2e-4))

        changed_future = spy.clone()
        changed_future[:, -1] *= 2.0
        changed = build_reference_features(
            prices,
            changed_future,
            volume,
            spy_volume,
            progress,
            window_size=4,
            rollout_size=2,
        )
        self.assertTrue(torch.equal(base[:, 0], changed[:, 0]))

    def test_spy_changes_observation_but_not_which_asset_is_rewarded(self):
        actor = ConstantReferenceActor(4, model.BUY_ACTION)
        prices = torch.tensor([[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]])
        volumes = torch.ones_like(prices)
        progress = torch.linspace(-1.0, 1.0, 6).reshape(1, -1)
        flat_spy = torch.full_like(prices, 500.0)
        rising_spy = torch.arange(500.0, 506.0).reshape(1, -1)
        first = collect_reference_rollout(
            actor, prices, flat_spy, volumes, volumes, progress, rollout_size=2
        )
        second = collect_reference_rollout(
            actor, prices, rising_spy, volumes, volumes, progress, rollout_size=2
        )
        self.assertTrue(torch.equal(first.rewards, second.rewards))
        self.assertFalse(torch.equal(first.states, second.states))
        self.assertEqual(first.positions.tolist(), [[1, 1]])

    def test_reference_ppo_update_changes_parameters(self):
        torch.manual_seed(7)
        actor = model.TradingActor(4, REFERENCE_FEATURE_DIM, hidden_dim=16, depth=1)
        critic = model.TradingCritic(4, REFERENCE_FEATURE_DIM, hidden_dim=16, depth=1)
        optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=1e-3
        )
        prices = torch.tensor(
            [[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]]
        ).repeat(2, 1)
        spy = torch.tensor(
            [[500.0, 499.0, 501.0, 502.0, 501.0, 503.0]]
        ).repeat(2, 1)
        volume = torch.ones_like(prices)
        progress = torch.linspace(-1.0, 1.0, 6).repeat(2, 1)
        cfg = OmegaConf.create(
            {
                "data": {"rollout_size": 2, "price_feature_scale": 100.0},
                "model": {
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "transaction_cost": 1e-4,
                    "risk_penalty": 0.0,
                    "ppo_clip": 0.2,
                    "ppo_value_clip": 0.2,
                    "max_grad_norm": 1.0,
                    "loss_weights": {"policy": 1.0, "value": 0.5, "entropy": 0.001},
                },
                "train": {"ppo_epochs": 1},
            }
        )
        before = [parameter.detach().clone() for parameter in actor.parameters()]
        rollout, losses = reference_train_step(
            actor,
            critic,
            optimizer,
            prices,
            spy,
            volume,
            volume,
            progress,
            cfg,
        )
        self.assertEqual(rollout.states.shape, (2, 2, 4, REFERENCE_FEATURE_DIM))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, actor.parameters())))
        self.assertTrue(np.isfinite(losses["loss"]))

    def test_dataset_completes_and_time_matches_asset_with_spy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            asset_secs = np.array([0, 60, 120, 180, 240, 360, 420], dtype=np.int64)
            spy_secs = np.array([0, 60, 120, 180, 300, 360, 420], dtype=np.int64)
            np.save(
                path / "ASSET.npy",
                np.stack((asset_secs, np.arange(100_000, 100_700, 100), np.ones(7)), axis=1),
            )
            np.save(
                path / "ST-SPY.npy",
                np.stack((spy_secs, np.arange(500_000, 500_700, 100), np.ones(7)), axis=1),
            )

            rows = []
            for symbol in ("ASSET", "ST-SPY"):
                rows.extend(
                    (
                        {
                            "sample_id": symbol,
                            "date": pd.Timestamp("2025-01-02"),
                            "ctx_idx": 0,
                            "sod_idx": 0,
                            "eod_idx": 2,
                            "sod_sec": 0,
                            "eod_sec": 120,
                            "context_sod_sec": 0,
                            "context_eod_sec": 120,
                            "is_tradable": True,
                        },
                        {
                            "sample_id": symbol,
                            "date": pd.Timestamp("2025-01-03"),
                            "ctx_idx": 0,
                            "sod_idx": 3,
                            "eod_idx": 6,
                            "sod_sec": 180,
                            "eod_sec": 420,
                            "context_sod_sec": 180,
                            "context_eod_sec": 420,
                            "is_tradable": True,
                        },
                    )
                )
            dataset = MarketReferenceDataset(
                pd.DataFrame(rows),
                str(path),
                "ST-SPY",
                window_size=4,
                rollout_size=4,
            )
            item = dataset[0]
            self.assertEqual(item["_id"], "ASSET")
            self.assertEqual(item["secs"].tolist(), [0, 60, 120, 180, 240, 300, 360, 420])
            self.assertEqual(item["prices"].shape, (8,))
            self.assertEqual(item["reference_prices"].shape, (8,))
            self.assertAlmostEqual(item["prices"][5], 100.4, places=4)
            self.assertAlmostEqual(item["reference_prices"][4], 500.3, places=4)


if __name__ == "__main__":
    unittest.main()
