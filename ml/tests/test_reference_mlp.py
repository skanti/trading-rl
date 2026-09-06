import tempfile
import unittest
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from ml import model
from ml.reference_mlp import (
    MLP_REFERENCE_FEATURE_DIM,
    build_shifted_price_features,
    collect_shifted_mlp_rollout,
    shifted_mlp_train_step,
)
from ml.reference_dataset import MarketReferenceDataset
from ml.reference_mlp_train import MAX_MLP_PARAMETERS, build_mlp_models


def tiny_actor(window_size: int = 4) -> model.TradingActor:
    return model.TradingActor(
        window_size=window_size,
        feature_dim=MLP_REFERENCE_FEATURE_DIM,
        hidden_dim=16,
        depth=2,
        scalar_dim=1,
    )


class ShiftedReferenceMLPTest(unittest.TestCase):
    def test_build_mlp_models_enforces_the_reference_feature_layout(self):
        cfg = OmegaConf.create(
            {
                "model": {
                    "mlp": {
                        "window_size": 8,
                        "feature_dim": MLP_REFERENCE_FEATURE_DIM,
                        "hidden_dim": 16,
                        "depth": 2,
                        "action_dim": 3,
                        "scalar_dim": 1,
                    }
                }
            }
        )
        actor, critic = build_mlp_models(cfg, torch.device("cpu"))
        parameters = sum(
            parameter.numel()
            for parameter in chain(actor.parameters(), critic.parameters())
        )
        self.assertEqual(actor.window_size, 8)
        self.assertEqual(actor.scalar_dim, 1)
        self.assertLess(parameters, MAX_MLP_PARAMETERS)
        cfg.model.mlp.feature_dim = MLP_REFERENCE_FEATURE_DIM + 1
        with self.assertRaises(ValueError):
            build_mlp_models(cfg, torch.device("cpu"))

    def test_dataset_returns_prices_only_and_time_matches_spy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            asset_secs = np.array([0, 60, 120, 180, 240, 360, 420], dtype=np.int64)
            spy_secs = np.array([0, 60, 120, 180, 300, 360, 420], dtype=np.int64)
            np.save(
                path / "ASSET.npy",
                np.stack((asset_secs, np.arange(100_000, 100_700, 100), np.ones(7)), axis=1),
            )
            np.save(
                path / "SPY.npy",
                np.stack((spy_secs, np.arange(500_000, 500_700, 100), np.ones(7)), axis=1),
            )
            rows = []
            for symbol in ("ASSET", "SPY"):
                for date, sod, eod, sod_idx, eod_idx in (
                    ("2025-01-02", 0, 120, 0, 2),
                    ("2025-01-03", 180, 420, 3, 6),
                ):
                    rows.append(
                        {
                            "sample_id": symbol,
                            "date": pd.Timestamp(date),
                            "ctx_idx": 0,
                            "sod_idx": sod_idx,
                            "eod_idx": eod_idx,
                            "sod_sec": sod,
                            "eod_sec": eod,
                            "context_sod_sec": sod,
                            "context_eod_sec": eod,
                            "is_tradable": True,
                        }
                    )
            dataset = MarketReferenceDataset(
                pd.DataFrame(rows), str(path), "SPY", window_size=4, rollout_size=4
            )
            item = dataset[0]
            self.assertEqual(set(item), {"_id", "prices", "reference_prices", "secs"})
            self.assertEqual(item["secs"].tolist(), [0, 60, 120, 180, 240, 300, 360, 420])
            self.assertAlmostEqual(item["prices"][5], 100.4, places=4)
            self.assertAlmostEqual(item["reference_prices"][4], 500.3, places=4)

    def test_price_features_are_scale_invariant_and_shift_causally(self):
        asset = torch.tensor([[100.0, 101.0, 99.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]])
        spy = torch.tensor([[500.0, 501.0, 502.0, 499.0, 503.0, 504.0, 505.0, 506.0, 507.0]])
        first = build_shifted_price_features(
            asset, spy, context_ticks=5, window_size=4, rollout_size=3
        )
        scaled = build_shifted_price_features(
            asset * 17.0,
            spy * 3.0,
            context_ticks=5,
            window_size=4,
            rollout_size=3,
        )
        changed = asset.clone()
        changed[:, 6] *= 1.1
        changed_features = build_shifted_price_features(
            changed, spy, context_ticks=5, window_size=4, rollout_size=3
        )

        self.assertEqual(first.shape, (1, 3, 4, 4))
        self.assertTrue(torch.allclose(first, scaled, atol=2e-5))
        self.assertTrue(torch.equal(first[:, 0], changed_features[:, 0]))
        self.assertFalse(torch.equal(first[:, 1], changed_features[:, 1]))

    def test_rollout_shifts_sampled_state_into_the_next_window(self):
        actor = tiny_actor()
        torch.nn.init.zeros_(actor.main[-1].weight)
        asset = torch.arange(100.0, 109.0).unsqueeze(0)
        spy = torch.arange(500.0, 509.0).unsqueeze(0)
        rollout = collect_shifted_mlp_rollout(
            actor,
            asset,
            spy,
            context_ticks=5,
            rollout_size=3,
            sampling="greedy",
        )

        self.assertEqual(rollout.states.shape, (1, 3, 4, 8))
        self.assertEqual(rollout.actions.tolist(), [[model.BUY_ACTION] * 3])
        self.assertEqual(rollout.positions.tolist(), [[1, 1, 1]])
        self.assertTrue(
            torch.allclose(
                rollout.scalars[0, :, 0], torch.tensor([1.0, 2.0 / 3.0, 1.0 / 3.0])
            )
        )
        # Context carries only price features. The sampled action and resulting
        # inventory first appear at the newest slot of the following window.
        self.assertTrue(torch.equal(rollout.states[0, 0, :, 4:], torch.zeros(4, 4)))
        self.assertEqual(rollout.states[0, 1, -1, 4:].tolist(), [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(rollout.states[0, 2, -2:, 4].tolist(), [1.0, 1.0])
        self.assertEqual(rollout.states[0, 2, -2:, 5].tolist(), [1.0, 1.0])

    def test_reference_changes_observations_but_not_asset_rewards(self):
        actor = tiny_actor()
        torch.nn.init.zeros_(actor.main[-1].weight)
        asset = torch.arange(100.0, 109.0).unsqueeze(0)
        flat_spy = torch.full_like(asset, 500.0)
        moving_spy = torch.tensor(
            [[500.0, 501.0, 499.0, 502.0, 498.0, 503.0, 497.0, 504.0, 496.0]]
        )
        flat = collect_shifted_mlp_rollout(
            actor, asset, flat_spy, 5, 3, sampling="greedy"
        )
        moving = collect_shifted_mlp_rollout(
            actor, asset, moving_spy, 5, 3, sampling="greedy"
        )
        self.assertFalse(torch.equal(flat.states, moving.states))
        self.assertTrue(torch.equal(flat.rewards, moving.rewards))

    def test_ppo_update_changes_mlp_parameters(self):
        torch.manual_seed(13)
        actor = tiny_actor()
        critic = model.TradingCritic(
            window_size=4,
            feature_dim=MLP_REFERENCE_FEATURE_DIM,
            hidden_dim=16,
            depth=2,
            scalar_dim=1,
        )
        optimizer = torch.optim.AdamW(
            list(actor.parameters()) + list(critic.parameters()), lr=1e-3
        )
        asset = torch.arange(100.0, 109.0).repeat(2, 1)
        spy = torch.arange(500.0, 509.0).repeat(2, 1)
        cfg = OmegaConf.create(
            {
                "data": {
                    "context_ticks": 5,
                    "rollout_size": 3,
                    "price_feature_scale": 100.0,
                },
                "model": {
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "transaction_cost": 1e-4,
                    "risk_penalty": 0.0,
                    "ppo_clip": 0.2,
                    "ppo_value_clip": 0.2,
                    "max_grad_norm": 1.0,
                    "loss_weights": {
                        "policy": 1.0,
                        "value": 0.5,
                        "entropy": 0.001,
                    },
                },
                "train": {"ppo_epochs": 1},
            }
        )
        before = actor.main[-1].weight.detach().clone()
        rollout, losses = shifted_mlp_train_step(
            actor, critic, optimizer, asset, spy, cfg
        )
        self.assertEqual(rollout.actions.shape, (2, 3))
        self.assertFalse(torch.equal(before, actor.main[-1].weight))
        self.assertTrue(np.isfinite(losses["loss"]))


if __name__ == "__main__":
    unittest.main()
