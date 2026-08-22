import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

import model
from gpt import CausalTradingTransformer, PairPriceTokenizer
from reference import collect_token_rollout, token_train_step
from reference_dataset import MarketReferenceDataset, trailing_validation_start
from reference_train import MAX_MODEL_PARAMETERS, build_transformer


def tiny_transformer(max_seq_len: int = 16) -> CausalTradingTransformer:
    return CausalTradingTransformer(
        vocab_size=4096,
        max_seq_len=max_seq_len,
        dim=32,
        n_layer=2,
        n_head=4,
        mlp_ratio=2,
        dropout=0.0,
        action_dim=3,
    )


class ReferencePPOTest(unittest.TestCase):
    def test_trailing_four_week_start_is_inclusive(self):
        self.assertEqual(
            trailing_validation_start(pd.Timestamp("2026-08-21"), 4),
            pd.Timestamp("2026-07-25"),
        )

    def test_joint_price_tokenizer_has_exactly_4096_scale_invariant_tokens(self):
        tokenizer = PairPriceTokenizer()
        asset = torch.tensor([[100.0, 100.1, 99.9, 102.0, 101.0]])
        spy = torch.tensor([[500.0, 500.2, 501.0, 499.0, 505.0]])
        tokens = tokenizer.encode(asset, spy)
        scaled = tokenizer.encode(asset * 17.0, spy * 3.0)

        self.assertEqual(tokenizer.vocab_size, 4096)
        self.assertTrue(torch.equal(tokens, scaled))
        self.assertGreaterEqual(tokens.min().item(), 0)
        self.assertLess(tokens.max().item(), 4096)
        self.assertEqual(tokens[0, 0].item(), 32 * 64 + 32)

    def test_cached_logits_match_full_causal_forward(self):
        torch.manual_seed(4)
        transformer = tiny_transformer().eval()
        tokens = torch.randint(0, 4096, (2, 12))
        inventory = torch.randint(0, 3, (2, 12))
        previous_actions = torch.randint(0, 4, (2, 12))
        full_logits, full_values = transformer(tokens, inventory, previous_actions)

        transformer.setup_cache(batch_size=2, max_seq_len=12)
        cached_logits = []
        cached_values = []
        logits, values = transformer.cached_forward(
            tokens[:, :7], inventory[:, :7], previous_actions[:, :7]
        )
        cached_logits.append(logits)
        cached_values.append(values)
        for index in range(7, 12):
            logits, values = transformer.cached_forward(
                tokens[:, index : index + 1],
                inventory[:, index : index + 1],
                previous_actions[:, index : index + 1],
            )
            cached_logits.append(logits)
            cached_values.append(values)

        self.assertEqual(transformer.cache_length, 12)
        self.assertTrue(
            torch.allclose(torch.cat(cached_logits, dim=1), full_logits, atol=2e-5)
        )
        self.assertTrue(
            torch.allclose(torch.cat(cached_values, dim=1), full_values, atol=2e-5)
        )

    def test_future_token_cannot_change_earlier_outputs(self):
        torch.manual_seed(8)
        transformer = tiny_transformer().eval()
        tokens = torch.randint(0, 4096, (1, 10))
        inventory = torch.ones_like(tokens)
        previous_actions = torch.full_like(tokens, 3)
        first_logits, first_values = transformer(tokens, inventory, previous_actions)
        changed = tokens.clone()
        changed[:, -1] = (changed[:, -1] + 1) % 4096
        second_logits, second_values = transformer(
            changed, inventory, previous_actions
        )
        self.assertTrue(torch.equal(first_logits[:, :-1], second_logits[:, :-1]))
        self.assertTrue(torch.equal(first_values[:, :-1], second_values[:, :-1]))

    def test_reset_cache_does_not_leak_the_previous_sequence(self):
        torch.manual_seed(6)
        transformer = tiny_transformer().eval()
        transformer.setup_cache(batch_size=1, max_seq_len=12)
        stale_tokens = torch.randint(0, 4096, (1, 12))
        stale_inventory = torch.randint(0, 3, (1, 12))
        stale_actions = torch.randint(0, 4, (1, 12))
        transformer.cached_forward(stale_tokens, stale_inventory, stale_actions)

        tokens = torch.randint(0, 4096, (1, 9))
        inventory = torch.randint(0, 3, (1, 9))
        previous_actions = torch.randint(0, 4, (1, 9))
        expected_logits, expected_values = transformer(
            tokens, inventory, previous_actions
        )
        transformer.reset_cache()
        cached_logits, cached_values = transformer.cached_forward(
            tokens, inventory, previous_actions
        )

        self.assertEqual(transformer.cache_length, 9)
        self.assertTrue(torch.allclose(cached_logits, expected_logits, atol=2e-5))
        self.assertTrue(torch.allclose(cached_values, expected_values, atol=2e-5))

    def test_previous_actions_are_causal_model_inputs(self):
        torch.manual_seed(9)
        transformer = tiny_transformer().eval()
        tokens = torch.randint(0, 4096, (1, 10))
        inventory = torch.ones_like(tokens)
        previous_actions = torch.full_like(tokens, 3)
        first_logits, first_values = transformer(tokens, inventory, previous_actions)

        changed_actions = previous_actions.clone()
        changed_actions[:, 5] = model.BUY_ACTION
        second_logits, second_values = transformer(
            tokens, inventory, changed_actions
        )

        self.assertTrue(torch.equal(first_logits[:, :5], second_logits[:, :5]))
        self.assertTrue(torch.equal(first_values[:, :5], second_values[:, :5]))
        self.assertFalse(torch.equal(first_logits[:, 5:], second_logits[:, 5:]))
        self.assertFalse(torch.equal(first_values[:, 5:], second_values[:, 5:]))

    def test_rollout_primes_context_then_grows_without_sliding(self):
        transformer = tiny_transformer(max_seq_len=8).eval()
        torch.nn.init.zeros_(transformer.policy_head.weight)
        tokenizer = PairPriceTokenizer()
        asset = torch.tensor([[100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7]])
        flat_spy = torch.full_like(asset, 500.0)
        moving_spy = torch.tensor([[500.0, 501.0, 500.0, 502.0, 501.0, 503.0, 502.0, 504.0]])

        first = collect_token_rollout(
            transformer,
            tokenizer,
            asset,
            flat_spy,
            context_ticks=4,
            rollout_size=3,
            sampling="greedy",
        )
        second = collect_token_rollout(
            transformer,
            tokenizer,
            asset,
            moving_spy,
            context_ticks=4,
            rollout_size=3,
            sampling="greedy",
        )
        self.assertEqual(first.token_ids.shape, (1, 7))
        self.assertEqual(transformer.cache_length, 7)
        self.assertEqual(first.actions.tolist(), [[model.BUY_ACTION] * 3])
        self.assertTrue(torch.equal(first.rewards, second.rewards))
        self.assertFalse(torch.equal(first.token_ids, second.token_ids))
        self.assertEqual(first.inventory_ids[0, 4:].tolist(), [1, 2, 2])
        self.assertEqual(first.previous_action_ids[0, 4:].tolist(), [3, 0, 0])

    def test_tokenized_ppo_update_changes_transformer_parameters(self):
        torch.manual_seed(12)
        transformer = tiny_transformer(max_seq_len=8)
        tokenizer = PairPriceTokenizer()
        optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-3)
        asset = torch.tensor(
            [[100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]]
        ).repeat(2, 1)
        spy = torch.tensor(
            [[500.0, 499.0, 501.0, 502.0, 501.0, 503.0, 502.0, 504.0]]
        ).repeat(2, 1)
        cfg = OmegaConf.create(
            {
                "data": {"context_ticks": 4, "rollout_size": 3},
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
        before = transformer.policy_head.weight.detach().clone()
        rollout, losses = token_train_step(
            transformer, tokenizer, optimizer, asset, spy, cfg
        )
        self.assertEqual(rollout.actions.shape, (2, 3))
        self.assertFalse(torch.equal(before, transformer.policy_head.weight))
        self.assertTrue(np.isfinite(losses["loss"]))

    def test_main_transformer_is_below_twenty_million_parameters(self):
        cfg = OmegaConf.load(Path(__file__).parents[1] / "main.yaml")
        transformer = build_transformer(cfg, torch.device("cpu"))
        self.assertLess(transformer.parameter_count(), MAX_MODEL_PARAMETERS)
        self.assertEqual(transformer.config.vocab_size, 4096)
        self.assertEqual(
            transformer.config.max_seq_len,
            int(cfg.data.context_ticks) + int(cfg.data.rollout_size),
        )

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
                path / "ST-SPY.npy",
                np.stack((spy_secs, np.arange(500_000, 500_700, 100), np.ones(7)), axis=1),
            )
            rows = []
            for symbol in ("ASSET", "ST-SPY"):
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
                pd.DataFrame(rows), str(path), "ST-SPY", window_size=4, rollout_size=4
            )
            item = dataset[0]
            self.assertEqual(set(item), {"_id", "prices", "reference_prices", "secs"})
            self.assertEqual(item["secs"].tolist(), [0, 60, 120, 180, 240, 300, 360, 420])
            self.assertAlmostEqual(item["prices"][5], 100.4, places=4)
            self.assertAlmostEqual(item["reference_prices"][4], 500.3, places=4)


if __name__ == "__main__":
    unittest.main()
