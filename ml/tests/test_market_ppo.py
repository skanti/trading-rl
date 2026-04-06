import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

import llama
from dataset import TokLoader
from tokenizer import Tokenizer
from train import (
    LlamaActor,
    LlamaCritic,
    RolloutBatch,
    batch_to_rollout,
    discounted_returns,
    market_rewards,
    ppo_update,
    regular_session_mask,
)


ANNO = "2010-01-01"


def et_seconds(ts: str) -> int:
    dt = pd.Timestamp(ts, tz="US/Eastern").tz_convert("UTC")
    origin = pd.Timestamp(ANNO, tz="UTC")
    return int((dt - origin).total_seconds())


def tiny_model_args(block_size: int = 6) -> llama.ModelArgs:
    return llama.ModelArgs(
        batch_size=4,
        block_size=block_size,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        dim=16,
        intermediate_size=32,
    )


class MarketPPOTest(unittest.TestCase):
    def test_tokenizer_uses_price_volume_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            secs = np.array([et_seconds(f"2025-01-02 09:{30 + i:02d}:00") for i in range(4)], dtype=np.int32)
            price = np.array([100000, 100100, 100200, 100300], dtype=np.int32)
            volume = np.array([1, 4, 9, 16], dtype=np.int32)
            ignored_num = np.array([10, 20, 30, 40], dtype=np.int32)
            arr = np.stack([secs, price, volume, ignored_num], axis=1)

            npy_path = tmp_path / "ABC.npy"
            np.save(npy_path, arr)

            tokenizer = Tokenizer(anno=ANNO, mapping={"ABC": 0}, seq_size=4, ctx_size=2, channels=2, vocab_size=128)
            tokenizer.parse(str(npy_path), np.arange(4))
            ctx, seq, ts = tokenizer.tokenize()

            arr[:, 3] = 9999
            np.save(npy_path, arr)
            tokenizer.parse(str(npy_path), np.arange(4))
            _, seq_num_changed, _ = tokenizer.tokenize()

            self.assertEqual(ctx.shape, (2,))
            self.assertEqual(seq.shape, (8,))
            self.assertEqual(ts.shape, (8, 5))
            self.assertTrue(np.array_equal(seq, seq_num_changed))

    def test_actor_critic_are_separate_last_token_models(self):
        torch.manual_seed(0)
        actor = LlamaActor(tiny_model_args())
        critic = LlamaCritic(tiny_model_args())
        actor_ptrs = {p.data_ptr() for p in actor.parameters()}
        critic_ptrs = {p.data_ptr() for p in critic.parameters()}

        def fail_next_token(*args, **kwargs):
            raise AssertionError("next_token should not be used for PPO")

        actor.backbone.next_token = fail_next_token
        ctx = torch.tensor([[0.0, 100.0], [1.0, 101.0]])
        seq = torch.randint(0, 64, (2, 4))
        ts = torch.zeros((2, 4, 5), dtype=torch.long)

        logits = actor(ctx=ctx, seq=seq, ts=ts)
        values = critic(ctx=ctx, seq=seq, ts=ts)

        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(values.shape, (2,))
        self.assertTrue(actor_ptrs.isdisjoint(critic_ptrs))

    def test_batch_to_rollout_uses_seq_size_times_channels_context(self):
        batch = {
            "ctx": torch.tensor([[0.0, 100.0]]),
            "seq": torch.arange(12).reshape(1, 12),
            "ts": torch.zeros((1, 12, 5), dtype=torch.long),
            "prices": torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]),
            "secs": torch.tensor([[et_seconds(f"2025-01-02 09:{30 + i:02d}:00") for i in range(6)]]),
        }
        cfg_data = OmegaConf.create({"seq_size": 4, "channels": 2, "rollout_size": 2})

        rollout = batch_to_rollout(batch=batch, cfg_data=cfg_data, device="cpu")

        self.assertEqual(rollout.seq.shape, (2, 8))
        self.assertEqual(rollout.ts.shape, (2, 8, 5))
        self.assertEqual(rollout.price_now.tolist(), [[13.0, 14.0]])
        self.assertEqual(rollout.price_next.tolist(), [[14.0, 15.0]])

    def test_tokloader_returns_context_plus_rollout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            secs = np.array([et_seconds(f"2025-01-02 09:{27 + i:02d}:00") for i in range(8)], dtype=np.int32)
            price = np.arange(100000, 100800, 100, dtype=np.int32)
            volume = np.ones(8, dtype=np.int32)
            ignored_num = np.arange(8, dtype=np.int32)
            np.save(tmp_path / "ABC.npy", np.stack([secs, price, volume, ignored_num], axis=1))
            days = pd.DataFrame(
                [{"sample_id": "ABC", "date": pd.Timestamp("2025-01-02"), "sod_idx": 3, "eod_idx": 7}]
            )
            loader = TokLoader(
                days=days,
                mapping={"ABC": 0},
                data_dir=str(tmp_path),
                seq_size=4,
                ctx_size=2,
                channels=2,
                vocab_size=128,
                anno=ANNO,
                rollout_size=2,
                should_augment=False,
            )

            item = loader[0]

            self.assertEqual(item["seq"].shape, (12,))
            self.assertEqual(item["ts"].shape, (12, 5))
            self.assertEqual(item["prices"].shape, (6,))

    def test_market_hours_and_forced_close(self):
        secs = torch.tensor(
            [
                et_seconds("2025-01-02 09:29:00"),
                et_seconds("2025-01-02 09:30:00"),
                et_seconds("2025-01-02 16:00:00"),
                et_seconds("2025-01-02 16:01:00"),
            ]
        )
        mask = regular_session_mask(secs, ANNO)
        self.assertEqual(mask.tolist(), [False, True, True, False])

        actions = torch.tensor([[2, 2], [0, 0]])
        price_now = torch.tensor([[100.0, 101.0], [100.0, 101.0]])
        price_next = torch.tensor([[101.0, 102.0], [101.0, 102.0]])
        rewards, info = market_rewards(actions, price_now, price_next, transaction_cost=0.0)

        self.assertGreater(rewards[0].sum().item(), 0)
        self.assertLess(rewards[1].sum().item(), 0)
        self.assertEqual(info["forced_closes"].tolist(), [True, True])
        self.assertEqual(info["end_positions"].tolist(), [0.0, 0.0])

    def test_ppo_update_cpu_sanity(self):
        torch.manual_seed(0)
        actor = LlamaActor(tiny_model_args())
        critic = LlamaCritic(tiny_model_args())
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)

        b, t, context_tokens = 2, 2, 4
        rollout = RolloutBatch(
            ctx=torch.tensor([[0.0, 100.0]] * (b * t)),
            seq=torch.randint(0, 64, (b * t, context_tokens)),
            ts=torch.zeros((b * t, context_tokens, 5), dtype=torch.long),
            price_now=torch.tensor([[100.0, 101.0], [100.0, 101.0]]),
            price_next=torch.tensor([[101.0, 100.0], [99.0, 102.0]]),
            secs_now=torch.zeros((b, t), dtype=torch.long),
            secs_next=torch.zeros((b, t), dtype=torch.long),
            batch_size=b,
            rollout_size=t,
        )
        with torch.no_grad():
            dist = actor.distribution(rollout.ctx, rollout.seq, rollout.ts)
            actions_flat = dist.sample()
            logprobs_old = dist.log_prob(actions_flat).reshape(b, t)
            values_old = critic(rollout.ctx, rollout.seq, rollout.ts).reshape(b, t)

        actions = actions_flat.reshape(b, t)
        rewards, _ = market_rewards(actions, rollout.price_now, rollout.price_next)
        returns = discounted_returns(rewards, gamma=0.9)
        advantages = returns - values_old
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        cfg = OmegaConf.create(
            {
                "model": {
                    "loss_weights": {"policy": 1.0, "value": 1.0, "entropy": 0.01},
                    "ppo_clip": 0.2,
                    "ppo_value_clip": 0.2,
                    "max_grad_norm": 1.0,
                },
                "train": {"ppo_epochs": 1},
            }
        )
        before = [p.detach().clone() for p in actor.parameters()]

        losses = ppo_update(
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            rollout=rollout,
            actions=actions,
            logprobs_old=logprobs_old,
            values_old=values_old,
            returns=returns,
            advantages=advantages,
            cfg=cfg,
        )

        self.assertTrue(all(np.isfinite(v) for v in losses.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, actor.parameters())))


if __name__ == "__main__":
    unittest.main()
