import os
import shutil
import logging
from itertools import chain
import re
from glob import glob
import time

import numpy as np
from omegaconf import OmegaConf, DictConfig
import einops
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from rich.logging import RichHandler
import argparse

import model, utils
from environment import Bars

logger = logging.getLogger("RL")
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

torch.set_float32_matmul_precision("high")


def main(cfg: DictConfig) -> None:
    # params
    exp_dir = cfg.general.experiment_dir
    out_dir = f"{exp_dir}/out"
    os.makedirs(out_dir, exist_ok=True)

    # device & dtype
    device = cfg.model.device
    batch_size = cfg.model.batch_size

    # environment
    env = Bars(cfg_data=cfg.data, cfg_split=cfg.train, device=device)

    # model and agent
    agent = model.Agent(**cfg.model.agent).to(device)
    critic = model.Critic(**cfg.model.critic).to(device)
    optimizer = optim.Adam(chain(agent.parameters(), critic.parameters()), lr=cfg.model.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.train.steps_num), eta_min=1e-6)
    # print model summary
    trainable_params_num = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    logger.info(f"Model summary, trainable_params_num={trainable_params_num/1e6:0.2f}M")
    # loss weights
    lw, rw = cfg.model.loss_weights, cfg.model.reward_weights
    gamma = cfg.model.gamma

    # logger
    csv_logger = utils.CSVLogger(exp_dir=exp_dir)

    # find checkpoint
    checkpoint_path = utils.load_most_recent_checkpoint(exp_dir)

    # load checkpoint
    i, j = 0, 0
    if checkpoint_path:
        logger.info(f"Loading checkpoint, checkpoint_path={checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        agent.load_state_dict(state_dict["agent"], strict=True)
        critic.load_state_dict(state_dict["critic"], strict=True)
        optimizer.load_state_dict(state_dict["optimizer"])
        if "lr_scheduler" in state_dict:
            lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        i = state_dict.get("global_step", 0)
    elif cfg.model.pretrained_path:
        logger.info(f"Loading pretrained model, pretrained_path={cfg.model.pretrained_path}")
        state_dict = torch.load(cfg.model.pretrained_path, map_location="cpu", weights_only=True)
        agent.load_state_dict(state_dict["agent"], strict=True)
        critic.load_state_dict(state_dict["critic"], strict=True)
        i = 0
    else:
        logger.info("No checkpoint found, training from scratch")

    # training loop
    logger.info(f"Starting training, global_step={i}")
    for i in tqdm(range(i, i + int(cfg.train.steps_num)), desc=cfg.general.experiment_name):
        should_save = i == 0 or (i + 1) % cfg.loop.save_interval == 0
        should_log = (i + 1) % cfg.loop.log_interval == 0

        with torch.no_grad():
            # multinomial
            inputs, actions, mask, rewards, info = env.play(agent, rw=rw, sampling="multinomial")
            # greedy
            # _, rewards_max, _, _ = env.play(agent, sampling="greedy")

        # get shapes
        b, n, h, c = inputs.shape

        # calculate discounted rewards
        returns = torch.zeros_like(rewards)
        next_return = torch.zeros(size=(b,), device=device)
        for t in reversed(range(n)):
            is_alive = 1.0 # mask[:, t]
            next_return = rewards[:, t] + gamma * next_return
            next_return = next_return * is_alive
            returns[:, t] = next_return

        # critic network (old values for advantage targets)
        with torch.no_grad():
            value_old = critic(inputs)

        # calculate returns & advantages (fixed for PPO epochs)
        advantages = returns - value_old
        mean, std = advantages.mean(), advantages.std()
        advantages = (advantages - mean) / (std + 1e-8)

        # get old logprobs for PPO ratio
        with torch.no_grad():
            logprobs_old, _ = agent.get_logits(inputs=inputs, actions=actions)
            logprobs_old = logprobs_old.sum(dim=-1)  # joint log prob over all actions

        # PPO epochs (no minibatches)
        ppo_epochs = int(cfg.train.get("ppo_epochs", 4))
        ppo_clip = float(cfg.model.get("ppo_clip", 0.1))
        ppo_value_clip = float(cfg.model.get("ppo_value_clip", ppo_clip))
        loss_policy = torch.tensor(0.0, device=device)
        loss_entropy = torch.tensor(0.0, device=device)
        loss_value = torch.tensor(0.0, device=device)
        loss = torch.tensor(0.0, device=device)
        for _ in range(ppo_epochs):
            # recompute logprobs and value each epoch
            logprobs, entropy = agent.get_logits(inputs=inputs, actions=actions)
            logprobs = logprobs.sum(dim=-1)  # joint log prob over all actions
            value = critic(inputs)

            ratio = torch.exp(logprobs - logprobs_old)
            clipped = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
            loss_policy = -(torch.min(ratio * advantages, clipped * advantages)).mean()
            loss_entropy = -entropy.mean()
            value_clipped = value_old + torch.clamp(value - value_old, -ppo_value_clip, ppo_value_clip)
            value_loss_unclipped = F.l1_loss(returns, value, reduction="none")
            value_loss_clipped = F.l1_loss(returns, value_clipped, reduction="none")
            loss_value = torch.max(value_loss_unclipped, value_loss_clipped).mean()
            # total loss
            loss = loss_policy * lw.policy + loss_value * lw.value + loss_entropy * lw.entropy

            # backprop
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

        lr_scheduler.step()

        # log
        metric = {}
        metric["loss_policy"] = loss_policy.item()
        metric["loss_entropy"] = loss_entropy.item()
        metric["loss_value"] = loss_value.item()
        metric["lr"] = lr_scheduler.get_last_lr()[0]
        # other metrics
        metric["completion"] = mask.float().mean().item()
        metric["reward"] = info["rewards"].mean().item()
        metric["alive"] = info["alive"].mean().item()
        metric["forward"] = info["forward"].mean().item()
        metric["effort"] = info["effort"].mean().item()
        metric["tilt"] = info["tilt"].mean().item()
        metric["heading"] = info["heading"].mean().item()

        # logger
        if should_log:
            metrics = {
                "step": i,
                "loss": loss.item(),
                "timestamp": time.time(),
                "stage": 0.0,
                **metric,
            }
            csv_logger.write(metrics)
        if should_save:
            # save playback data
            npz_path = f"{out_dir}/mj_playback.npz"
            np.savez(
                npz_path,
                model_path=cfg.data.model_path,
                sim_dt=cfg.data.sim_dt,
                rl_dt=cfg.data.rl_dt,
                goal=info["goal"].cpu().numpy(),
                states=info["states"].cpu().numpy(),
                rewards=rewards.cpu().numpy(),
            )

        if (i + 1) % cfg.loop.checkpoint_interval == 0:
            state_dict = {
                "agent": agent.state_dict(),
                "critic": critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }
            utils.save_checkpoint(out_dir=exp_dir, global_step=i + 1, state_dict=state_dict)
            logger.info(f"Saving checkpoint, step={i + 1}")


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    main(cfg)
