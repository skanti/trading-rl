import time
import os
import logging
from copy import deepcopy
from functools import partial
from pathlib import Path

import torch
import einops
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import fsspec
from omegaconf import DictConfig
from rich.logging import RichHandler
from scipy.spatial.transform import Rotation
import mujoco as mj
from mujoco import mjx
import torch.nn.functional as F

# import mujoco_warp as mjw
import jax
import jax.numpy as jp
from tqdm import tqdm
import einops

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()], force=True)
logger = logging.getLogger("ENVIRONMENT")


def quaternion_to_euler(quats: torch.Tensor) -> torch.Tensor:
    assert quats.shape[-1] == 4, "Input must be of shape (N, 4)"
    w, x, y, z = quats.unbind(dim=-1)

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp_clamped = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp_clamped)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)


def quat_rotate_vector(quat: torch.Tensor, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    assert quat.ndim == x.ndim
    # normalize quat
    qn = torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(eps)
    q = quat / qn

    w = q[..., 0:1]
    qv = q[..., 1:4]  # (x,y,z)

    t = 2.0 * torch.cross(qv, x, dim=-1)
    return x + w * t + torch.cross(qv, t, dim=-1)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    assert a.ndim == b.ndim
    denom = (torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1)).clamp_min(eps)
    return (a * b).sum(dim=-1) / denom

def add_checkerboard(spec: mj.MjSpec) -> None:
    thickness = 0.1
    # Make arena with textured floor.
    chequered = spec.add_texture(
        name="checkerboard",
        type=mj.mjtTexture.mjTEXTURE_2D,
        builtin=mj.mjtBuiltin.mjBUILTIN_CHECKER,
        width=200,
        height=200,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.3, 0.4, 0.5],
    )
    size = np.array([10.0, 10.0])
    mat = spec.add_material(name="grid", texrepeat=size * 10, reflectance=0.1)
    mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "checkerboard"
    geo = spec.worldbody.add_geom(
        type=mj.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, -thickness],
        size=[size[0], size[1], thickness / 2],
    )
    # geo.friction[0] *= 2.0  # increase tangential friction
    # geo.condim = 4 # enable friction
    geo.material = "grid"


class Bars:
    def __init__(self, cfg_data: DictConfig, cfg_split: DictConfig, device: str):
        # get model path
        self.window_size = cfg_data.window_size
        self.batch_size = cfg_split.batch_size
        self.goals_dim = cfg_data.goals_dim
        self.obs_dim = cfg_data.obs_dim
        self.actions_dim = cfg_data.actions_dim
        self.goobac_dim = self.goals_dim + self.obs_dim + self.actions_dim
        self.tok_res = 1.0 / cfg_data.vocab_size
        self.device = device

        # load the mj model
        spec = mj.MjSpec.from_file(cfg_data.model_path)
        # add checkered texture
        add_checkerboard(spec)
        # compile
        self.sim = spec.compile()

        # global sim options
        self.sim.opt.timestep = cfg_data.sim_dt  # Set the simulation timestep
        self.sim.opt.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
        self.sim.opt.iterations = 1000
        self.sim.opt.tolerance = 1e-9

        # other params
        self.duration = cfg_data.duration
        self.sim_dt = cfg_data.sim_dt
        self.rl_dt = cfg_data.rl_dt
        self.sim_hz = int(1.0 / self.sim.opt.timestep)
        self.rl_hz = int(1.0 / self.rl_dt)
        assert (
            self.sim_hz > self.rl_hz and self.sim_hz % self.rl_hz == 0
        ), f"sim_hz must be multiple of rl_hz, sim_hz={self.sim_hz}, rl_hz={self.rl_hz}"
        self.sim_rl_factor = self.sim_hz // self.rl_hz
        self.steps_num_max = int(self.duration * self.sim_hz)
        logger.info(
            f"Environment initialized, steps_num_max={self.steps_num_max}, duration={self.duration}, sim_hz={self.sim_hz}, rl_hz={self.rl_hz}, batch_size={self.batch_size}"
        )

    def play(self, agent, rw: DictConfig, sampling: str):
        # general params
        b = self.batch_size
        device = self.device
        tok_res = self.tok_res
        goals_dim, obs_dim, actions_dim, goobac_dim = (
            self.goals_dim,
            self.obs_dim,
            self.actions_dim,
            self.goobac_dim,
        )
        idx_goals = torch.arange(0, self.goals_dim, device=device)
        idx_obs = torch.arange(0, self.obs_dim, device=device) + self.goals_dim
        idx_actions = torch.arange(0, self.actions_dim, device=device) + self.goals_dim + self.obs_dim

        # to jax
        mj_sim = self.sim

        use_mjx = False
        if use_mjx:
            mjx_sim = mjx.put_model(mj_sim)
            mjx_data = mjx.make_data(mjx_sim)
            # expand
            # mjx_data = jax.tree.map(lambda x: einops.repeat(x, "... -> b ...", b=b), mjx_data)
            # step_fn = jax.vmap(mjx.step, in_axes=(None, 0))
            sim, data = mjx_sim, mjx_data
            # jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0))).lower(mjx_sim, mjx_data).compile()
            step_fn = mjx.step
        else:
            sim = mj_sim
            data = [mj.MjData(mj_sim) for x in range(b)]
            step_fn = lambda sim, data: [mj.mj_step(sim, d) for d in data]
            # reset
            [mj.mj_resetData(sim, d) for d in data]

        # sample goal
        zeros = torch.zeros(size=(b,), device=device)
        angle = torch.rand(size=(b,), device=device)*np.pi*2.0  # (b,)
        goal = torch.stack([torch.cos(angle), torch.sin(angle), zeros], dim=1)  # (b, goals_dim)

        # sample start
        angle = np.random.uniform(size=(b,))*np.pi*2.0  # (b,)
        zeros = np.zeros(shape=(b,))
        start_rpy = np.stack([zeros, zeros, angle], axis=1)  # (b, 3)
        start_quat = Rotation.from_euler("xyz", start_rpy).as_quat(scalar_first=True)
        for k in range(b):
            data[k].qpos[3:7] = start_quat[k]

        # history
        T = self.steps_num_max
        N = T // self.sim_rl_factor
        H = self.window_size
        inputs = torch.zeros(size=(b, H, goobac_dim), device=device)
        inputs_hist = torch.zeros(size=(b, N, H, goobac_dim), device=device)
        actions_hist = torch.zeros(size=(b, N, actions_dim), device=device)
        pos_last = torch.zeros(size=(b, 3), device=device)
        # trackers
        states = np.zeros(shape=(b, T + 1, sim.nq), dtype=np.float32)
        for i in range(T):
            states[:, i] = state_np = np.array([x.qpos.copy() for x in data])
            if i % self.sim_rl_factor == 0:
                # rl step
                j = i // self.sim_rl_factor
                obs = torch.from_numpy(state_np).float().to(device)
                # update obs
                pos = obs[:, :3].clone()
                obs[:, :3] = pos - pos_last
                # append to inputs
                inputs = torch.roll(inputs, shifts=-1, dims=1)
                inputs[:, -1, idx_goals] = goal
                inputs[:, -1, idx_obs] = obs
                # inputs[:, -1, idx_actions] = 0.0  # placeholder
                # update trackers last
                inputs_hist[:, j] = inputs.clone()

                # play
                actions = agent.play(inputs=inputs, sampling=sampling)

                # update inputs & actions trackers
                inputs[:, -1, idx_actions] = actions.clone()
                actions_hist[:, j] = actions.clone()
                actions_np = actions.cpu().numpy()
                pos_last = pos
                for k in range(b):
                    data[k].ctrl[:] = actions_np[k]

            # Step the physics
            # ctrl = np.stack([d.ctrl for d in data], axis=0)
            # assert np.allclose(ctrl, actions, atol=1e-3), "Control signals do not match!"
            # progress simulation
            assert i >= j
            step_fn(sim, data)

        # record final state
        states[:, -1] = np.array([x.qpos.copy() for x in data])
        # to torch
        states = torch.from_numpy(states).to(device)  # (b, T + 1, nq)
        poses = states[:, :, :3]  # (b, T + 1, 3)
        quats = states[:, :, 3:7]  # (b, T + 1, 4)
        eulers = quaternion_to_euler(quats)  # (b, T + 1, 3)
        fold_fn = lambda x: einops.rearrange(x, "b (n f) ... -> b n f ...", f=self.sim_rl_factor).mean(dim=2)
        # traveled distance
        goal_t = einops.repeat(goal, "b c -> b T c", b=b, T=T)  # (b, T, 3)
        delta = poses[:, 1:] - poses[:, :-1]  # (b, T, 3)
        forward = (delta*goal_t).sum(dim=2)  # (b, T)
        forward = fold_fn(forward)*rw.forward  # (b, T, 3) -> (b, N, 3)
        # heading
        heading = torch.tensor([-1.0, 0.0, 0.0], device=device) # (3,)
        heading = einops.repeat(heading, "c -> b T c", b=b, T=T) # (b, T, 3)
        heading = quat_rotate_vector(quats[:, 1:], heading)  # (b, T, 3)
        heading = F.cosine_similarity(heading, goal_t, dim=-1)
        heading = fold_fn(heading)*rw.heading  # (b, T) -> (b, N)
        # effort
        efforts = torch.zeros_like(actions_hist)  # (b, N, a)
        efforts[:, 1:] = actions_hist[:, 1:] - actions_hist[:, :-1]
        efforts = efforts.abs().mean(dim=2)*rw.effort  # (b, N)
        # tilt penalty
        tilt = eulers[:, 1:, :2].abs().mean(dim=2)  # (b, T)
        tilt = fold_fn(tilt)*rw.tilt  # (b, T) -> (b, N)
        # has flipped?
        has_flipped = (eulers[:, 1:, 0].abs() > np.pi / 2) | (eulers[:, 1:, 1].abs() > np.pi / 2)  # (b, T)
        has_flipped = fold_fn(has_flipped.float())  # (b, T) -> (b, N)

        # terminated
        mask = torch.cumsum(has_flipped, dim=1) == 0  # (b, N)
        assert torch.all(mask[:, 0]), "First step should not be terminated."
        m = mask.float()

        # alive bonus
        reward_alive = m*rw.alive # (b, N)

        # final rewards
        rewards = reward_alive + forward + heading - efforts - tilt  # (b, N)
        rewards[~mask] = -1.0  # penalty for terminated

        info = {
            "rewards": (rewards*m).sum(dim=1).cpu().numpy(),
            "alive": (reward_alive*m).sum(dim=1).cpu().numpy(),
            "forward": (forward*m).sum(dim=1).cpu().numpy(),
            "heading": (heading*m).sum(dim=1).cpu().numpy(),
            "effort": (efforts*m).sum(dim=1).cpu().numpy(),
            "tilt": (tilt*m).sum(dim=1).cpu().numpy(),
            "goal": goal,
            "states": states,
        }

        return inputs_hist, actions_hist, mask, rewards, info
