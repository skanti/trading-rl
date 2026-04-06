import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import einops
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()], force=True)
logger = logging.getLogger("MODEL")


class HELPER_TOKEN:
    PAD = 0
    START = 1
    STOP = 2
    NUM = 3

def init_orthogonal_discrete(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.zeros_(m.bias)

def make_blocks(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class Critic(nn.Module):
    def __init__(
        self,
        window_size: int,
        goals_dim: int,
        obs_dim: int,
        actions_dim: int,
        hidden_dim: int = 256,
        vocab_size: int = 128,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.goals_dim = goals_dim
        self.obs_dim = obs_dim
        self.actions_dim = actions_dim
        inputs_dim = goals_dim + obs_dim + actions_dim

        self.main = make_blocks(window_size * inputs_dim, hidden_dim, 1)

        # init
        init_orthogonal_discrete(self.main)
        # last layer init
        nn.init.orthogonal_(self.main[-1].weight, gain=1.0)
        nn.init.zeros_(self.main[-1].bias)

        # self.main = torch.compile(self.main, mode="default", fullgraph=True)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4

        # predict
        b, n, h, c = x.shape
        x = einops.rearrange(x, "b n h c -> (b n) (h c)")
        logits = self.main(x)
        logits = einops.rearrange(logits, "(b n) 1 -> b n", b=b, n=n)
        return logits


class Agent(nn.Module):
    def __init__(
        self,
        window_size: int,
        goals_dim: int,
        obs_dim: int,
        actions_dim: int,
        hidden_dim: int = 256,
        vocab_size: int = 128,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.goals_dim = goals_dim
        self.obs_dim = obs_dim
        self.actions_dim = actions_dim
        inputs_dim = goals_dim + obs_dim + actions_dim

        self.main = make_blocks(window_size * inputs_dim, hidden_dim, actions_dim * vocab_size)
        # init
        init_orthogonal_discrete(self.main)
        # last layer init
        nn.init.orthogonal_(self.main[-1].weight, gain=0.1)
        nn.init.zeros_(self.main[-1].bias)
        # self.main = torch.compile(self.main, mode="default", fullgraph=True)

        self.CN = np.radians(90) # 90 degrees in radians

    def load_checkpoint(self, checkpoint_path: str) -> None:
        with open(checkpoint_path, "rb") as f:
            sd = torch.load(f, map_location="cpu", weights_only=True)
        self.load_state_dict(sd, strict=True)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 2

        # predict
        logits = self.main(x)
        logits = einops.rearrange(logits, "b (a c) -> b a c", a=self.actions_dim, c=self.vocab_size)
        return logits

    def process_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.ndim == 3  # (b, h, inputs_dim)
        x = inputs
        # flatten
        toks = einops.rearrange(x, "b h c -> b (h c)")  # (b, s)
        return toks

    def tokenize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        assert actions.ndim == 2  # (b, actions_dim)
        device = actions.device
        # normalize
        x = actions.clone()
        x = x / self.CN + 0.5  # from [-15deg, 15deg] to [0, 1]
        assert (x >= 0).all() and (x <= 1).all()
        # tokenize
        K = self.vocab_size - 1
        toks = x * K  # scale to [0, K]
        toks = torch.round(toks).to(torch.int)
        m = (toks >= 0) & (toks < self.vocab_size)
        assert m.all(), f"Tokenization out of bounds"
        return toks

    def untokenize_actions(self, toks: torch.Tensor) -> torch.Tensor:
        assert toks.ndim == 2
        device = toks.device
        b, s = toks.shape
        assert s == self.actions_dim
        # untokenize/to floats
        K = self.vocab_size - 1
        actions = toks.to(torch.float32) / K  # scale to [0, 1]
        assert (actions >= 0).all() and (actions <= 1).all()
        # unormalize
        actions = (actions - 0.5) * self.CN  # scale to [-CN/2, CN/2]
        # clamp
        actions = torch.clamp(actions, -self.CN, self.CN)
        return actions

    def get_logits(self, inputs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert inputs.ndim == 4  # (b, n, h, goobac_dim)
        b, n, h, c = inputs.shape
        inputs = einops.rearrange(inputs, "b n h c -> (b n) h c")
        actions = einops.rearrange(actions, "b n a -> (b n) a")
        toks = self.process_inputs(inputs)  # (b, n)
        actions = self.tokenize_actions(actions)  # (b, a)
        # actual pass
        logits = self(toks)
        assert logits.ndim == 3  # (b, actions_dim, vocab_size)
        # get probs
        probs = Categorical(logits=logits)
        logprobs = probs.log_prob(actions)
        entropy = probs.entropy()
        # back to original shape
        logprobs = einops.rearrange(logprobs, "(b n) a -> b n a", b=b, n=n)
        entropy = einops.rearrange(entropy, "(b n) a -> b n a", b=b, n=n)
        return logprobs, entropy

    @torch.no_grad()
    def play(
        self,
        inputs: torch.Tensor,
        sampling: str = "multinomial",
    ) -> torch.Tensor:
        assert inputs.ndim == 3

        # tokenize
        toks = self.process_inputs(inputs)

        # call
        logits = self(toks)
        assert logits.ndim == 3  # (b, actions_dim, vocab_size)

        if sampling == "multinomial":
            toks = Categorical(logits=logits).sample()
        elif sampling == "argmax":
            toks = logits.argmax(dim=-1)
        else:
            raise Exception(f"Sampling not known, sampling={sampling}")

        # untokenize
        actions = self.untokenize_actions(toks)
        # assert (toks == self.tokenize_actions(actions)).all()

        return actions
