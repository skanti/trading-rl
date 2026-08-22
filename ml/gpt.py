"""Small causal GPT actor-critic and a 4,096-entry joint price tokenizer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import model as trading_model


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 4096
    max_seq_len: int = 9990
    dim: int = 256
    n_layer: int = 6
    n_head: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    action_dim: int = trading_model.ACTION_DIM

    def __post_init__(self) -> None:
        if self.vocab_size != 4096:
            raise ValueError("the trading tokenizer requires vocab_size=4096")
        if self.max_seq_len < 2 or self.dim < 1 or self.n_layer < 1 or self.n_head < 1:
            raise ValueError("invalid GPT dimensions")
        if self.dim % self.n_head:
            raise ValueError("dim must be divisible by n_head")
        if self.mlp_ratio < 1:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.action_dim != trading_model.ACTION_DIM:
            raise ValueError(f"action_dim must be {trading_model.ACTION_DIM}")


class PairPriceTokenizer:
    """Quantize scale-free asset/SPY log returns into one joint token per tick.

    Each leg receives 64 mu-law levels. Their Cartesian product is exactly
    ``64 * 64 = 4096`` tokens. Mu-law gives small, common intraday returns much
    finer resolution than rare tail moves while retaining a configurable range.
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        bins_per_price: int = 64,
        max_abs_log_return: float = 0.20,
        mu: float = 255.0,
    ) -> None:
        if vocab_size != bins_per_price * bins_per_price:
            raise ValueError("vocab_size must equal bins_per_price squared")
        if vocab_size != 4096 or bins_per_price != 64:
            raise ValueError("the configured dictionary must contain 4096 joint price tokens")
        if max_abs_log_return <= 0 or mu <= 0:
            raise ValueError("tokenizer scale and mu must be positive")
        self.vocab_size = int(vocab_size)
        self.bins_per_price = int(bins_per_price)
        self.max_abs_log_return = float(max_abs_log_return)
        self.mu = float(mu)

    @staticmethod
    def log_returns(prices: torch.Tensor) -> torch.Tensor:
        if prices.ndim != 2 or prices.shape[1] < 1:
            raise ValueError("prices must have shape (batch, sequence)")
        if not torch.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("prices must be positive and finite")
        returns = torch.zeros_like(prices, dtype=torch.float32)
        returns[:, 1:] = torch.log(prices[:, 1:].float() / prices[:, :-1].float())
        return returns

    def quantize_returns(self, returns: torch.Tensor) -> torch.Tensor:
        scaled = (returns.float() / self.max_abs_log_return).clamp(-1.0, 1.0)
        compact = scaled.sign() * torch.log1p(self.mu * scaled.abs()) / math.log1p(self.mu)
        bins = torch.floor((compact + 1.0) * (self.bins_per_price / 2.0)).to(torch.long)
        return bins.clamp_(0, self.bins_per_price - 1)

    def encode(self, prices: torch.Tensor, reference_prices: torch.Tensor) -> torch.Tensor:
        if prices.shape != reference_prices.shape:
            raise ValueError("asset and reference prices must have identical shapes")
        asset_bins = self.quantize_returns(self.log_returns(prices))
        reference_bins = self.quantize_returns(self.log_returns(reference_prices))
        tokens = asset_bins * self.bins_per_price + reference_bins
        if tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.vocab_size):
            raise AssertionError("tokenizer emitted an out-of-vocabulary token")
        return tokens


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.dim // config.n_head
        self.dropout = float(config.dropout)
        self.c_attn = nn.Linear(config.dim, 3 * config.dim)
        self.c_proj = nn.Linear(config.dim, config.dim)
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def setup_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        shape = (batch_size, self.n_head, max_seq_len, self.head_dim)
        self.k_cache = torch.empty(shape, device=device, dtype=dtype)
        self.v_cache = torch.empty(shape, device=device, dtype=dtype)

    def clear_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None

    def forward(self, inputs: torch.Tensor, cache_start: int | None = None) -> torch.Tensor:
        batch, sequence, width = inputs.shape
        qkv = self.c_attn(inputs).view(
            batch, sequence, 3, self.n_head, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query, key, value = (
            tensor.transpose(1, 2) for tensor in (query, key, value)
        )

        if cache_start is None:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            if self.k_cache is None or self.v_cache is None:
                raise RuntimeError("KV cache has not been initialized")
            cache_stop = cache_start + sequence
            if batch > self.k_cache.shape[0] or cache_stop > self.k_cache.shape[2]:
                raise ValueError("KV cache is too small for this batch or sequence")
            self.k_cache[:batch, :, cache_start:cache_stop] = key
            self.v_cache[:batch, :, cache_start:cache_stop] = value
            cached_key = self.k_cache[:batch, :, :cache_stop]
            cached_value = self.v_cache[:batch, :, :cache_stop]
            if cache_start == 0:
                attention_mask = None
                is_causal = sequence > 1
            else:
                query_positions = torch.arange(
                    cache_start, cache_stop, device=inputs.device
                ).unsqueeze(1)
                key_positions = torch.arange(cache_stop, device=inputs.device).unsqueeze(0)
                attention_mask = key_positions <= query_positions
                is_causal = False
            attended = F.scaled_dot_product_attention(
                query,
                cached_key,
                cached_value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=is_causal,
            )
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, width)
        return self.c_proj(attended)


class GPTBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.dim)
        hidden = config.mlp_ratio * config.dim
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, config.dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: torch.Tensor, cache_start: int | None = None) -> torch.Tensor:
        inputs = inputs + self.attn(self.ln_1(inputs), cache_start)
        return inputs + self.mlp(self.ln_2(inputs))


class CausalTradingTransformer(nn.Module):
    """Shared GPT-2-style trunk with policy and value heads.

    The shared trunk is both smaller and more appropriate than two independent
    language models: actor and critic consume the same causal market history.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = GPTConfig(**kwargs)
        config = self.config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.inventory_embedding = nn.Embedding(3, config.dim)
        self.previous_action_embedding = nn.Embedding(4, config.dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(GPTBlock(config) for _ in range(config.n_layer))
        self.final_norm = nn.LayerNorm(config.dim)
        self.policy_head = nn.Linear(config.dim, config.action_dim, bias=False)
        self.value_head = nn.Linear(config.dim, 1, bias=False)
        self.cache_length = 0
        self.cache_batch_size = 0
        self.apply(self._init_weights)
        nn.init.normal_(self.policy_head.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.01)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def setup_cache(
        self,
        batch_size: int,
        max_seq_len: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        max_seq_len = int(max_seq_len or self.config.max_seq_len)
        if max_seq_len > self.config.max_seq_len:
            raise ValueError("requested cache exceeds max_seq_len")
        parameter = next(self.parameters())
        dtype = dtype or parameter.dtype
        first_cache = self.blocks[0].attn.k_cache
        if (
            first_cache is not None
            and first_cache.shape[0] >= batch_size
            and first_cache.shape[2] >= max_seq_len
            and first_cache.device == parameter.device
            and first_cache.dtype == dtype
        ):
            self.cache_batch_size = int(first_cache.shape[0])
            self.cache_length = 0
            return
        for block in self.blocks:
            block.attn.setup_cache(batch_size, max_seq_len, parameter.device, dtype)
        self.cache_batch_size = int(batch_size)
        self.cache_length = 0

    def reset_cache(self) -> None:
        self.cache_length = 0

    def clear_cache(self) -> None:
        for block in self.blocks:
            block.attn.clear_cache()
        self.cache_batch_size = 0
        self.cache_length = 0

    def _hidden(
        self,
        token_ids: torch.Tensor,
        inventory_ids: torch.Tensor,
        previous_action_ids: torch.Tensor,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        if (
            token_ids.shape != inventory_ids.shape
            or token_ids.shape != previous_action_ids.shape
            or token_ids.ndim != 2
        ):
            raise ValueError("token, inventory, and previous-action IDs must share shape")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must be integers")
        if inventory_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("inventory_ids must be integers")
        if previous_action_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("previous_action_ids must be integers")
        if token_ids.numel() and ((token_ids < 0).any() or (token_ids >= self.config.vocab_size).any()):
            raise ValueError("token_ids are outside the configured vocabulary")
        if inventory_ids.numel() and ((inventory_ids < 0).any() or (inventory_ids > 2).any()):
            raise ValueError("inventory_ids must encode short/flat/long as 0/1/2")
        if previous_action_ids.numel() and (
            (previous_action_ids < 0).any() or (previous_action_ids > 3).any()
        ):
            raise ValueError("previous_action_ids must encode buy/nothing/sell/none as 0/1/2/3")
        sequence = token_ids.shape[1]
        start = int(cache_start or 0)
        stop = start + sequence
        if stop > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        positions = torch.arange(start, stop, device=token_ids.device)
        hidden = (
            self.token_embedding(token_ids)
            + self.inventory_embedding(inventory_ids)
            + self.previous_action_embedding(previous_action_ids)
            + self.position_embedding(positions).unsqueeze(0)
        )
        hidden = self.embedding_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden, cache_start)
        return self.final_norm(hidden)

    def forward(
        self,
        token_ids: torch.Tensor,
        inventory_ids: torch.Tensor,
        previous_action_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._hidden(token_ids, inventory_ids, previous_action_ids)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    def cached_forward(
        self,
        token_ids: torch.Tensor,
        inventory_ids: torch.Tensor,
        previous_action_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cache_batch_size:
            raise RuntimeError("call setup_cache before cached_forward")
        if token_ids.shape[0] > self.cache_batch_size:
            raise ValueError("input batch exceeds the initialized cache")
        start = self.cache_length
        hidden = self._hidden(
            token_ids, inventory_ids, previous_action_ids, cache_start=start
        )
        self.cache_length += token_ids.shape[1]
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    def distribution(
        self,
        token_ids: torch.Tensor,
        inventory_ids: torch.Tensor,
        previous_action_ids: torch.Tensor,
    ) -> Categorical:
        logits, _ = self(token_ids, inventory_ids, previous_action_ids)
        return Categorical(logits=logits)
