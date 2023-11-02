import math
from pathlib import Path
from typing import Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from typing_extensions import Self
from torch import Tensor
import torch.nn.functional as F
from enum import Enum
import json
from utils import find_multiple


class ModelFamily(Enum):
    LLAMA = "llama"
    PHI = "phi"


@dataclass
class Config:
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    lm_head_bias: bool = False
    norm_eps: float = 1e-6
    # determines whether to share the attention norm per block
    # only used for phi-1.5
    shared_attention_norm: bool = False
    # no. of query groups for MHA, GQA
    _n_query_groups: Optional[int] = None
    rope_condense_ratio: int = 1
    rope_base: int = 10000
    model_family: str = ModelFamily.LLAMA.value
    _intermediate_size: int = 5632
    extra_tokens: int = 0

    @property
    def n_query_groups(self) -> int:
        if self._n_query_groups is None:
            return self.n_head
        return self._n_query_groups  # type: ignore

    # no. of embeddings per head
    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @property
    def rope_n_elem(self) -> int:
        return int(self.rotary_percentage * self.head_size)

    @property
    def intermediate_size(self) -> int:
        if not self._intermediate_size:
            return self.n_embd * 4
        return self._intermediate_size

    @classmethod
    def from_name(cls, name: str) -> Self:
        return name_to_config[name]

    def __post_init__(self) -> None:
        # used to pad the vocab size to a multiple of 512
        if self.padded_vocab_size is None:
            self.padded_vocab_size = (
                find_multiple(self.vocab_size, self.padding_multiple)
                + self.extra_tokens
            )
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)


tinyllama_chat_config = Config(
    model_family=ModelFamily.LLAMA.value,
    block_size=2048,
    vocab_size=32000,
    padding_multiple=64,
    n_layer=22,
    n_head=32,
    n_embd=2048,
    _n_query_groups=4,
    rotary_percentage=1.0,
    parallel_residual=False,
    bias=False,
    _intermediate_size=5632,
    norm_eps=1e-5,
    extra_tokens=3,
)

tinyllama_config = Config(
    model_family=ModelFamily.LLAMA.value,
    block_size=2048,
    vocab_size=32000,
    padding_multiple=64,
    n_layer=22,
    n_head=32,
    n_embd=2048,
    rotary_percentage=1.0,
    parallel_residual=False,
    bias=False,
    norm_eps=1e-5,
    _intermediate_size=5632,
    _n_query_groups=4,
)

name_to_config = {
    "TinyLlama-1.1B-Chat-v0.3": tinyllama_chat_config,
    "TinyLlama-1.1B-intermediate-step-480k-1T": tinyllama_config,
}


# https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
# norm without centering mean - learnable weight parameter
class RMSNorm(nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.weight * x_normed).to(dtype=dtype)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


# return a tuple of (cos, sin) of shape (seq_len, n_elem)
def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
) -> Tuple[Tensor, Tensor]:
    # create a range of indices from 0 to n_elem
    theta = 1.0 / (base ** torch.arange(0, n_elem, 2).float() / n_elem)

    # create a range of indices from 0 to seq_len
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(
            config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=RMSNorm(config.n_embd, eps=config.norm_eps),
            )
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[Tensor] = None

    def set_kv_cache(
        self,
        batch_size: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)
        max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA downgrades it to use the inefficient implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            # this will be resolved by https://github.com/pytorch/pytorch/issues/96099
            ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
            self.mask_cache = torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:  # type: ignore
            block.attn.kv_cache = None

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(
                f"Cannot attend to {value}, block size is only {self.config.block_size}"
            )
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        elif value != self.cos.size(0):
            # override
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def rope_cache(
        self, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
        )

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        T = idx.shape[1]
        if self.max_seq_length < T:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}."
            )

        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        # wte is embedding
        x = self.transformer.wte(idx)  # type: ignore # token embeddings of shape (b, t, n_embd)
        # self attention blocks
        for block in self.transformer.h:  # type: ignore
            x = block(x, cos, sin, mask, input_pos)
        # RMSnorm
        x = self.transformer.ln_f(x)  # type: ignore
        return self.lm_head(x)  # (b, t, vocab_size)


# classic MLP with a projection from hidden size to intermediate size
# and back to hidden
class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = (
            None
            if config.shared_attention_norm
            else RMSNorm(config.n_embd, eps=config.norm_eps)
        )
        self.mlp = LLaMAMLP(config)
        self.config = config

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        n_1 = self.norm_1(x)
        h = self.attn(n_1, cos, sin, mask, input_pos)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = self.mlp(n_2) + h + x
        else:
            x = h + x
            x = self.mlp(self.norm_2(x)) + x
        return x


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)


class KVCache(nn.Module):
    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False
        )

    def forward(self, input_pos: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        # use correct dtype
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(k.dtype)
        # update cache with new values
        # https://pytorch.org/docs/stable/generated/torch.Tensor.index_copy_.html#torch.Tensor.index_copy_
        k = self.k.index_copy(2, input_pos, k)
        v = self.k.index_copy(2, input_pos, v)
        return k, v


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        # size of all q,k,v projections of all heads
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # n_embd is the size of the input embedding for a token
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.kv_cache: Optional[torch.Tensor] = None
        self.config = config

    def forward(
        self,
        x: Tensor,
        # for positional embeddings
        cos: Tensor,
        sin: Tensor,
        mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        # batch size, sequence length, n_embd (i.e. embedding size)
        B, T, C = x.shape

        # perform attention projections
        qkv: Tensor = self.attn(x)

        # get the number of query groups for MHA, GQA
        # if this value is 1, then we are using standard multi-head attention
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1 key, and 1 value

        qkv = qkv.view(
            B, T, self.config.n_query_groups, total_qkv, self.config.head_size
        )
        # qkv is hidden_state per token per query-key-value combination per query group per batch entry
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        q, k, v = qkv.split([q_per_kv, 1, 1], dim=2)

        # repeat k and v for non multi-head attention
        # flash attention also requires this
        if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        # Reshape to (B, nh_q, T, hs)
        q = q.reshape(B, -1, T, self.config.head_size)
        k = k.reshape(B, -1, T, self.config.head_size)
        v = v.reshape(B, -1, T, self.config.head_size)

        # we only apply rope to the first N tokens
        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        y = self.scaled_dot_product_attention(q, k, v, mask)
        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # scale by sqrt of head size
        scale = 1.0 / math.sqrt(self.config.head_size)
        # can also use flash attention
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> KVCache:
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
        v_shape = (batch_size, heads, max_seq_length, self.config.head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError(
                    "Please pass the `rope_cache_length=gpt.cos.size(-1)` value"
                )
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                rope_cache_length + self.config.head_size - self.config.rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device, dtype=dtype)  # type: ignore
