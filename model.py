import math
from typing import Any, Optional, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from typing_extensions import Self
from torch import Tensor


@dataclass
class Config:
    name: str = ""
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
    _n_query_groups: Optional[int] = None

    @property
    def n_query_groups(self) -> int:
        if self.n_query_groups is None:
            return self.n_head
        return self.n_query_groups

    # no. of embeddings per head
    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @property
    def rope_n_elem(self) -> int:
        return int(self.rotary_percentage * self.head_size)


# https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
# norm without centering mean - learnable weight parameter
class RMSNorm(nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: Tensor):
        dtype = x.dtype
        x = x.float()
        norm_x = torch.mean(x * x, self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.weight * x_normed).to(dtype)

    def reset_parameters(self):
        nn.init.ones_(self.weight)


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
        self.mask_cache: Optional[torch.Tensor] = None


    def set_kv_cache(
        self,
        batch_size: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.shape[-1] # type: ignore
        max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h: # type: ignore
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # https://github.com/pytorch/pytorch/issues/96099
            ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
            self.mask_cache = torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h: # type: ignore
            block.attn.kv_cache = None

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


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)


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
        if self.config.n_query_groups != self.config.n_head and (
            input_pos is None or self.config.n_query_groups != 1
        ):
            # expand all singleton dimensions to required size
            k = k.expand(
                B, self.config.n_query_groups, q_per_kv, T, self.config.head_size
            )
            v = v.expand(
                B, self.config.n_query_groups, q_per_kv, T, self.config.head_size
            )

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
        device: torch.device,
        dtype: torch.dtype,
        rope_cache_length: Optional[int] = 0,
    ) -> "KVCache":
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
        v_shape = (batch_size, heads, max_seq_length, self.config.head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError("Please pass the `rope_cache_length=gpt.cos.size(-1)` value")
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                rope_cache_length + self.config.head_size - self.config.rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device, dtype=dtype)

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
