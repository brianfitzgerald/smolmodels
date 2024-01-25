import torch
from torch import nn
from einops import rearrange
import math
import torch


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class WrapperBlock(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.layer_norm = T5LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        inner_dim = int(dim * mult)
        self.wi = nn.Linear(dim, inner_dim, bias=False)
        self.wo = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


# bucket the position bias to 32 positions


class T5RelativePositionBias(nn.Module):
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=12):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, causal=True, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(j - i, j, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, "i j h -> h i j")
        return qk_dots + (bias * self.scale)


# TODO use scaled dot product attention
class T5SelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        relative_attn_bias=False,
        num_buckets=32,
        heads=12,
        dim_head=64,
        causal=False,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.causal = causal

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_o = nn.Linear(inner_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        if relative_attn_bias:
            self.to_relative_attention_bias = nn.Embedding(num_buckets, heads)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)  # (b, h, n, n)

        # sim = self.relative_position_bias(sim)

        # mask (b, n)

        mask_value = -torch.finfo(sim.dtype).max

        if mask is not None:
            sim = sim.masked_fill_(~mask[:, None, :, None], mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(
                j - i + 1
            )
            sim = sim.masked_fill(causal_mask, mask_value)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # aggregate

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")

        # combine heads and linear output

        return self.to_o(out)


class T5CrossAttention(nn.Module):
    def __init__(self, *, dim, context_dim=None, heads=12, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else dim

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_o = nn.Linear(inner_dim, dim, bias=False)

        # self.relative_position_bias = T5RelativePositionBias(
        #     scale = dim_head ** -0.5,
        #     causal = False,
        #     heads = heads
        #     )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, mask=None, context_mask=None):
        b, n, _, h = *x.shape, self.heads

        kv_input = context if context is not None else x

        q, k, v = self.to_q(x), self.to_k(kv_input), self.to_v(kv_input)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)

        # sim = self.relative_position_bias(sim)

        # mask

        mask_value = -torch.finfo(sim.dtype).max

        if mask is not None:
            sim = sim.masked_fill_(~mask, mask_value)

        if context_mask is not None:
            sim = sim.masked_fill_(~context_mask[:, None, :], mask_value)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # aggregate

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")

        # combine heads and linear output

        return self.to_o(out)


class T5Encoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        # max_seq_len,
        depth,
        heads=12,
        dim_head=64,
        causal=False,
        mlp_mult=4,
        dropout=0.0,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_tokens, dim)
        # self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.block = nn.ModuleList([])
        for i in range(depth):
            self.block.append(
                nn.ModuleList(
                    [
                        WrapperBlock(
                            dim,
                            T5SelfAttention(
                                relative_attn_bias=i == 0,
                                dim=dim,
                                heads=heads,
                                dim_head=dim_head,
                                causal=causal,
                                dropout=dropout,
                            ),
                        ),
                        WrapperBlock(
                            dim,
                            FeedForward(dim=dim, mult=mlp_mult, dropout=dropout),
                        ),
                    ]
                )
            )

        self.final_layer_norm = T5LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.embed_tokens(x)
        # x = x + self.pos_emb(torch.arange(x.shape[1], device = x.device))

        for attn, mlp in self.block:  # type: ignore
            x = attn(x, mask=mask)
            x = mlp(x)

        x = self.final_layer_norm(x)

        return x


# T5 Decoder


class T5Decoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        # max_seq_len,
        depth,
        heads=12,
        dim_head=64,
        causal=True,
        mlp_mult=4,
        dropout=0.0,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_tokens, dim)
        # self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.block = nn.ModuleList([])
        for i in range(depth):
            self.block.append(
                nn.ModuleList(
                    [
                        WrapperBlock(
                            dim,
                            T5SelfAttention(
                                relative_attn_bias=i == 0,
                                dim=dim,
                                heads=heads,
                                dim_head=dim_head,
                                causal=causal,
                                dropout=dropout,
                            ),
                        ),
                        WrapperBlock(
                            dim,
                            T5CrossAttention(
                                dim=dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        WrapperBlock(
                            dim, FeedForward(dim=dim, mult=mlp_mult, dropout=dropout)
                        ),
                    ]
                )
            )

        self.final_layer_norm = T5LayerNorm(dim)

    def forward(self, x, context, mask=None, context_mask=None):
        x = self.embed_tokens(x)
        # x = x + self.pos_emb(torch.arange(x.shape[1], device = x.device))

        for attn, cross_attn, mlp in self.block:  # type: ignore
            x = attn(x, mask=mask)
            x = cross_attn(x, context=context, mask=mask, context_mask=context_mask)
            x = mlp(x)

        x = self.final_layer_norm(x)

        return x


# T5


class T5(nn.Module):
    def __init__(
        self,
        *,
        dim,
        # max_seq_len,
        enc_n_positions: int,
        num_encoder_layers: int,
        enc_heads: int,
        enc_dim_head: int,
        enc_mlp_mult: int,
        vocab_size: int,
        num_decoder_layers: int,
        dec_heads: int,
        dec_dim_head: int,
        dec_mlp_mult: int,
        dropout: float = 0.0,
        tie_token_emb: bool = True,
    ):
        super().__init__()

        self.shared = nn.Embedding(vocab_size, dim)
        # self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.encoder = T5Encoder(
            dim=dim,
            # max_seq_len = max_seq_len,
            num_tokens=enc_n_positions,
            depth=num_encoder_layers,
            heads=enc_heads,
            dim_head=enc_dim_head,
            mlp_mult=enc_mlp_mult,
            dropout=dropout,
        )

        self.decoder = T5Decoder(
            dim=dim,
            # max_seq_len= max_seq_len,
            num_tokens=vocab_size,
            depth=num_decoder_layers,
            heads=dec_heads,
            dim_head=dec_dim_head,
            mlp_mult=dec_mlp_mult,
            dropout=dropout,
        )

        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # tie weights
        if tie_token_emb:
            self.encoder.embed_tokens.weight = self.decoder.embed_tokens.weight

    def forward(self, src, tgt, mask=None, context_mask=None):
        x = self.shared(src)
        # x = x + self.pos_emb(torch.arange(x.shape[1], device = x.device))
        x = self.encoder(src, mask=mask)
        x = self.decoder(tgt, x, mask=mask, context_mask=context_mask)
        x = self.lm_head(x)
        return x


def remap_state_dict(state_dict):
    """
    remap from HF checkpoint
    """
    for k, v in list(state_dict.items()):
        ln = k.split(".")
        if len(ln) < 4:
            continue
        model_module = ln[0]
        layer_idx = ln[2]
        new_key = None
        # self attn layer
        block_sub_idx = int(ln[4])
        if ln[5] == "SelfAttention" or ln[5] == "EncDecAttention":
            attn_layer = ln[6]
            task = ln[7]
            new_key = f"{model_module}.block.{layer_idx}.{block_sub_idx}.fn.to_{attn_layer}.{task}"
            state_dict[new_key] = v
        elif ln[5] == "layer_norm":
            task = ln[6]
            new_key = (
                f"{model_module}.block.{layer_idx}.{block_sub_idx}.layer_norm.{task}"
            )
        elif ln[5] == "DenseReluDense":
            sub_layer = ln[6]
            task = ln[7]
            new_key = f"{model_module}.block.{layer_idx}.{block_sub_idx}.fn.{sub_layer}.{task}"

        if new_key:
            state_dict[new_key] = v
            del state_dict[k]
    return state_dict
