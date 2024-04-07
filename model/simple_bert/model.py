import torch
from torch import nn, Tensor
from dataclasses import dataclass
import math
import torch.nn.functional as F
from typing import Tuple
import lightning.pytorch as pl
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from model.utils import HyperParams


@dataclass
class SimpleBERTConfig:
    hidden_size: int
    num_attention_heads: int
    pos_emb_radius: int
    embed_size: int
    num_layers: int


class SelfAttention(nn.Module):

    def __init__(self, config: SimpleBERTConfig) -> None:
        """
        Multi head self attention
        Uses pre-layernorm, and relative position embeddings
        No dropout for simplicity
        """
        super().__init__()

        self.config = config
        emb_size = config.hidden_size
        n_head = config.num_attention_heads
        assert emb_size % n_head == 0

        # Rotational position embeddings
        pos_emb_radius = config.pos_emb_radius
        # split the embedding size into n_head parts
        pos_emb_units = config.embed_size // n_head

        # set as parameter so we can learn it
        self.pos_emb_k = nn.Parameter(torch.zeros(pos_emb_radius * 2, pos_emb_units))
        torch.nn.init.normal_(self.pos_emb_k, mean=0, std=0.02)

        # Disable bias - from cramming paper, removes unused parameters
        # qkv = 3 * emb_size
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)

        self.proj = nn.Linear(emb_size, emb_size, bias=False)

    def forward(self, x: Tensor):

        # x is context we are attending to
        # usually the hidden state of the transformer
        # the context is not divided by heads - each head attends to the whole context
        # the context is the output of the previous layer, not the qkv scores

        batch_size, context_size, emb_size = x.size()
        assert emb_size == self.config.hidden_size

        n_head = self.config.num_attention_heads
        head_size = emb_size // n_head

        pos_emb_size, head_size = self.pos_emb_k.size()
        pos_emb_radius = self.config.pos_emb_radius
        assert pos_emb_size == pos_emb_radius * 2

        # calculate in batch, and move the head to the batch dim
        # so out is (batch_size, n_head, context_size, head_size)
        k: Tensor = (
            self.key(x)
            .view(batch_size, context_size, n_head, head_size)
            .transpose(1, 2)
        )
        q: Tensor = (
            self.query(x)
            .view(batch_size, context_size, n_head, head_size)
            .transpose(1, 2)
        )
        v: Tensor = (
            self.value(x)
            .view(batch_size, context_size, n_head, head_size)
            .transpose(1, 2)
        )

        # Get the relative position embeddings for the context
        # i.e. matmul q and the position embeddings split out for each head
        # Only add position embeddings to keys, but not values
        att_rel_pos = q @ self.pos_emb_k.view(1, 1, pos_emb_size, head_size).transpose(
            -2, -1
        )
        # create the relative position embeddings for the context
        # att_idxs = (context_size, context_size)
        att_idxs = (
            torch.clamp(
                torch.arange(context_size)[None, :]
                - torch.arange(context_size)[:, None],
                -pos_emb_radius,
                pos_emb_radius - 1,
            )
            % pos_emb_size
        ).to("cuda")

        # att_rel_pos = (batch_size, n_head, context_size, context_size)
        # this is the relative position embeddings for each head
        # gather retrieves the position embeddings for each value in k
        att_pos = torch.gather(
            att_rel_pos,
            3,
            att_idxs.expand((batch_size, n_head, context_size, context_size)),
        )
        assert att_pos.shape == (batch_size, n_head, context_size, context_size)

        attn_val = q @ k.transpose(-2, -1) + att_pos

        # Scaling attention by the size of the key,
        # as per the original transformer paper
        # this helps keep the gradients from exploding
        attn_scale = 1 / math.sqrt(k.shape[-1])

        # Softmax over the last dim, and scale by the attention scale
        # attn_vals = (batch_size, n_head, context_size, context_size)
        att = F.softmax((attn_val + att_pos) * attn_scale, dim=-1)

        # perform the attention calculation
        # (batch_size, n_head, context_size, context_size) @ (batch_size, n_head, context_size, head_size)
        # returns (batch_size, n_head, context_size, head_size)
        y = att @ v
        # combine the heads
        y = y.transpose(1, 2).contiguous().view(batch_size, context_size, emb_size)

        y = self.proj(y)
        return y


class Block(nn.Module):

    def __init__(self, config: SimpleBERTConfig) -> None:
        super().__init__()

        embed_size = config.embed_size
        # eps - added to the variance in the normalization to avoid dividing by zero
        self.norm1 = nn.LayerNorm(embed_size, eps=1e-6)
        self.attn = SelfAttention(config)
        self.norm2 = nn.LayerNorm(embed_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
        )
        self.pre_layernorm = True
        # No dropout layer, removed in Cramming

    def forward(self, x: Tensor) -> Tensor:

        if self.pre_layernorm:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        else:
            x = self.norm1(x + self.attn(x))
            x = self.norm2(x + self.mlp(x))

        return x


class BERT(nn.Module):

    def __init__(self, config: SimpleBERTConfig, vocab_size: int) -> None:
        super().__init__()

        self.config = config
        embed_size, n_layers = config.embed_size, config.num_layers

        self.token_emb = nn.Embedding(vocab_size, embed_size)
        # Layer norm for each embedding
        self.norm_emb = nn.LayerNorm(embed_size, eps=1e-6)

        self.transformer = nn.Sequential(*[Block(config) for _ in range(n_layers)])

        self.norm_final = nn.LayerNorm(embed_size, eps=1e-6)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize all Linear (attention and MLP) weights to normal(0, 0.02)
        and all biases to 0
        Initialize all Embedding weights to normal(0, 0.02)
        and all LayerNorms to bias of 0 and weight of 1
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: Tensor) -> Tensor:

        x = self.token_emb(x)
        x = self.norm_emb(x)

        x = self.transformer(x)

        # Added in Cramming
        x = self.norm_final(x)

        return x


class MLMHead(nn.Module):
    """
    Head for masked language modeling
    Cramming uses sparse prediction, but not implemented here
    Project the embeddings back to the vocab size, and calculate the loss

    # TODO - implement sparse prediction
    """

    def __init__(self, config: SimpleBERTConfig, vocab_size: int) -> None:
        super().__init__()

        self.proj = nn.Linear(config.embed_size, vocab_size, bias=False)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        logits: Tensor = self.proj(x)
        # ignore_index is the padding token
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0
        )
        return logits, loss


class SimpleBertForMaskedLM(pl.LightningModule):
    """
    BERT model with a language modeling head
    """

    def __init__(self, hparams: HyperParams) -> None:
        super().__init__()

        config = SimpleBERTConfig(
            hidden_size=128,
            pos_emb_radius=16,
            embed_size=768,
            num_attention_heads=12,
            num_layers=12,
        )

        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(
            hparams.base_model_checkpoint
        )
        vocab_size = self.tokenizer.vocab_size

        self.bert = BERT(config, vocab_size)
        self.head = MLMHead(config, vocab_size)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.bert(x)
        return self.head(x, y)
