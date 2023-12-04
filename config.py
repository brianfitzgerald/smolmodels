from typing import Optional
from dataclasses import dataclass
from typing_extensions import Self
from enum import Enum
from utils import find_multiple
from pydantic import BaseModel


class ModelFamily(Enum):
    LLAMA = "llama"
    PHI = "phi"
    STABLE_LM = "stablelm"
    MISTRAL = "mistral"


class Config(BaseModel):
    name: str
    organization: str
    model_family: str = ModelFamily.LLAMA.value
    # aka sliding_window in transformers
    block_size: int = 2048
    vocab_size: int = 32000
    padding_multiple: int = 64
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    hidden_size: int = 4096
    rotary_percentage: float = 1
    parallel_residual: bool = True
    bias: bool = True
    lm_head_bias: bool = False
    norm_eps: float = 1e-6
    # determines whether to share the attention norm per block
    # only used for phi-1.5
    shared_attention_norm: bool = False
    # no. of query groups for MHA, GQA
    # aka num_key_value_heads
    rope_condense_ratio: int = 1
    rope_base: int = 10000
    intermediate_size: int = 5632
    extra_tokens: int = 0
    gelu_approximate: str = "none"

    n_query_groups: int = 0
    head_size: int = 0
    rope_n_elem: int = 0

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**name_to_config[name].dict())

    @property
    def hf_path(self) -> str:
        return f"{self.organization}/{self.name}"

    def model_post_init(self, __context) -> None:
        assert self.hidden_size % self.n_head == 0
        self.head_size = self.hidden_size // self.n_head

        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = (
                find_multiple(self.vocab_size, self.padding_multiple)
                + self.extra_tokens
            )
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

        self.rope_n_elem = int(self.rotary_percentage * self.head_size)


tiny_llama_config = Config(
    name="TinyLlama-1.1B-intermediate-step-480k-1T",
    organization="TinyLlama",
    model_family=ModelFamily.LLAMA.value,
    n_layer=22,
    n_head=32,
    hidden_size=2048,
    parallel_residual=False,
    bias=False,
    intermediate_size=5632,
    norm_eps=1e-5,
    n_query_groups=4,
)

tiny_llama_chat_config = tiny_llama_config.model_copy(
    update=dict(name="TinyLlama-1.1B-Chat-v0.6", extra_tokens=3)
)

capybara_config = Config(
    name="Nous-Capybara-3B-V1.9",
    organization="NousResearch",
    model_family=ModelFamily.STABLE_LM.value,
    vocab_size=50304,
    n_layer=32,
    n_head=32,
    hidden_size=2560,
    parallel_residual=False,
    bias=False,
    norm_eps=1e-5,
    intermediate_size=6912,
    n_query_groups=4,
)

model_configs = [tiny_llama_config, tiny_llama_chat_config, capybara_config]

name_to_config = {c.name: c for c in model_configs}
