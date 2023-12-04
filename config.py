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
    vocab_size: int = 50304
    padding_multiple: int = 64
    padded_vocab_size: Optional[int] = None
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    hidden_size: int = 4096
    rotary_percentage: float = 1
    parallel_residual: bool = False
    bias: bool = False
    lm_head_bias: bool = False
    norm_eps: float = 1e-6
    # determines whether to share the attention norm per block
    # only used for phi-1.5
    shared_attention_norm: bool = False
    # rope_scaling in transformers
    rope_condense_ratio: int = 1
    rope_base: int = 10000
    intermediate_size: int = 5632
    extra_tokens: int = 0
    gelu_approximate: str = "none"

    num_key_value_heads: Optional[int] = 32
    head_size: int = 0
    rope_n_elem: int = 0

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**name_to_config[name].dict())

    @property
    def hf_path(self) -> str:
        return f"{self.organization}/{self.name}"

    def model_post_init(self, __context) -> None:
        assert self.hidden_size % self.num_attention_heads == 0
        self.head_size = self.hidden_size // self.num_attention_heads

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
        if self.num_key_value_heads is not None:
            assert self.num_attention_heads % self.num_key_value_heads == 0
        else:
            self.num_key_value_heads = self.num_attention_heads

        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

        self.rope_n_elem = int(self.rotary_percentage * self.head_size)


tiny_llama_config = Config(
    name="TinyLlama-1.1B-intermediate-step-480k-1T",
    organization="TinyLlama",
    model_family=ModelFamily.LLAMA.value,
    num_hidden_layers=22,
    num_attention_heads=32,
    hidden_size=2048,
    intermediate_size=5632,
    norm_eps=1e-5,
    num_key_value_heads=4,
)

tiny_llama_chat_config = tiny_llama_config.model_copy(
    update=dict(name="TinyLlama-1.1B-Chat-v0.6", extra_tokens=3)
)

capybara_config = Config(
    name="Nous-Capybara-3B-V1.9",
    organization="NousResearch",
    model_family=ModelFamily.STABLE_LM.value,
    num_hidden_layers=32,
    num_attention_heads=32,
    hidden_size=2560,
    norm_eps=1e-5,
    intermediate_size=6912,
    num_key_value_heads=4,
)

rocket_config = Config(
    name="rocket-3B",
    organization="pansophic",
    model_family=ModelFamily.STABLE_LM.value,
    num_hidden_layers=32,
    num_attention_heads=32,
    hidden_size=2560,
    norm_eps=1e-5,
    intermediate_size=6912,
    rotary_percentage=0.25,
)

open_hermes_config = Config(
    name="OpenHermes-2.5-Mistral-7B",
    organization="OpenHermes",
    model_family=ModelFamily.MISTRAL.value,
    num_hidden_layers=32,
    num_attention_heads=32,
    hidden_size=4096,
    norm_eps=1e-5,
    intermediate_size=14336,
    num_key_value_heads=8,
    vocab_size=32002
)

model_configs = [
    tiny_llama_config,
    tiny_llama_chat_config,
    capybara_config,
    rocket_config,
    open_hermes_config
]

name_to_config = {c.name: c for c in model_configs}
