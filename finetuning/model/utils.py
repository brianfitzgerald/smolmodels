import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Union

import lightning.pytorch as pl
import torch


PROMPT_EXPANSION_TASK_PREFIX = "Expand the following prompt to add more detail: "
SAFETY_TASK_PREFIX = (
    "Rewrite the following prompt to remove any unsafe or copyrighted content: "
)
IGNORE_TOKEN_INDEX = -100
PAD_TOKEN_ID = 0

OptimizerChoice = Literal["AdamW", "Adafactor", "AdamW8bit"]
DataModuleChoice = Literal[
    "ultra_feedback",
    "code_contests",
    "conversation",
    "conversation_dpo",
    "playwright_summary_to_script",
    "gsm8k_reasoning",
    "gsm8k",
    "connections",
    "writing_grpo",
    "writing_dpo",
]
TuningModeChoice = Literal["dpo", "sft", "grpo", "reward"]


class ModelChoice(Enum):
    T5 = "t5"
    CAUSAL_LM = "causal_lm"
    SIMPLE_BERT = "simple_bert"
    GPT = "gpt"


@dataclass
class LMHyperParams:
    base_model_checkpoint: str = "google/flan-t5-small"
    tokenizer_checkpoint: Optional[str] = None
    max_seq_length: int = 2048
    learning_rate: float = 3e-4
    adam_epsilon: float = 1e-8
    warmup_steps_count: Optional[int] = None
    warmup_ratio: Optional[float] = None
    train_batch_size: int = 4
    val_batch_size: int = 2
    num_train_epochs: int = 25
    gradient_accumulation_steps: int = 2
    n_gpus: int = 1
    max_grad_norm: Optional[float] = None
    seed: int = 42
    weight_decay: float = 0.0
    optimizer: OptimizerChoice = "AdamW8bit"
    tuning_type: TuningModeChoice = "sft"

    def warmup_steps(self, train_steps: Union[int, float]) -> int:
        if self.warmup_ratio:
            return int(self.warmup_ratio * train_steps)
        elif self.warmup_steps_count:
            return self.warmup_steps_count
        else:
            raise ValueError("Either warmup_steps_count or warmup_ratio must be set")

    @property
    def tokenizer_checkpoint_value(self) -> str:
        if self.tokenizer_checkpoint:
            return self.tokenizer_checkpoint
        return self.base_model_checkpoint


class SmModel(pl.LightningModule):
    def __init__(self, hparams: LMHyperParams) -> None:
        super().__init__()
        self.params = hparams
        self.model_choice = ModelChoice.CAUSAL_LM


def ensure_directory(directory: str, clear: bool = True):
    """
    Create a directory and parents if it doesn't exist, and clear it if it does.
    """
    Path(directory).mkdir(exist_ok=True, parents=True)
    if clear:
        shutil.rmtree(directory)
    Path(directory).mkdir(exist_ok=True, parents=True)


def short_hash(input_string: str, truncate_to: int = 8) -> str:
    hash_object = hashlib.sha256(input_string.encode())
    full_hash = hash_object.hexdigest()
    short_hash = full_hash[:truncate_to]
    return short_hash


def save_dataclass_to_json(dataclass_instance, file_path: str):
    with open(file_path, "w") as file:
        json.dump(asdict(dataclass_instance), file, indent=4)


def get_available_device() -> str:
    return (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
