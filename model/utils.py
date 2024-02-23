from typing import List, Literal
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore
from pathlib import Path
import shutil
import lightning.pytorch as pl
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.utils.data import DataLoader
import os
from dataclasses import dataclass

PROMPT_EXPANSION_TASK_PREFIX = "Expand the following prompt to add more detail: "
SAFETY_TASK_PREFIX = "Rewrite the following prompt to remove any unsafe or copyrighted content: "
IGNORE_TOKEN_INDEX = -100
PAD_TOKEN_ID = 0

OptimizerChoice = Literal["AdamW", "Adafactor"]

@dataclass
class HyperParams:
    base_model_checkpoint: str = "google/flan-t5-small"
    max_seq_length: int = 2048
    learning_rate: float = 3e-4
    adam_epsilon: float = 1e-8
    warmup_steps: int = 50
    train_batch_size: int = 2
    eval_batch_size: int = 2
    num_train_epochs: int = 25
    gradient_accumulation_steps: int = 4
    n_gpus: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42
    weight_decay: float = 0.0
    optimizer: OptimizerChoice = "AdamW"


class FineTunerDataset(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__()

        self.train_dataset = None
        self.val_dataset = None
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.cpu_count = min(len(os.sched_getaffinity(0)), 16)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.cpu_count)  # type: ignore

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.cpu_count)  # type: ignore


def compute_metrics(inputs: List[str], generated: List[str]):
    rouge = ROUGEScore()
    bleu = BLEUScore()

    rouge_scores = rouge(inputs, generated)
    bleu_score = bleu(inputs, generated)

    return {
        **rouge_scores,
        "bleu": bleu_score.item(),
    }


def ensure_directory(directory: str, clear: bool = True):
    """
    Create a directory and parents if it doesn't exist, and clear it if it does.
    """
    Path(directory).mkdir(exist_ok=True, parents=True)
    if clear:
        shutil.rmtree(directory)
    Path(directory).mkdir(exist_ok=True, parents=True)
