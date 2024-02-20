from typing import List
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore
from pathlib import Path
import shutil
import lightning.pytorch as pl
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.utils.data import DataLoader

PROMPT_EXPANSION_TASK_PREFIX = "Expand the following prompt to add more detail: "
IGNORE_TOKEN_INDEX = -100
PAD_TOKEN_ID = 0


class HyperParams:
    max_seq_length: int = 2048
    learning_rate: float = 2e-4
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

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=35)  # type: ignore

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=35)  # type: ignore


def compute_metrics(inputs: List[str], generated: List[str]):
    rouge = ROUGEScore()
    bleu = BLEUScore()

    rouge_score = rouge(inputs, generated)
    blue_score = bleu(inputs, generated)

    return {
        "rouge1": rouge_score["rouge1"].item(),
        "rouge2": rouge_score["rouge2"].item(),
        "rougeL": rouge_score["rougeL"].item(),
        "bleu": blue_score["bleu"].item(),
    }


def ensure_directory(directory: str, clear: bool = True):
    """
    Create a directory and parents if it doesn't exist, and clear it if it does.
    """
    Path(directory).mkdir(exist_ok=True, parents=True)
    if clear:
        shutil.rmtree(directory)
    Path(directory).mkdir(exist_ok=True, parents=True)
