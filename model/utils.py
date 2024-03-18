from typing import List, Literal
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore
from pathlib import Path
import shutil
from dataclasses import dataclass

PROMPT_EXPANSION_TASK_PREFIX = "Expand the following prompt to add more detail: "
SAFETY_TASK_PREFIX = (
    "Rewrite the following prompt to remove any unsafe or copyrighted content: "
)
IGNORE_TOKEN_INDEX = -100
PAD_TOKEN_ID = 0

OptimizerChoice = Literal["AdamW", "Adafactor", "AdamW8bit"]


@dataclass
class HyperParams:
    base_model_checkpoint: str = "google/flan-t5-small"
    max_seq_length: int = 2048
    learning_rate: float = 3e-4
    adam_epsilon: float = 1e-8
    warmup_steps: int = 100
    train_batch_size: int = 4
    eval_batch_size: int = 2
    num_train_epochs: int = 25
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    seed: int = 42
    weight_decay: float = 0.0
    optimizer: OptimizerChoice = "AdamW8bit"



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
