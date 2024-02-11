from typing import List
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore

from torch import Tensor

TASK_PREFIX = "Expand the following prompt to add more detail: "


class HyperParams:
    max_seq_length: int = 256
    learning_rate: float = 1e-4
    adam_epsilon: float = 1e-8
    warmup_steps: int = 50
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_train_epochs: int = 25
    gradient_accumulation_steps: int = 2
    n_gpus: int = 1
    fp_16: bool = False
    max_grad_norm: float = 10.0
    seed: int = 42
    weight_decay: float = 0.0


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
