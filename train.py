print("Loading dependencies - torch...")
import torch
import torch.nn.functional as F
from typing import Optional
import fire
from tabulate import tabulate
import pandas as pd
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from enum import Enum
from pathlib import Path
import shutil

from model.t5 import T5FineTuner
from model.llama import LlamaFineTuner

print("Loading dependencies - lightning...")
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

print("Loading dependencies - project...")
from model.data import PromptUpsampleDataModule
from model.params import HyperParams


def calculate_bpc(model, evaluation_data):
    """
    Bits per character
    """
    total_loss = 0.0
    total_characters = 0

    model.eval()

    with torch.no_grad():
        for input_seq, target_seq in evaluation_data:
            input_seq = torch.tensor(input_seq).unsqueeze(0)
            target_seq = torch.tensor(target_seq).unsqueeze(0)

            output_seq = model(input_seq)
            output_seq = output_seq.squeeze(0)

            loss = F.cross_entropy(output_seq, target_seq)
            total_loss += loss.item()
            total_characters += target_seq.size(1)

    average_loss = total_loss / total_characters
    bpc = average_loss / torch.log(torch.tensor(2.0))

    return bpc.item()


class LogPredictionSamplesCallback(pl.Callback):
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        wandb_logger: Optional[WandbLogger] = None,
        max_new_tokens: int = 100,
    ):
        self.tokenizer = tokenizer
        self.wandb_logger = wandb_logger
        self.max_new_tokens = max_new_tokens

        # TODO clear existing log files
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        shutil.rmtree(self.log_dir)
        self.log_dir.mkdir(exist_ok=True)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        self.log_prediction_samples(trainer, pl_module, outputs, batch, batch_idx, 0)

    def log_prediction_samples(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx > 0:
            return
        input_ids = batch["input_ids"]
        labels = batch["input_ids"]
        out = pl_module.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
        )

        n = len(input_ids)
        columns = ["Epoch", "Sample Index", "Input", "Output", "Target"]
        table_columns = []
        table_columns.append([trainer.current_epoch] * n)
        table_columns.append(list(range(n)))

        for feature in [input_ids, out, labels]:
            decoded = self.tokenizer.batch_decode(
                feature, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            table_columns.append(decoded)

        run_name = "latest"
        if self.wandb_logger:
            run_name = self.wandb_logger.experiment.name
            self.wandb_logger.log_table("Validation Samples", columns, table_columns)

        rows = [list(row) for row in zip(*table_columns)]
        rows_df = pd.DataFrame(rows, columns=columns)
        rows_df.to_csv(
            self.log_dir / f"{run_name}_samples.csv", mode="a", header=False, index=False
        )

        new_rows = tabulate(
            rows,
            headers=columns,
            tablefmt="simple_outline",
            maxcolwidths=[10, 10, 100, 100, 100],
        )
        with open(self.log_dir / f"{run_name}_samples.txt", "a") as f:
            f.write(new_rows)
            f.write("\n")


class ModelChoice(Enum):
    T5 = "t5"
    LLAMA = "llama"


def main(wandb: bool = False, model_choice: str = ModelChoice.LLAMA.value):
    params = HyperParams()
    loggers = []

    model = None
    if model_choice == ModelChoice.LLAMA.value:
        model = LlamaFineTuner(params, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    elif model_choice == ModelChoice.T5.value:
        model = T5FineTuner(params, "google/flan-t5-base")
    assert model

    wandb_logger = None

    if wandb:
        wandb_logger = WandbLogger(project="t5-upsampled-prompts")
        loggers.append(wandb_logger)
        wandb_logger.watch(model)

    dm = PromptUpsampleDataModule(
        model.tokenizer,
        batch_size=params.train_batch_size,
        max_token_length=params.max_seq_length,
    )

    trainer = pl.Trainer(
        accumulate_grad_batches=params.gradient_accumulation_steps,
        max_epochs=params.num_train_epochs,
        precision=16 if params.fp_16 else 32,
        gradient_clip_val=params.max_grad_norm,
        val_check_interval=0.1,
        callbacks=[
            LogPredictionSamplesCallback(
                model.tokenizer, wandb_logger, params.max_seq_length
            )
        ],
        logger=loggers,
    )
    trainer.fit(model, datamodule=dm)


fire.Fire(main)
