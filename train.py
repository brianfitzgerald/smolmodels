print("Loading dependencies - torch...")
import torch
import torch.nn.functional as F
from typing import Optional
from fire import Fire
from tabulate import tabulate
import pandas as pd
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from enum import Enum
from pathlib import Path
import shutil
from fsspec.core import url_to_fs

from model.t5 import T5FineTuner
from model.llama import LlamaFineTuner

print("Loading dependencies - lightning...")
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl

print("Loading dependencies - project...")
from model.data import PromptUpsampleDataModule
from model.utils import HyperParams


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
        labels = batch["labels"]
        labels[labels[:, :] == -100] = pl_module.model.config.pad_token_id
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
            table_rows = list(zip(*table_columns))
            self.wandb_logger.log_table("Validation Samples", columns, table_rows)

        rows = [list(row) for row in zip(*table_columns)]
        rows_df = pd.DataFrame(rows, columns=columns)
        rows_df.to_csv(
            self.log_dir / f"{run_name}_samples.csv",
            mode="a",
            header=False,
            index=False,
        )

        new_rows = tabulate(
            rows,
            headers=columns,
            maxcolwidths=[10, 10, 100, 100, 100],
        )
        with open(self.log_dir / f"{run_name}_samples.txt", "a") as f:
            f.write(new_rows)
            f.write("\n")


class ModelChoice(Enum):
    T5 = "t5"
    LLAMA = "llama"


class HfModelCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        hf_save_dir = filepath + ".dir"
        if trainer.is_global_zero:
            trainer.lightning_module.model.save_pretrained(hf_save_dir)
            trainer.lightning_module.tokenizer.save_pretrained(hf_save_dir)

    # https://github.com/Lightning-AI/lightning/pull/16067
    def _remove_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        hf_save_dir = filepath + ".dir"
        if trainer.is_global_zero:
            fs, _ = url_to_fs(hf_save_dir)
            if fs.exists(hf_save_dir):
                fs.rm(hf_save_dir, recursive=True)


def main(wandb: bool = False, model_choice: str = ModelChoice.T5.value):
    params = HyperParams()
    loggers = []

    model = None
    if model_choice == ModelChoice.LLAMA.value:
        model = LlamaFineTuner(params, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    elif model_choice == ModelChoice.T5.value:
        model = T5FineTuner(params, "google/flan-t5-small")
    assert model

    wandb_logger = None

    if wandb:
        project_name = f"{model_choice}_prompt_upsample"
        wandb_logger = WandbLogger(project=project_name)
        loggers.append(wandb_logger)
        wandb_logger.watch(model)

    is_sequence_to_sequence = model_choice == ModelChoice.T5.value

    dm = PromptUpsampleDataModule(
        model.tokenizer,
        batch_size=params.train_batch_size,
        max_token_length=params.max_seq_length,
        sequence_to_sequence=is_sequence_to_sequence,
    )

    sample_callback = LogPredictionSamplesCallback(
        model.tokenizer, wandb_logger, params.max_seq_length
    )

    checkpoint_callback = HfModelCheckpoint(
        dirpath="checkpoints",
        filename="best_model",
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        accumulate_grad_batches=params.gradient_accumulation_steps,
        max_epochs=params.num_train_epochs,
        precision=16 if params.fp_16 else 32,
        gradient_clip_val=params.max_grad_norm,
        val_check_interval=0.1,
        callbacks=[sample_callback, checkpoint_callback],
        logger=loggers,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    Fire(main)
