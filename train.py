print("Loading dependencies - torch...")
from typing import Optional
from fire import Fire
from tabulate import tabulate
import pandas as pd
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from enum import Enum
from pathlib import Path
import shutil
from fsspec.core import url_to_fs
from dataclasses import dataclass
from dataset.function_calling import FunctionCallingDataModule

from model.t5 import T5FineTuner
from model.llama import LlamaFineTuner

print("Loading dependencies - lightning...")
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar


print("Loading dependencies - project...")
from dataset.parti import PromptSafetyDataModule, PromptUpsampleDataModule
from model.utils import IGNORE_TOKEN_INDEX, PAD_TOKEN_ID, HyperParams, FineTunerDataset, compute_metrics


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
        labels[labels[:, :] == IGNORE_TOKEN_INDEX] = PAD_TOKEN_ID
        out = pl_module.model.generate(
            input_ids,
            max_length=self.max_new_tokens,
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

        metrics = compute_metrics(table_columns[3], table_columns[4])

        run_name = "latest"
        if self.wandb_logger:
            run_name = self.wandb_logger.experiment.name
            table_rows = list(zip(*table_columns))
            self.wandb_logger.log_table("Validation Samples", columns, table_rows)
            self.wandb_logger.log_metrics(metrics)

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


# https://github.com/Lightning-AI/pytorch-lightning/issues/3096#issuecomment-1441278197
class HfModelCheckpoint(ModelCheckpoint):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.FILE_EXTENSION = ""

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        print(f"Saving checkpoint at {filepath}")
        if trainer.is_global_zero:
            trainer.lightning_module.model.save_pretrained(filepath)
            trainer.lightning_module.tokenizer.save_pretrained(filepath)

    # https://github.com/Lightning-AI/lightning/pull/16067
    def _remove_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        print(f"Removing checkpoint at {filepath}")
        if trainer.is_global_zero:
            fs, _ = url_to_fs(filepath)
            if fs.exists(filepath):
                fs.rm(filepath, recursive=True)


@dataclass
class ModelConfig:
    model: type[pl.LightningModule]
    data_module: type[FineTunerDataset]
    wandb_project_name: str


CONFIGS = {
    "fn_calling": ModelConfig(
        LlamaFineTuner, FunctionCallingDataModule, "llama-function-calling"
    ),
    "prompt_upsample": ModelConfig(
        T5FineTuner, PromptUpsampleDataModule, "t5-prompt-upsampling"
    ),
    "prompt_safety": ModelConfig(
        T5FineTuner, PromptSafetyDataModule, "t5-prompt-safety"
    ),
}


def main(wandb: bool = False, config: str = "prompt_safety"):
    params = HyperParams()
    loggers = []

    model_config = CONFIGS[config]
    model = model_config.model(params)
    data_module = model_config.data_module(
        params.train_batch_size, model.tokenizer, params.max_seq_length
    )

    wandb_logger = None

    if wandb:
        project_name = model_config.wandb_project_name
        wandb_logger = WandbLogger(project=project_name)
        loggers.append(wandb_logger)
        wandb_logger.watch(model)

    sample_callback = LogPredictionSamplesCallback(
        model.tokenizer, wandb_logger, params.max_seq_length
    )

    checkpoint_callback = HfModelCheckpoint(
        dirpath="checkpoints",
        filename="best_model",
        monitor="val_loss",
        mode="min",
    )

    progress_bar_callback = TQDMProgressBar(refresh_rate=10)

    trainer = pl.Trainer(
        accumulate_grad_batches=params.gradient_accumulation_steps,
        max_epochs=params.num_train_epochs,
        precision="16",
        gradient_clip_val=params.max_grad_norm,
        # val_check_interval=0.1,
        callbacks=[sample_callback, checkpoint_callback, progress_bar_callback],
        logger=loggers,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    Fire(main)
