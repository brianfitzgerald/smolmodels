from typing import Optional
from tabulate import tabulate
import pandas as pd
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from pathlib import Path
import shutil
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl
from fsspec.core import url_to_fs

from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl


from model.utils import (
    IGNORE_TOKEN_INDEX,
    PAD_TOKEN_ID,
    compute_metrics,
)


class LogPredictionSamplesCallback(pl.Callback):
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        wandb_logger: Optional[WandbLogger] = None,
        max_new_tokens: int = 256,
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
            maxcolwidths=[10, 10, 50, 50, 50],
        )
        print(new_rows)
        with open(self.log_dir / f"{run_name}_samples.txt", "a") as f:
            f.write(new_rows)
            f.write("\n")


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
