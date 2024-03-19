from typing import Optional
from fire import Fire
from tabulate import tabulate
import pandas as pd
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from enum import Enum
from pathlib import Path
from fsspec.core import url_to_fs
from dataclasses import dataclass
from dataset.prompt_classifier import PromptClassifierDataModule
from dataset.function_calling import FunctionCallingDataModule
import random
import string
import torch.multiprocessing
from weakref import proxy


from model.t5 import T5Model
from model.roberta import RobertaClassifier
from model.llama import LlamaFineTuner

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar, LearningRateMonitor
from lightning.fabric.plugins.environments.lightning import LightningEnvironment


from dataset.prompt_upsample import PromptSafetyDataModule, PromptUpsampleDataModule
from model.utils import (
    IGNORE_TOKEN_INDEX,
    PAD_TOKEN_ID,
    HyperParams,
    ensure_directory,
    PROMPT_EXPANSION_TASK_PREFIX,
    SAFETY_TASK_PREFIX,
)
from dataset.utils import FineTunerDataset
from synthetic_data.utils import SAFE_PROMPT_IDS_TO_LABEL_NAMES

torch.multiprocessing.set_sharing_strategy("file_system")


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

        self.log_dir = Path("logs")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        self.log_prediction_samples(trainer, pl_module, outputs, batch, batch_idx, 0)

    def log_prediction_samples(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """
        Log sample outputs. Metrics are logged in the training loop.
        """
        if batch_idx > 0:
            return

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        n = len(input_ids)
        if pl_module.params.objective == "classification":
            out = pl_module.model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_classes = out.logits.argmax(dim=-1).cpu().numpy().tolist()
            labels_list = labels.cpu().numpy().tolist()

            predicted_class_names = [
                SAFE_PROMPT_IDS_TO_LABEL_NAMES[i] for i in predicted_classes
            ]
            label_class_names = [SAFE_PROMPT_IDS_TO_LABEL_NAMES[i] for i in labels_list]

            column_names = ["Epoch", "Sample Index", "Prompt", "Expected", "Predicted"]

            feature_columns = []
            feature_columns.append([trainer.current_epoch] * n)
            feature_columns.append(list(range(n)))

            decoded = self.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            feature_columns.append(decoded)
            feature_columns.append(label_class_names)
            feature_columns.append(predicted_class_names)

        else:
            labels[labels[:, :] == IGNORE_TOKEN_INDEX] = PAD_TOKEN_ID
            out = pl_module.model.generate(
                input_ids,
                max_length=self.max_new_tokens,
            )

            column_names = ["Epoch", "Sample Index", "Input", "Output", "Target"]

            feature_columns = []
            feature_columns.append([trainer.current_epoch] * n)
            feature_columns.append(list(range(n)))

            for feature in [input_ids, out, labels]:
                decoded = self.tokenizer.batch_decode(
                    feature, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                feature_columns.append(decoded)

        run_name = "latest"
        if self.wandb_logger:
            run_name = self.wandb_logger.experiment.name
            table_rows = list(zip(*feature_columns))
            self.wandb_logger.log_table("Validation Samples", column_names, table_rows)

        rows = [list(row) for row in zip(*feature_columns)]
        rows_df = pd.DataFrame(rows, columns=column_names)
        rows_df.to_csv(
            self.log_dir / f"{run_name}_samples.csv",
            mode="a",
            header=False,
            index=False,
        )

        new_rows = tabulate(
            rows,
            headers=column_names,
            maxcolwidths=[10, 10, 50, 50, 50],
        )
        print(new_rows)
        with open(self.log_dir / f"{run_name}_samples.txt", "a") as f:
            f.write(new_rows)
            f.write("\n")


class ModelChoice(Enum):
    T5 = "t5"
    LLAMA = "llama"


# https://github.com/Lightning-AI/pytorch-lightning/issues/3096#issuecomment-1441278197
class HfModelCheckpoint(ModelCheckpoint):
    """
    Overrides default checkpoint saving behavior to instead save the HF model and tokenizer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.FILE_EXTENSION = ""

    def save_dir(self, filepath: str) -> str:
        return f"{filepath}.hf"

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))
        hf_save_dir = self.save_dir(filepath)
        if trainer.is_global_zero:
            trainer.lightning_module.model.save_pretrained(hf_save_dir)
            trainer.lightning_module.tokenizer.save_pretrained(hf_save_dir)

    # https://github.com/Lightning-AI/lightning/pull/16067
    def _remove_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        hf_save_dir = self.save_dir(filepath)
        if trainer.is_global_zero:
            fs, _ = url_to_fs(hf_save_dir)
            if fs.exists(hf_save_dir):
                fs.rm(hf_save_dir, recursive=True)


@dataclass
class ModelConfig:
    model: type[pl.LightningModule]
    data_module: type[FineTunerDataset]
    wandb_project_name: str
    hyperparams: HyperParams = HyperParams()
    task_prefix: str = PROMPT_EXPANSION_TASK_PREFIX
    ckpt_name: str = "superprompt-v1"


PROMPT_UPSAMPLING_PROJECT = "t5-prompt-upsampling"
PROMPT_SAFETY_PROJECT = "t5-prompt-safety"
PROMPT_CLASSIFIER_PROJECT = "roberta-prompt-safety-classifier"

CONFIGS = {
    "fn_calling": ModelConfig(
        LlamaFineTuner,
        FunctionCallingDataModule,
        "llama-function-calling",
        HyperParams("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ),
    "prompt_upsample_small": ModelConfig(
        T5Model,
        PromptUpsampleDataModule,
        PROMPT_UPSAMPLING_PROJECT,
        HyperParams(base_model_checkpoint="google/flan-t5-small"),
    ),
    "prompt_upsample": ModelConfig(
        T5Model,
        PromptUpsampleDataModule,
        PROMPT_UPSAMPLING_PROJECT,
        HyperParams(base_model_checkpoint="google/flan-t5-base"),
    ),
    "prompt_safety": ModelConfig(
        T5Model,
        PromptSafetyDataModule,
        PROMPT_SAFETY_PROJECT,
        HyperParams(
            base_model_checkpoint="google/flan-t5-base",
            gradient_accumulation_steps=4,
            train_batch_size=2,
            optimizer="AdamW8bit",
        ),
        task_prefix=SAFETY_TASK_PREFIX,
        ckpt_name="saferprompt-v1",
    ),
    "safety_classifier": ModelConfig(
        RobertaClassifier,
        PromptClassifierDataModule,
        PROMPT_CLASSIFIER_PROJECT,
        HyperParams(
            base_model_checkpoint="distilbert/distilroberta-base",
            train_batch_size=16,
            eval_batch_size=8,
            gradient_accumulation_steps=1,
            optimizer="AdamW",
            num_train_epochs=25,
            warmup_steps=1000,
            learning_rate=1e-4,
            adam_epsilon=1e-8,
            max_seq_length=512,
            objective="classification",
        ),
        ckpt_name="safer-prompt-classifier",
    ),
}


def main(
    wandb: bool = False,
    distributed: bool = False,
    config: str = "safety_classifier",
    **kwargs,
):
    assert not kwargs, f"Unrecognized arguments: {kwargs}"

    loggers = []

    model_config = CONFIGS[config]
    hparams = model_config.hyperparams
    model = model_config.model(hparams)
    data_module = model_config.data_module(
        hparams.train_batch_size, model.tokenizer, hparams.max_seq_length
    )

    wandb_logger = None
    run_name = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    run_name = f"{config}-{run_name}"

    if wandb:
        project_name = model_config.wandb_project_name
        wandb_logger = WandbLogger(name=run_name, project=project_name)
        loggers.append(wandb_logger)
        wandb_logger.watch(model)

    ensure_directory("logs", clear=True)
    sample_callback = LogPredictionSamplesCallback(model.tokenizer, wandb_logger)

    checkpoint_callback = HfModelCheckpoint(
        dirpath="checkpoints",
        filename=run_name,
        monitor="val_loss_epoch",
        mode="min",
    )

    progress_bar_callback = TQDMProgressBar(refresh_rate=10)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    precision = "32" if model_config.model == T5Model else "16-mixed"

    strategy = "ddp" if distributed else "auto"
    callbacks = [
        sample_callback,
        checkpoint_callback,
        progress_bar_callback,
    ]
    if wandb:
        callbacks.append(lr_monitor)

    trainer = pl.Trainer(
        accumulate_grad_batches=hparams.gradient_accumulation_steps,
        max_epochs=hparams.num_train_epochs,
        precision=precision,
        gradient_clip_val=hparams.max_grad_norm,
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_algorithm="norm",
        log_every_n_steps=1,
        num_nodes=1,
        devices="auto",
        plugins=LightningEnvironment(),
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    Fire(main)
