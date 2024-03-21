from typing import Optional, List
from fire import Fire
from tabulate import tabulate
import pandas as pd
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from enum import Enum
from pathlib import Path
from fsspec.core import url_to_fs
from dataclasses import dataclass
from dataset.prompt_classifier import (
    ClipdropSyntheticClassesDataModule,
    ClipdropBinaryDataModule,
    ClipdropSafetyFamousFiguresDataModule,
)
from dataset.function_calling import FunctionCallingDataModule
import random
import string
import torch.multiprocessing
from weakref import proxy
from typing import Dict
import json


from model.t5 import T5Model
from model.roberta import RobertaClassifier, RobertaClassifierMultilabel
from model.llama import LlamaFineTuner
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    TQDMProgressBar,
    LearningRateMonitor,
    EarlyStopping,
)
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

torch.multiprocessing.set_sharing_strategy("file_system")


class LogPredictionSamplesCallback(pl.Callback):
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        run_name: str,
        wandb_logger: Optional[WandbLogger] = None,
        max_new_tokens: int = 256,
    ):
        self.tokenizer = tokenizer
        self.wandb_logger = wandb_logger
        self.max_new_tokens = max_new_tokens
        self.run_name = run_name

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
        labels_dict: Dict[int, str] = pl_module.id_to_labels
        h_params: HyperParams = pl_module.params
        column_names = ["Epoch", "Sample Index", "Prompt", "Expected", "Predicted"]

        n = len(input_ids)
        if h_params.objective == "binary_classification":
            out = pl_module.model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_classes = out.logits.argmax(dim=-1).cpu().numpy().tolist()
            labels_list = labels.cpu().numpy().tolist()

            predicted_class_names = [labels_dict[i] for i in predicted_classes]
            label_class_names = [labels_dict[i] for i in labels_list]

            decoded_prompts = self.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            feature_columns = [
                [trainer.current_epoch] * n,
                list(range(n)),
                decoded_prompts,
                label_class_names,
                predicted_class_names,
            ]

        elif h_params.objective == "multilabel_classification":
            out = pl_module.model(input_ids=input_ids, attention_mask=attention_mask)
            labels = labels.cpu().numpy().tolist()
            predicted_classes = torch.sigmoid(out.logits).squeeze(dim=0)
            predicted_classes = predicted_classes.cpu().numpy().tolist()

            decoded_prompts = self.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            labels_str = [json.dumps(label) for label in labels]
            predicted_classes_str = [
                json.dumps([round(l, 2) for l in label]) for label in predicted_classes
            ]

            feature_columns = [
                [trainer.current_epoch] * n,
                list(range(n)),
                decoded_prompts,
                labels_str,
                predicted_classes_str,
            ]

        else:
            labels[labels[:, :] == IGNORE_TOKEN_INDEX] = PAD_TOKEN_ID
            out = pl_module.model.generate(
                input_ids,
                max_length=self.max_new_tokens,
            )

            feature_columns = []
            feature_columns.append([trainer.current_epoch] * n)
            feature_columns.append(list(range(n)))

            for feature in [input_ids, labels, out]:
                decoded = self.tokenizer.batch_decode(
                    feature, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                feature_columns.append(decoded)

        if self.wandb_logger:
            table_rows = list(zip(*feature_columns))
            self.wandb_logger.log_table("samples/validation", column_names, table_rows)

        rows = [list(row) for row in zip(*feature_columns)]
        rows_df = pd.DataFrame(rows, columns=column_names)
        rows_df.to_csv(
            self.log_dir / f"{self.run_name}_samples.csv",
            mode="a",
            header=False,
            index=False,
        )

        new_rows = tabulate(
            rows,
            tablefmt="simple_grid",
            headers=column_names,
            maxcolwidths=[10, 10, 50, 50, 50],
        )
        print(new_rows)
        with open(self.log_dir / f"{self.run_name}_samples.txt", "a") as f:
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


class GradientNormLogger(pl.Callback):
    """
    Logs the gradient norm.
    """

    def on_after_backward(self, trainer, model):
        model.log("grad_norm_total", gradient_norm(model))


def gradient_norm(model: pl.LightningModule):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


@dataclass
class ModelConfig:
    model: type[pl.LightningModule]
    data_module: type[FineTunerDataset]
    project_name: str
    hyperparams: HyperParams = HyperParams()
    task_prefix: str = PROMPT_EXPANSION_TASK_PREFIX


PROMPT_UPSAMPLING_PROJECT = "t5-prompt-upsampling"
PROMPT_SAFETY_PROJECT = "t5-prompt-safety"
PROMPT_CLASSIFIER_PROJECT = "roberta-prompt-safety-classifier"
BINARY_CLASSIFIER_PROJECT = "roberta-safety-classifier-binary"

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
    "saferprompt": ModelConfig(
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
    ),
    "safety_classifier_synthetic": ModelConfig(
        RobertaClassifier,
        ClipdropSyntheticClassesDataModule,
        PROMPT_CLASSIFIER_PROJECT,
        HyperParams(
            base_model_checkpoint="distilbert/distilroberta-base",
            num_train_epochs=25,
            learning_rate=1e-4,
            max_seq_length=512,
            labels_set="clipdrop_binary",
            objective="binary_classification",
        ),
    ),
    "safety_classifier_binary": ModelConfig(
        RobertaClassifier,
        ClipdropBinaryDataModule,
        BINARY_CLASSIFIER_PROJECT,
        HyperParams(
            base_model_checkpoint="distilbert/distilroberta-base",
            num_train_epochs=50,
            learning_rate=1e-7,
            max_seq_length=512,
            labels_set="clipdrop_binary",
            objective="binary_classification",
        ),
    ),
    "safety_classifier_multilabel": ModelConfig(
        RobertaClassifierMultilabel,
        ClipdropSafetyFamousFiguresDataModule,
        "roberta-safety-classifier-multilabel",
        HyperParams(
            base_model_checkpoint="distilbert/distilroberta-base",
            num_train_epochs=100,
            learning_rate=1e-5,
            max_seq_length=512,
            labels_set="clipdrop_multilabel",
            objective="multilabel_classification",
        ),
    ),
}


def main(
    wandb: bool = False,
    distributed: bool = False,
    config: str = "safety_classifier_binary",
    **kwargs,
):
    assert not kwargs, f"Unrecognized arguments: {kwargs}"

    torch.manual_seed(42)

    model_config = CONFIGS[config]
    hparams = model_config.hyperparams
    model = model_config.model(hparams)
    data_module = model_config.data_module(
        hparams.train_batch_size, model.tokenizer, hparams.max_seq_length
    )

    wandb_logger = None
    run_name = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    run_name = f"{config}-{run_name}"

    loggers: List[Logger] = []

    if wandb:
        project_name = model_config.project_name
        wandb_logger = WandbLogger(name=run_name, project=project_name)
        loggers.append(wandb_logger)

    ensure_directory("logs", clear=True)
    sample_callback = LogPredictionSamplesCallback(
        model.tokenizer, run_name, wandb_logger
    )

    quality_metric = "val_loss_epoch"

    checkpoint_callback = HfModelCheckpoint(
        dirpath="/weka/home-brianf/smolmodels_checkpoints",
        filename=run_name,
        monitor=quality_metric,
        mode="max",
    )

    grad_norm_callback = GradientNormLogger()

    progress_bar_callback = TQDMProgressBar(refresh_rate=10)
    lr_monitor_callback = LearningRateMonitor(logging_interval="step")
    early_stopping_callback = EarlyStopping(
        monitor=quality_metric, patience=5, mode="min", check_finite=True
    )

    precision = "32" if model_config.model == T5Model else "16-mixed"

    strategy = "ddp" if distributed else "auto"
    callbacks = [
        sample_callback,
        checkpoint_callback,
        progress_bar_callback,
        early_stopping_callback,
        grad_norm_callback,
    ]
    if wandb:
        callbacks.append(lr_monitor_callback)

    trainer = pl.Trainer(
        accumulate_grad_batches=hparams.gradient_accumulation_steps,
        max_epochs=hparams.num_train_epochs,
        precision=precision,
        gradient_clip_val=hparams.max_grad_norm,
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_algorithm="value",
        log_every_n_steps=1,
        num_nodes=1,
        devices="auto",
        plugins=LightningEnvironment(),
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    Fire(main)
