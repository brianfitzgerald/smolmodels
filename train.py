print("Loading dependencies - torch...")
from fire import Fire
from enum import Enum
from dataclasses import dataclass

from lightning import seed_everything
from dataset.function_calling import FunctionCallingDataModule
import random
import string

from model.t5 import T5FineTuner
from model.llama import LlamaFineTuner
from model.simple_bert import SimpleBertForMaskedLM

print("Loading dependencies - lightning...")
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar
from model.callbacks import LogPredictionSamplesCallback, HfModelCheckpoint


print("Loading dependencies - project...")
from dataset.parti import PromptUpsampleDataModule
from dataset.bert_pretrain import BertPretrainDataset
from model.utils import SmDataset, HyperParams


class ModelChoice(Enum):
    T5 = "t5"
    LLAMA = "llama"
    SIMPLE_BERT = "simple_bert"


@dataclass
class ModelConfig:
    model: type[pl.LightningModule]
    data_module: type[SmDataset]
    wandb_project_name: str
    hyperparams: HyperParams = HyperParams()


PROMPT_UPSAMPLING_PROJECT = "t5-prompt-upsampling"
PROMPT_SAFETY_PROJECT = "t5-prompt-safety"
BERT_PRETRAIN_PROJECT = "simple-bert-pretrain"

CONFIGS = {
    "fn_calling": ModelConfig(
        LlamaFineTuner,
        FunctionCallingDataModule,
        "llama-function-calling",
        HyperParams("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ),
    "prompt_upsample_small": ModelConfig(
        T5FineTuner,
        PromptUpsampleDataModule,
        PROMPT_UPSAMPLING_PROJECT,
        HyperParams(base_model_checkpoint="google/flan-t5-small"),
    ),
    "prompt_upsample": ModelConfig(
        T5FineTuner,
        PromptUpsampleDataModule,
        PROMPT_UPSAMPLING_PROJECT,
        HyperParams(base_model_checkpoint="google/flan-t5-base"),
    ),
    "simple_bert_pretrain": ModelConfig(
        SimpleBertForMaskedLM,
        BertPretrainDataset,
        BERT_PRETRAIN_PROJECT,
        HyperParams(base_model_checkpoint="bert-base-uncased", max_seq_length=128),
    ),
}


def main(wandb: bool = False, config: str = "simple_bert_pretrain"):
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

    sample_callback = LogPredictionSamplesCallback(model.tokenizer, wandb_logger)
    seed_everything(hparams.seed)

    checkpoint_callback = HfModelCheckpoint(
        dirpath="checkpoints",
        filename=run_name,
        monitor="val_loss",
        mode="min",
    )

    progress_bar_callback = TQDMProgressBar(refresh_rate=10)
    precision = "32" if model_config.model == T5FineTuner else "16-mixed"

    trainer = pl.Trainer(
        accumulate_grad_batches=hparams.gradient_accumulation_steps,
        max_epochs=hparams.num_train_epochs,
        precision=precision,
        gradient_clip_val=hparams.max_grad_norm,
        val_check_interval=0.01,
        callbacks=[sample_callback, checkpoint_callback, progress_bar_callback],
        logger=loggers,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    Fire(main)
