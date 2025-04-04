from loguru import logger
from typing import Optional

import random
import string
from dataclasses import dataclass

from dotenv import load_dotenv
from fire import Fire
from lightning import seed_everything
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

logger.info("Loading dependencies - Lightning...")
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger

logger.info("Loading dependencies - Project...")
from dataset.function_calling import FunctionCallingDataModule
from dataset.squad import (
    SquadExtractiveQADataModule,
    SquadDataModule,
    DollyEntityExtractionDataModule,
    UltraFeedbackDataModule,
)
from dataset.parti import PromptUpsampleDataModule
from dataset.pretrain import BertPretrainDataset, TinyStoriesDataset
from model.callbacks import (
    GradientNormLogger,
    HfModelCheckpoint,
    LogLLMPredictionSamplesCallback,
)
from model.causal_lm import AutoLMFineTuner
from model.pretrain.bert import SimpleBertForMaskedLM, get_sane_normalizers
from model.pretrain.gpt import GPT
from model.t5 import T5FineTuner
from model.utils import LMHyperParams, ModelChoice, SmDataset, SmModel


@dataclass
class ModelConfig:
    model: type[SmModel]
    data_module: type[SmDataset]
    wandb_project_name: str
    hyperparams: LMHyperParams


PROMPT_UPSAMPLING_PROJECT = "t5-prompt-upsampling"
PROMPT_SAFETY_PROJECT = "t5-prompt-safety"
EXTRACTIVE_QA_PROJECT = "t5-extractive-qa"
SQUAD_QA_PROJECT = "t5-squad-qa"

SMOL_LM_HPARAMS = LMHyperParams(
    base_model_checkpoint="microsoft/Phi-3-mini-128k-instruct",
    learning_rate=2e-05,
    warmup_ratio=0.1,
    optimizer="AdamW",
    train_batch_size=1,
    val_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    max_seq_length=2048,
)

CONFIGS = {
    "fn_calling": ModelConfig(
        AutoLMFineTuner,
        FunctionCallingDataModule,
        "llama-function-calling",
        LMHyperParams("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ),
    "prompt_upsample_small": ModelConfig(
        T5FineTuner,
        PromptUpsampleDataModule,
        PROMPT_UPSAMPLING_PROJECT,
        LMHyperParams(base_model_checkpoint="google/flan-t5-small"),
    ),
    "prompt_upsample": ModelConfig(
        T5FineTuner,
        PromptUpsampleDataModule,
        PROMPT_UPSAMPLING_PROJECT,
        LMHyperParams(base_model_checkpoint="google/flan-t5-base"),
    ),
    "simple_bert_pretrain": ModelConfig(
        SimpleBertForMaskedLM,
        BertPretrainDataset,
        "simple-bert-pretrain",
        LMHyperParams(
            # base model is only used for tokenizer
            base_model_checkpoint="bert-base-uncased",
            learning_rate=1e-3,
            warmup_ratio=0.5,
            weight_decay=0.01,
            max_grad_norm=0.5,
            num_train_epochs=1,
            train_batch_size=128,
            gradient_accumulation_steps=16,
            max_seq_length=512,
        ),
    ),
    "tiny_stories": ModelConfig(
        GPT,
        TinyStoriesDataset,
        "tinystories-gpt-pretrain",
        LMHyperParams(
            learning_rate=1e-4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            max_grad_norm=0.5,
            num_train_epochs=1,
            train_batch_size=32,
            val_batch_size=1,
            gradient_accumulation_steps=16,
            max_seq_length=512,
            tokenizer_checkpoint="EleutherAI/gpt-neox-20b",
        ),
    ),
    "info_extraction": ModelConfig(
        T5FineTuner,
        SquadExtractiveQADataModule,
        EXTRACTIVE_QA_PROJECT,
        LMHyperParams(
            base_model_checkpoint="google/flan-t5-base",
            warmup_ratio=0.2,
            optimizer="Adafactor",
        ),
    ),
    "t5_squad": ModelConfig(
        T5FineTuner,
        SquadDataModule,
        SQUAD_QA_PROJECT,
        LMHyperParams(
            base_model_checkpoint="google/flan-t5-base",
            learning_rate=1e-3,
            adam_epsilon=1e-30,
            warmup_ratio=0.1,
            optimizer="Adafactor",
            train_batch_size=32,
            val_batch_size=16,
            gradient_accumulation_steps=4,
            num_train_epochs=10,
            max_seq_length=512,
        ),
    ),
    # https://huggingface.co/blog/smollm
    # https://github.com/huggingface/alignment-handbook/blob/main/recipes/zephyr-7b-gemma/sft/config_full.yaml
    "smol_squad": ModelConfig(
        AutoLMFineTuner, SquadDataModule, "smollm-1.7b-squad", SMOL_LM_HPARAMS
    ),
    "smol_dolly": ModelConfig(
        AutoLMFineTuner,
        DollyEntityExtractionDataModule,
        "phi-dolly-ie",
        SMOL_LM_HPARAMS,
    ),
    "qwen_dpo": ModelConfig(
        AutoLMFineTuner,
        UltraFeedbackDataModule,
        "qwen-dpo",
        LMHyperParams(
            base_model_checkpoint="Qwen/Qwen2.5-0.5B",
            warmup_steps_count=10,
            tuning_type="dpo",
            train_batch_size=1,
            gradient_accumulation_steps=1,
            optimizer="AdamW",
        ),
    ),
}


def main(
    wandb: bool = False,
    config: str = "qwen_dpo",
    run_name: Optional[str] = None,
    **kwargs,
):
    assert not kwargs, f"Unknown arguments: {kwargs}"

    load_dotenv(".env")
    model_config = CONFIGS[config]
    hparams = model_config.hyperparams

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(  # type: ignore
        hparams.tokenizer_checkpoint_value
    )
    data_module = model_config.data_module(
        hparams.train_batch_size, tokenizer, hparams.max_seq_length
    )
    model = model_config.model(hparams)
    precision = "32" if model_config.model == T5FineTuner else "16-mixed"
    suffix = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    if run_name:
        run_name = f"{config}-{run_name}-{suffix}"
    else:
        run_name = f"{config}-{suffix}"

    start_training(
        precision,
        run_name,
        model,
        tokenizer,
        data_module,
        hparams,
        wandb,
        model_config.wandb_project_name,
    )


def start_training(
    precision: str,
    run_name: str,
    model: pl.LightningModule,
    tokenizer: PreTrainedTokenizer,
    data_module: pl.LightningDataModule,
    hparams: LMHyperParams,
    wandb: bool = False,
    project_name: Optional[str] = None,
):
    """
    Main fit function, split out to allow invoking from a notebook
    """
    loggers = []

    wandb_logger = None

    if wandb:
        wandb_logger = WandbLogger(name=run_name, project=project_name)
        loggers.append(wandb_logger)
        wandb_logger.watch(model)
    else:
        loggers.append(CSVLogger("logs", name=run_name))

    model_choice: ModelChoice = model.model_choice
    sample_callback = LogLLMPredictionSamplesCallback(
        tokenizer, model_choice, wandb_logger
    )

    learning_rate_callback = LearningRateMonitor(logging_interval="step")
    gradient_norm_callback = GradientNormLogger()

    if model_choice == ModelChoice.SIMPLE_BERT:
        tokenizer._tokenizer.normalizer = get_sane_normalizers(  # type: ignore
            force_english_keyboard=True,
            strip_accents=True,
            force_lowercase=True,
        )

    if model_choice == ModelChoice.GPT:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    seed_everything(hparams.seed)

    if model_choice in (ModelChoice.SIMPLE_BERT, ModelChoice.GPT):
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename=run_name,
            monitor="val_loss",
            mode="min",
        )
    else:
        checkpoint_callback = HfModelCheckpoint(
            dirpath="checkpoints",
            filename=run_name,
            monitor="val_loss",
            mode="min",
        )

    progress_bar_callback = TQDMProgressBar(refresh_rate=1)

    effective_batch_size = (
        hparams.train_batch_size * hparams.gradient_accumulation_steps
    )
    logger.info(
        f"Effective batch size: {effective_batch_size} ({hparams.gradient_accumulation_steps}x{hparams.train_batch_size})"
    )

    trainer = pl.Trainer(
        accumulate_grad_batches=hparams.gradient_accumulation_steps,
        max_epochs=hparams.num_train_epochs,
        precision=precision,  # type: ignore
        gradient_clip_val=hparams.max_grad_norm,
        val_check_interval=0.1,
        callbacks=[
            sample_callback,
            checkpoint_callback,
            progress_bar_callback,
            learning_rate_callback,
            gradient_norm_callback,
        ],
        logger=loggers,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    Fire(main)
