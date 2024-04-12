print("Loading dependencies - torch...")
from fire import Fire
from dataclasses import dataclass

from lightning import seed_everything
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from dataset.function_calling import FunctionCallingDataModule
import random
import string

from model.t5 import T5FineTuner
from model.llama import LlamaFineTuner
from model.simple_bert import SimpleBertForMaskedLM
from model.utils import SmModel

print("Loading dependencies - lightning...")
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar
from model.callbacks import LogPredictionSamplesCallback, HfModelCheckpoint
from lightning.pytorch.callbacks import ModelCheckpoint


print("Loading dependencies - project...")
from dataset.parti import PromptUpsampleDataModule
from dataset.bert_pretrain import BertPretrainDataset
from model.utils import ModelChoice, SmDataset, HyperParams


MODEL_CHOICES = {
    SimpleBertForMaskedLM: ModelChoice.SIMPLE_BERT,
    T5FineTuner: ModelChoice.T5,
    LlamaFineTuner: ModelChoice.LLAMA,
}


@dataclass
class ModelConfig:
    model: type[SmModel]
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
        HyperParams(
            # base model is only used for tokenizer
            base_model_checkpoint="bert-base-uncased",
            learning_rate=1e-3,
            warmup_ratio=0.5,
            weight_decay=0.01,
            max_grad_norm=0.5,
            num_train_epochs=1,
            train_batch_size=24,
            gradient_accumulation_steps=96,
            max_seq_length=256,
        ),
    ),
}


def main(wandb: bool = False, config: str = "simple_bert_pretrain"):
    loggers = []

    model_config = CONFIGS[config]
    hparams = model_config.hyperparams
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(  # type: ignore
        hparams.tokenizer_checkpoint
    )
    data_module = model_config.data_module(
        hparams.train_batch_size, tokenizer, hparams.max_seq_length
    )
    model = model_config.model(hparams, tokenizer)

    wandb_logger = None
    run_name = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    run_name = f"{config}-{run_name}"

    if wandb:
        project_name = model_config.wandb_project_name
        wandb_logger = WandbLogger(name=run_name, project=project_name)
        loggers.append(wandb_logger)
        wandb_logger.watch(model)

    model_choice: ModelChoice = MODEL_CHOICES[model.__class__]  # type: ignore
    sample_callback = LogPredictionSamplesCallback(
        tokenizer, model_choice, wandb_logger
    )
    seed_everything(hparams.seed)

    if model_choice == ModelChoice.SIMPLE_BERT:
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

    progress_bar_callback = TQDMProgressBar(refresh_rate=10)
    precision = "32" if model_config.model == T5FineTuner else "16-mixed"

    trainer = pl.Trainer(
        accumulate_grad_batches=hparams.gradient_accumulation_steps,
        max_epochs=hparams.num_train_epochs,
        precision=precision,
        gradient_clip_val=hparams.max_grad_norm,
        val_check_interval=0.1,
        callbacks=[sample_callback, checkpoint_callback, progress_bar_callback],
        logger=loggers,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    Fire(main)
