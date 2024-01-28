print("Loading torch")
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from typing import Optional
import fire
from torch import Tensor
from tabulate import tabulate

print("Loading lightning")
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

print("Loading HF")
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.optimization import get_inverse_sqrt_schedule
from t5_data import PromptUpsampleDataModule
from datasets import Dataset


class HyperParams:
    model_name: str = "google/flan-t5-small"
    max_seq_length: int = 512
    learning_rate: float = 1e-4
    adam_epsilon: float = 1e-8
    warmup_steps: int = 10
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_train_epochs: int = 2
    gradient_accumulation_steps: int = 4
    n_gpus: int = 1
    early_stop_callback: bool = False
    fp_16: bool = False
    max_grad_norm: float = 1.0
    seed: int = 42
    weight_decay: float = 0.0


def calculate_bpc(model, evaluation_data):
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


class T5FineTuner(pl.LightningModule):
    def __init__(self, params: HyperParams):
        super(T5FineTuner, self).__init__()
        self.params = params
        self.hparams.update(vars(params))
        self.save_hyperparameters()

        self.model: T5ForConditionalGeneration = (
            T5ForConditionalGeneration.from_pretrained(self.params.model_name)
        )
        self.tokenizer = T5Tokenizer.from_pretrained(self.params.model_name)

    def forward(self, input_ids, labels=None):
        out = self.model(
            input_ids,
            labels=labels,
        )
        return out

    def _step(self, batch):
        outputs = self(
            input_ids=batch["input_ids"],
            labels=batch["label_input_ids"],
        )

        return outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        # perplexity_score = get_perplexity(logits, input_ids)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"val_loss": loss}

    def on_valiation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {
            "avg_val_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        # emulates the original optimizer in https://github.com/google-research/bert/blob/master/optimization.py#L65
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.params.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.params.learning_rate,
            eps=self.params.adam_epsilon,
        )
        scheduler = get_inverse_sqrt_schedule(
            optimizer, num_warmup_steps=self.params.warmup_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }


class LogPredictionSamplesCallback(pl.Callback):
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        wandb_logger: Optional[WandbLogger] = None,
    ):
        self.tokenizer = tokenizer
        self.wandb_logger = wandb_logger

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        self.log_prediction_samples(trainer, pl_module, outputs, batch, batch_idx, 0)

    def log_prediction_samples(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        input_ids = batch["input_ids"]
        target_ids = batch["label_input_ids"]
        out = pl_module.model.generate(input_ids)

        columns = ["Input", "Output", "Target"]
        table_columns = []
        for feature in [input_ids, out, target_ids]:
            decoded = self.tokenizer.batch_decode(
                feature, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            table_columns.append(decoded)

        if self.wandb_logger:
            self.wandb_logger.log_table(
                "Prediction Samples",
            )
        else:
            table_rows = [list(row) for row in zip(*table_columns)]
            print(tabulate(table_rows, headers=columns))


def main(wandb: bool = False):
    params = HyperParams()
    loggers = []

    model = T5FineTuner(params)

    wandb_logger = None

    if wandb:
        wandb_logger = WandbLogger(project="t5-upsampled-prompts")
        loggers.append(wandb_logger)
        wandb_logger.watch(model)

    dm = PromptUpsampleDataModule(params.model_name, batch_size=8, max_token_length=512)
    trainer = pl.Trainer(
        accumulate_grad_batches=params.gradient_accumulation_steps,
        max_epochs=params.num_train_epochs,
        precision=16 if params.fp_16 else 32,
        gradient_clip_val=params.max_grad_norm,
        val_check_interval=0.25,
        callbacks=[
            LogPredictionSamplesCallback(
                model.tokenizer,
                wandb_logger,
            )
        ],
        logger=loggers,
    )
    trainer.fit(model, datamodule=dm)


fire.Fire(main)
