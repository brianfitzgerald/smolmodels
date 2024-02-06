print("Loading torch")
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from typing import Optional
import fire
from torch import Tensor
from tabulate import tabulate
import pandas as pd

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
    model_name: str = "google/flan-t5-base"
    max_seq_length: int = 256
    learning_rate: float = 2e-5
    adam_epsilon: float = 1e-8
    warmup_steps: int = 50
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_train_epochs: int = 25
    gradient_accumulation_steps: int = 2
    n_gpus: int = 1
    fp_16: bool = False
    max_grad_norm: float = 10.0
    seed: int = 42
    weight_decay: float = 0.0


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

    def forward(self, input_ids, attention_mask, labels):
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out

    def _step(self, batch):
        # print(f"input tokens: {self.tokenizer.decode(batch['input_ids'][0])}")
        # print(f"labels: {self.tokenizer.decode(batch['label_input_ids'][0])}")
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
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
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"val_loss": loss}

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
        max_new_tokens: int = 100,
    ):
        self.tokenizer = tokenizer
        self.wandb_logger = wandb_logger
        self.max_new_tokens = max_new_tokens

        # TODO clear existing log files

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        self.log_prediction_samples(trainer, pl_module, outputs, batch, batch_idx, 0)

    def log_prediction_samples(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        input_ids = batch["input_ids"]
        target_ids = batch["label_input_ids"]
        out = pl_module.model.generate(
            input_ids,
            max_length=self.max_new_tokens,
            max_new_tokens=self.max_new_tokens,
        )

        n = len(input_ids)
        columns = ["Epoch", "Sample Index", "Input", "Output", "Target"]
        table_columns = []
        table_columns.append([trainer.current_epoch] * n)
        table_columns.append(list(range(n)))

        for feature in [input_ids, out, target_ids]:
            decoded = self.tokenizer.batch_decode(
                feature, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            table_columns.append(decoded)

        run_name = self.wandb_logger.experiment.name if self.wandb_logger else "latest"

        rows = [list(row) for row in zip(*table_columns)]
        rows_df = pd.DataFrame(rows, columns=columns)
        rows_df.to_csv(
            f"run_{run_name}_samples.csv", mode="a", header=False, index=False
        )

        if batch_idx == 0:
            new_rows = tabulate(
                rows,
                headers=columns,
                tablefmt="simple_outline",
            )
            with open(f"run_{run_name}_samples.txt", "a") as f:
                f.write(new_rows)
                f.write("\n")


def main(wandb: bool = False):
    params = HyperParams()
    loggers = []

    model = T5FineTuner(params)

    wandb_logger = None

    if wandb:
        wandb_logger = WandbLogger(project="t5-upsampled-prompts")
        loggers.append(wandb_logger)
        wandb_logger.watch(model)

    dm = PromptUpsampleDataModule(
        params.model_name, batch_size=8, max_token_length=params.max_seq_length
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
