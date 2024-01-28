print("Loading torch")
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from dataclasses import dataclass
from typing import TypedDict

print("Loading lightning")
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks

print("Loading HF")
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from datasets import load_dataset
from t5_data import PromptUpsampleDataModule


class HyperParams:
    model_name: str = "google/flan-t5-small"
    max_seq_length: int = 512
    learning_rate: float = 3e-4
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_train_epochs: int = 2
    gradient_accumulation_steps: int = 16
    n_gpus: int = 1
    early_stop_callback: bool = False
    fp_16: bool = False
    opt_level: str = "O1"
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

        self.model: T5ForConditionalGeneration = (
            T5ForConditionalGeneration.from_pretrained(self.params.model_name)
        )
        self.tokenizer = T5Tokenizer.from_pretrained(self.params.model_name)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

    def _step(self, batch):

        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["label_input_ids"],
            decoder_attention_mask=batch["label_attention_mask"],
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def on_train_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {
            "avg_train_loss": avg_train_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        # perplexity_score = get_perplexity(logits, input_ids)
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
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None
    ):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {
            "loss": "{:.3f}".format(self.avg_loss),
            "lr": self.lr_scheduler.get_last_lr()[-1],
        }

        return tqdm_dict

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.params.train_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=4,
        )
        t_total = (
            (
                len(train_dataset)  # type: ignore
                // (self.params.train_batch_size * max(1, self.params.n_gpus))
            )
            // self.params.gradient_accumulation_steps
            * float(self.params.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.params.warmup_steps,
            num_training_steps=t_total,
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.params.eval_batch_size, num_workers=4  # type: ignore
        )



if __name__ == "__main__":
    params = HyperParams()
    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=3
    )

    model = T5FineTuner(params)
    dm = PromptUpsampleDataModule(params.model_name, batch_size=8, max_token_length=512)
    trainer = pl.Trainer(
        accumulate_grad_batches=params.gradient_accumulation_steps,
        max_epochs=params.num_train_epochs,
        precision=16 if params.fp_16 else 32,
        gradient_clip_val=params.max_grad_norm,
    )
    trainer.fit(model, datamodule=dm)
