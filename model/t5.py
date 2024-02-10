from transformers.optimization import get_inverse_sqrt_schedule
from model.params import HyperParams
from torch.optim import AdamW
from torch import Tensor

import pytorch_lightning as pl

from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer


class T5FineTuner(pl.LightningModule):
    def __init__(self, params: HyperParams, ckpt_name: str):
        super(T5FineTuner, self).__init__()
        self.params = params
        self.hparams.update(vars(params))

        self.model: T5ForConditionalGeneration = (
            T5ForConditionalGeneration.from_pretrained(ckpt_name)
        )
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(ckpt_name)
        self.ckpt_name = ckpt_name
        self.train_steps = 0
        self.save_hyperparameters()

    def forward(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor):
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out

    def _step(self, batch):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
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
