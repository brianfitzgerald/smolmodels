from transformers.optimization import get_cosine_schedule_with_warmup
from model.utils import HyperParams
from torch.optim import AdamW

import pytorch_lightning as pl

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer


class LlamaFineTuner(pl.LightningModule):
    def __init__(self, params: HyperParams, ckpt_name: str):
        super(LlamaFineTuner, self).__init__()
        self.params = params
        self.hparams.update(vars(params))

        self.model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(ckpt_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(ckpt_name)
        self.ckpt_name = ckpt_name
        self.train_steps = 0
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels):
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
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.params.learning_rate,
            eps=self.params.adam_epsilon,
            weight_decay=self.params.weight_decay,
        )
        print(f"Configuring optimizers: {self.train_steps}")
        total_training_steps = self.params.num_train_epochs * self.train_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.params.warmup_steps, total_training_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }
