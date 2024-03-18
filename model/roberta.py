from transformers.optimization import (
    get_linear_schedule_with_warmup,
)
from model.utils import HyperParams
from torch.optim import AdamW
from torch import Tensor
from transformers.models.roberta.modeling_roberta import (
    RobertaForSequenceClassification,
)
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from typing import Dict
from synthetic_data.utils import SAFE_PROMPT_LABELS

import lightning.pytorch as pl


class RobertaClassifier(pl.LightningModule):
    def __init__(self, params: HyperParams):
        super(RobertaClassifier, self).__init__()
        self.params = params
        self.hparams.update(vars(params))
        self.labels = SAFE_PROMPT_LABELS

        self.model: RobertaForSequenceClassification = (
            RobertaForSequenceClassification.from_pretrained(
                params.base_model_checkpoint,
                num_labels=len(self.labels),
                output_attentions=False,
                output_hidden_states=False,
            )
        )
        self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(
            params.base_model_checkpoint
        )  # type: ignore
        self.train_steps = 0
        self.save_hyperparameters()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            abels=labels,
        )

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
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"val_loss": loss}

    def configure_optimizers(self) -> Dict:
        # TODO double check right optimizer for roberta

        if self.params.optimizer == "AdamW":
            optimizer = AdamW(
                self.parameters(),
                lr=self.params.learning_rate,
                eps=self.params.adam_epsilon,
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=10
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                },
            }

        else:
            raise ValueError(f"Unsupported optimizer: {self.params.optimizer}")
