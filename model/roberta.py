from transformers.optimization import (
    Adafactor,
    get_inverse_sqrt_schedule,
    AdafactorSchedule,
)
from model.utils import HyperParams
from torch.optim import AdamW
from torch import Tensor
from transformers import AutoTokenizer
from transformers.models.roberta.modeling_roberta import (
    RobertaForSequenceClassification,
)
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer

import lightning.pytorch as pl


class RobertaClassifier(pl.LightningModule):
    def __init__(self, params: HyperParams):
        super(RobertaClassifier, self).__init__()
        self.params = params
        self.hparams.update(vars(params))

        self.model: RobertaForSequenceClassification = (
            RobertaForSequenceClassification.from_pretrained(
                params.base_model_checkpoint
            )
        )  # type: ignore
        self.tokenizer: RobertaTokenizer = AutoTokenizer.from_pretrained(
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

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        if self.params.optimizer == "AdamW":
            optimizer = AdamW(
                self.parameters(),
                lr=self.params.learning_rate,
                eps=self.params.adam_epsilon,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.params.optimizer}")

        return {
            "optimizer": optimizer,
        }
