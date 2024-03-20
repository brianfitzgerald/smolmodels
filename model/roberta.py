from torch.optim.optimizer import Optimizer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from model.utils import HyperParams
from torch.optim import AdamW
from torch import Tensor
from transformers.models.roberta.modeling_roberta import (
    RobertaForSequenceClassification,
)
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from typing import Dict
from synthetic_data.utils import SAFERPROMPT_LABELS, ANNOTATED_LABELS
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

import lightning.pytorch as pl
from lightning.pytorch.utilities import grad_norm


class RobertaClassifier(pl.LightningModule):
    def __init__(self, params: HyperParams):
        super(RobertaClassifier, self).__init__()
        self.params = params
        self.hparams.update(vars(params))
        self.labels_to_id = (
            SAFERPROMPT_LABELS
            if params.labels_set == "clipdrop_synthetic"
            else ANNOTATED_LABELS
        )
        self.id_to_labels = {v: k for k, v in self.labels_to_id.items()}

        self.model: RobertaForSequenceClassification = (
            RobertaForSequenceClassification.from_pretrained(
                params.base_model_checkpoint,
                num_labels=len(self.labels_to_id),
                output_attentions=False,
                output_hidden_states=False,
            )
        )
        self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(
            params.base_model_checkpoint
        )  # type: ignore
        self.train_steps = 0
        self.save_hyperparameters()

        # TODO micro or macro?
        n_classes: int = len(self.labels_to_id)
        self.accuracy = Accuracy("multiclass", num_classes=n_classes, average="macro")
        self.f1 = F1Score("multiclass", num_classes=n_classes, average="macro")
        self.precision = Precision("multiclass", num_classes=n_classes, average="macro")
        self.recall = Recall("multiclass", num_classes=n_classes, average="macro")
    
    def forward(
        self,
        batch: Dict[str, Tensor],
    ):

        input_ids: Tensor = batch["input_ids"]
        attention_mask: Tensor = batch["attention_mask"]
        labels: Tensor = batch["labels"]

        out: SequenceClassifierOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        logits = out.logits
        metrics = {
            "accuracy": self.accuracy(logits, labels),
            "f1": self.f1(logits, labels),
            "precision": self.precision(logits, labels),
            "recall": self.recall(logits, labels),
        }

        return out.loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        for k, v in metrics.items():
            self.log(f"metrics/train_{k}", v, on_step=True, on_epoch=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, metrics = self(batch)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        for k, v in metrics.items():
            self.log(f"metrics/val_{k}", v, on_step=True, on_epoch=True, logger=True)
        return {"val_loss": loss}

    def configure_optimizers(self) -> Dict:
        # TODO double check right optimizer for roberta

        if self.params.optimizer == "AdamW":
            optimizer = AdamW(
                self.parameters(),
                lr=self.params.learning_rate,
                eps=self.params.adam_epsilon,
            )
            n_steps: int = self.trainer.estimated_stepping_batches # type: ignore
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.params.warmup_steps,
                num_training_steps=n_steps,
                num_cycles=1,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                },
            }

        else:
            raise ValueError(f"Unsupported optimizer: {self.params.optimizer}")

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)
