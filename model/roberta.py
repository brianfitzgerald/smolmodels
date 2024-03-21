from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.optimization import (
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
from synthetic_data.labels import LABEL_SETS_DICT
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
import torch

import lightning.pytorch as pl


class RobertaClassifier(pl.LightningModule):
    def __init__(
        self, params: HyperParams, problem_type: str = "single_label_classification"
    ):
        super(RobertaClassifier, self).__init__()
        self.params = params
        if not params.labels_set:
            raise ValueError(f"Labels set {params.labels_set} not found")
        self.labels_to_id = LABEL_SETS_DICT[params.labels_set]
        self.id_to_labels = {v: k for k, v in self.labels_to_id.items()}

        self.model: RobertaForSequenceClassification = (
            RobertaForSequenceClassification.from_pretrained(
                params.base_model_checkpoint,
                output_attentions=False,
                problem_type=problem_type,
                num_labels=len(self.labels_to_id),
                output_hidden_states=False,
            )
        )
        self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(
            params.base_model_checkpoint
        )  # type: ignore
        self.train_steps = 0
        self.save_hyperparameters()

    def setup(self, stage: str):
        n_labels: int = len(self.labels_to_id)
        self.accuracy = Accuracy("multiclass", num_classes=n_labels, average="macro")
        self.f1 = F1Score("multiclass", num_classes=n_labels, average="macro")
        self.precision = Precision("multiclass", num_classes=n_labels, average="macro")
        self.recall = Recall("multiclass", num_classes=n_labels, average="macro")

    def forward(
        self,
        batch: Dict[str, Tensor],
    ):

        input_ids: Tensor = batch["input_ids"]
        attention_mask: Tensor = batch["attention_mask"]
        labels: Tensor = batch["labels"]

        if self.params.objective == "multilabel_classification":
            labels = labels.float()

        out: SequenceClassifierOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        metric_value = out.logits

        if self.params.objective == "multilabel_classification":
            # convert to logits
            metric_value = (torch.sigmoid(out.logits).squeeze(dim=0) > 0.5).float()

        metrics = {
            "accuracy": self.accuracy(metric_value, labels),
            "f1": self.f1(metric_value, labels),
            "precision": self.precision(metric_value, labels),
            "recall": self.recall(metric_value, labels),
        }

        return out.loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        for k, v in metrics.items():
            self.log(f"train_metrics/{k}", v, on_step=True, on_epoch=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, metrics = self(batch)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        for k, v in metrics.items():
            self.log(f"val_metrics/{k}", v, on_step=True, on_epoch=True, logger=True)
        return {"val_loss": loss}

    def configure_optimizers(self) -> Dict:
        # TODO double check right optimizer for roberta

        if self.params.optimizer == "AdamW":
            optimizer = AdamW(
                self.parameters(),
                lr=self.params.learning_rate,
                eps=self.params.adam_epsilon,
            )
            n_steps: int = self.trainer.estimated_stepping_batches  # type: ignore
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
                    "interval": "step",
                },
            }

        else:
            raise ValueError(f"Unsupported optimizer: {self.params.optimizer}")


class RobertaClassifierMultilabel(RobertaClassifier):
    def __init__(self, params: HyperParams):
        super(RobertaClassifierMultilabel, self).__init__(
            params, problem_type="multi_label_classification"
        )

    def setup(self, stage: str):
        n_labels: int = len(self.labels_to_id)
        self.accuracy = Accuracy("multilabel", num_labels=n_labels, average="macro")
        self.f1 = F1Score("multilabel", num_labels=n_labels, average="macro")
        self.precision = Precision("multilabel", num_labels=n_labels, average="macro")
        self.recall = Recall("multilabel", num_labels=n_labels, average="macro")
