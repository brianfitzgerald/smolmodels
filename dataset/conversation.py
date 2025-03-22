import torch

from model.utils import (
    IGNORE_TOKEN_INDEX,
    SmDataset,
)
from synthetic_data.utils import Conversation
from datasets import load_dataset, Dataset

COLS_TO_REMOVE = [
    "conversation",
]


class ConversationDataModule(SmDataset):
    """
    Generic data module for conversation datasets.
    """

    def post_setup(self):
        self.train_dataset: Dataset = self.train_dataset.remove_columns(COLS_TO_REMOVE)
        self.val_dataset: Dataset = self.val_dataset.remove_columns(COLS_TO_REMOVE)

    def process_samples_batch(self, examples: dict):
        out = self._tokenize_conversation(examples["conversation"])
        return out

    def _tokenize_conversation(self, conversation: Conversation) -> dict:
        tokenized_out: dict = self.tokenizer.apply_chat_template(
            conversation,  # type: ignore
            chat_template=self.custom_template,
            truncation=True,
            padding=True,
            max_length=self.config.max_sequence_length,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        if self.config.train_on_inputs:
            return tokenized_out
        else:
            labels, assistant_mask = (
                tokenized_out["input_ids"],
                tokenized_out["assistant_masks"],
            )
            labels, assistant_mask = torch.tensor(labels), torch.tensor(assistant_mask)
            labels = torch.where(assistant_mask == 0, IGNORE_TOKEN_INDEX, labels)
            return {
                "input_ids": tokenized_out["input_ids"],
                "attention_mask": tokenized_out["attention_mask"],
                "assistant_mask": tokenized_out["assistant_masks"],
                "labels": labels,
            }


def extract_answer_from_dataset(text):
    """
    Extracts the answer from the dataset.
    The dataset separates the answer using the '####' delimiter.
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


class ConversationDPODataModule(ConversationDataModule):
    """
    Conversation data module, assumes a dataset with `chosen`, `rejected`, and `prompt` columns.
    The columns are formatted into a conversation and tokenized.
    """

    def process_samples_batch(self, examples):
        batch_out = {
            "chosen": examples["chosen"],
            "rejected": examples["rejected"],
            "prompt": examples["prompt"],
        }
        return batch_out
