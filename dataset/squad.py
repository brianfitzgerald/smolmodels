import json
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from unidecode import unidecode

from model.utils import IGNORE_TOKEN_INDEX, SmDataset
from synthetic_data.prompts import ENTITY_EXTRACTION_TUNING_INSTRUCTION
from synthetic_data.utils import ShareGPTConversation


def format_squad_extractive(sample: dict) -> Tuple[str, str]:
    sample["json_schema"] = unidecode(sample["json_schema"])
    sample["context"] = unidecode(sample["context"])

    schema: dict = json.loads(sample["json_schema"])
    json_schema = {key: type(value).__name__ for key, value in schema.items()}
    input_out = f"Extract the following information using the provided schema: \t{json.dumps(json_schema)}\tand the following context: \t{sample['context']}\n"
    labels_out = json.dumps(schema)

    return input_out, labels_out


class SquadExtractiveQADataModule(SmDataset):
    def __init__(
        self,
        *args,
    ):
        super().__init__(*args)

        self.dataset_name = "roborovski/squad-extractive-qa"

    def process_samples_batch(self, examples):
        inputs, labels = [], []
        for i in range(len(examples["id"])):  # type: ignore
            sample = {k: v[i] for k, v in examples.items()}
            sample_input, sample_labels = format_squad_extractive(sample)
            inputs.append(sample_input)
            labels.append(sample_labels)

        return self._tokenize(inputs, labels)


class SquadDataModule(SmDataset):
    def __init__(
        self,
        *args,
    ):
        super().__init__(*args)
        self.dataset_name = "rajpurkar/squad_v2"

    def process_samples_batch(self, examples):
        inputs, labels = [], []
        for i in range(len(examples["id"])):  # type: ignore
            sample = {k: v[i] for k, v in examples.items()}
            answers = sample["answers"]["text"]
            sample_input = (
                f"Question: {sample['question']}\nContext: {sample['context']}\n"
            )
            sample_labels = answers[0] if answers else "Cannot answer this question"
            inputs.append(sample_input)
            labels.append(sample_labels)

        return self._tokenize(inputs, labels)


class DollyEntityExtractionDataModule(SmDataset):
    def __init__(
        self,
        *args,
    ):
        super().__init__(*args)

        self.dataset_name = "roborovski/dolly-entity-extraction"
        self.cpu_count = 1

    def process_samples_batch(self, examples):
        input_ids_out, labels_out, attention_masks_out = [], [], []
        for i in range(len(examples["context"])):  # type: ignore
            sample = {k: v[i] for k, v in examples.items()}

            conversation: ShareGPTConversation = [
                {
                    "role": "system",
                    "content": ENTITY_EXTRACTION_TUNING_INSTRUCTION,
                },
                {
                    "role": "user",
                    "content": sample["context"],
                },
                {
                    "role": "user",
                    "content": sample["json_query"],
                },
            ]

            conversation_completion = conversation + [
                {
                    "role": "assistant",
                    "content": sample["json_data"],
                }
            ]

            prompt_ids: Tensor = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )[0]  # type: ignore
            input_ids: Tensor = self.tokenizer.apply_chat_template(
                conversation_completion, tokenize=True, return_tensors="pt"
            )[0]  # type: ignore

            user_prompt_len = prompt_ids.shape[0]
            labels = torch.tensor(
                [IGNORE_TOKEN_INDEX] * user_prompt_len
                + input_ids[user_prompt_len:].tolist()
            )
            pad_amt = self.max_token_length - labels.shape[0]
            labels = F.pad(labels, (0, pad_amt), value=self.tokenizer.pad_token_id)
            input_ids = F.pad(
                input_ids, (0, pad_amt), value=self.tokenizer.pad_token_id
            )

            labels_out.append(labels)
            input_ids_out.append(input_ids)
            attention_masks_out.append(torch.ones_like(input_ids))

        input_ids_out = torch.stack(input_ids_out)
        labels_out = torch.stack(labels_out)
        attention_masks_out = torch.stack(attention_masks_out)
        return {
            "input_ids": input_ids_out,
            "attention_mask": attention_masks_out,
            "labels": labels_out,
        }
