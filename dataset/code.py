from typing import Optional

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from loguru import logger
from transformers.tokenization_utils import PreTrainedTokenizer
import lightning.pytorch as pl
from trl import DataCollatorForCompletionOnlyLM
from rich.text import Text
import torch.nn.functional as F

from model.utils import SmDataset, TuningModeChoice
from synthetic_data.utils import dictl


DPO_COLS_TO_TOKENIZE = ["chosen", "rejected", "prompt"]


class CodeContestsDataModule(SmDataset):
    def load_dataset(self):
        # Load dataset and split
        dataset = Dataset.from_parquet("codecontests_dpo_v2_filtered.parquet")
        if self.max_samples:
            dataset = dataset.select(range(self.max_samples))  # type: ignore
        dataset = dataset.train_test_split(test_size=0.1)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

    def process_samples_batch(self, examples: dict):
        # No need to tokenize when using DPOTrainer

        prompts = [
            f"{example['name']}\n{example['description']}"
            for example in dictl(examples)
        ]

        batch_out = {
            "chosen": examples["chosen"],
            "rejected": examples["rejected"],
            "prompt": prompts,
        }
        return batch_out


def rec_extract_assistant_messages(messages, index=-1):
    """Recursively extract the last assistant messages from the end of the conversation."""
    if messages[index]["role"] == "assistant":
        return [messages[index]]
    else:
        return rec_extract_assistant_messages(messages, index - 1)


def create_triplets(
    example,
    default_system_message: Optional[str] = None,
):
    """Create the triplets (prompt, chosen, rejected)"""
    prompt_messages = example["chosen"][:-1]
    if example["chosen"][0]["role"] != "system" and default_system_message:
        prompt_messages.insert(0, {"role": "system", "content": default_system_message})
    chosen_messages = rec_extract_assistant_messages(example["chosen"])
    rejected_messages = rec_extract_assistant_messages(example["rejected"])

    return {
        "prompt": prompt_messages,
        "chosen": chosen_messages,
        "rejected": rejected_messages,
    }


class UltraFeedbackDataModule(pl.LightningDataModule):
    def __init__(
        self,
        max_samples: Optional[int] = None,
        default_system_message: Optional[str] = None,
    ):
        self.dataset_name = "argilla/ultrafeedback-binarized-preferences-cleaned"
        self.max_samples = max_samples
        self.default_system_message = default_system_message
        self.num_workers = 1
        # Not used
        self.cache_dir = "dataset_caches/ultrafeedback"
        self.use_cache = False

    def setup(self, stage: Optional[str] = None):
        # TODO offline generate reference logps
        # TODO filter by p95 length, and compute max length for tokenization
        # Load dataset and split
        dataset = load_dataset(self.dataset_name)["train"]  # type: ignore
        logger.info(f"Loaded dataset with {len(dataset)} samples")
        if self.max_samples:
            dataset = dataset.select(range(self.max_samples))  # type: ignore
        dataset = dataset.train_test_split(test_size=0.1)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

    def process_samples_batch(self, examples: dict):
        out_dict = {k: [] for k in DPO_COLS_TO_TOKENIZE}
        for i in range(len(examples["prompt"])):
            example = {k: v[i] for k, v in examples.items()}
            triplets = create_triplets(example, self.default_system_message)
            for response_role in DPO_COLS_TO_TOKENIZE:
                out_dict[response_role].append(triplets[response_role])
        return out_dict


SFT_COLS = ["conversations"]


class EvolCodeAlpacaDataModule(SmDataset):
    def load_dataset(self):
        # Load dataset and split
        logger.info("Loading dataset")
        dataset = load_dataset("AlekseyKorshuk/evol-codealpaca-v1-dpo")[
            "train"
        ].train_test_split(test_size=0.01)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

    def process_samples_batch(self, examples):
        batch_out = {
            "chosen": examples["chosen"],
            "rejected": examples["rejected"],
            "prompt": examples["question"],
        }
        return batch_out

    def process_samples_batch_sft(self, examples):
        """
        Convert to conversation format
        """
        out = {k: [] for k in SFT_COLS}
        for i in range(len(examples["question"])):
            system, question = examples["system"][i], examples["question"][i]
            conv = [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ]
            out["conversations"].append(conv)
        return out


class ConversationDataModule(SmDataset):
    """
    Generic data module for conversation datasets.
    """

    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
        tuning_mode: TuningModeChoice,
        max_samples: Optional[int] = None,
        use_cache: bool = True,
    ):
        super().__init__(
            batch_size, tokenizer, max_token_length, tuning_mode, max_samples, use_cache
        )
        self.collator = DataCollatorForCompletionOnlyLM(
            "assistant", tokenizer=tokenizer
        )

    def load_dataset(self):
        # Load dataset and split
        logger.info("Loading dataset")
        dataset = Dataset.from_parquet(
            "../codecontests_cot_sft_formatted_thoughts_conversations.parquet"
        )
        dataset = dataset.train_test_split(test_size=0.1)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]
        self.messages_field = "conversation"

    def post_setup(self):
        self.train_dataset = self.train_dataset.remove_columns("conversation")
        self.val_dataset = self.val_dataset.remove_columns("conversation")

    def process_samples_batch_sft(self, examples: dict):
        if isinstance(examples[self.messages_field][0], list):
            output_texts = []
            for i in range(len(examples[self.messages_field])):
                output_texts.append(
                    self.tokenizer.apply_chat_template(
                        examples[self.messages_field][i], tokenize=False
                    )
                )
            sample_batch = output_texts
        else:
            sample_batch = self.tokenizer.apply_chat_template(
                examples[self.messages_field], tokenize=False
            )

        tokenized_out = []
        for s in sample_batch:
            s = self.tokenizer(
                s,
                padding="max_length",
                truncation=True,
                return_special_tokens_mask=True,
            )
            tokenized_out.append(s)

        dict_out = self.collator(tokenized_out)
        return dict_out

    def visualize_sample(self, input_dict) -> Text:
        input_ids = input_dict["input_ids"]
        attention_mask = input_dict["attention_mask"]
        attention_mask = F.pad(
            attention_mask,
            (0, input_dict["attention_mask"].shape[0] - attention_mask.size(0)),
        )

        rich_text = Text()

        for token, mask in zip(input_ids, attention_mask):
            decoded = self.tokenizer.decode(token)
            if mask == 1:
                rich_text.append(decoded, style="bright_green")
            else:
                rich_text.append(decoded, style="bright_red")
        return rich_text
