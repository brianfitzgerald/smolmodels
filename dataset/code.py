from typing import Optional

from datasets.arrow_dataset import Dataset

from trl_wrapper.wrapper_config import SmDataset, DatasetConfig
from synthetic_data.utils import dictl
from transformers.tokenization_utils import PreTrainedTokenizer


DPO_COLS_TO_TOKENIZE = ["chosen", "rejected", "prompt"]


class CodeContestsDataModule(SmDataset):
    def load_dataset(self):
        # Load dataset and split
        dataset = Dataset.from_parquet("codecontests_dpo_v2_filtered.parquet")
        if self.config.max_samples:
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


class UltraFeedbackDataModule(SmDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, config: DatasetConfig):
        super().__init__(tokenizer, config)

        self.system_message = "You are a helpful AI assistant."
        self.config.dataset_path = "argilla/ultrafeedback-binarized-preferences-cleaned"

    def process_samples_batch(self, examples: dict):
        out_dict = {k: [] for k in DPO_COLS_TO_TOKENIZE}
        for i in range(len(examples["prompt"])):
            example = {k: v[i] for k, v in examples.items()}
            triplets = create_triplets(example, self.system_message)
            for response_role in DPO_COLS_TO_TOKENIZE:
                out_dict[response_role].append(triplets[response_role])
        return out_dict
