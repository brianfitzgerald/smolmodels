from transformers.tokenization_utils import PreTrainedTokenizer
import pandas as pd
from datasets import Dataset
from typing import List


from dataset.utils import FineTunerDataset


class PromptClassifierDataModule(FineTunerDataset):
    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)
        self.prompt_column = "prompt"
        self.label_column = "class_label"
        self.cache_dir = "dataset_caches/saferprompt_classifier"

    def setup(self, stage: str):
        print(f"Loading dataset for stage {stage}")
        dataset_csv = pd.read_csv("data_files/classified_prompts_40k.csv")
        self.dataset = Dataset.from_pandas(dataset_csv)

        self.train_dataset = self.train_dataset.map(
            self.prepare_sample,
            batched=True,
            load_from_cache_file=True,
            cache_file_name=f"{self.cache_dir}/training.parquet",
            num_proc=self.cpu_count,
        )

        self.val_dataset = self.val_dataset.map(
            self.prepare_sample,
            batched=True,
            load_from_cache_file=True,
            cache_file_name=f"{self.cache_dir}/validation.parquet",
            num_proc=self.cpu_count,
        )


    def prepare_sample(self, examples: dict):

        inputs: List[str] = examples[self.label_column]

        inputs_tokenized = self.tokenizer(
            inputs,
            add_special_tokens=True,
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return inputs_tokenized


