import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import lightning.pytorch as pl
import torch
from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger
from rich.text import Text
from torch.utils.data import DataLoader
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_utils import SchedulerType
from transformers.training_args import OptimizerNames
from trl.trainer.grpo_trainer import RewardFunc

from model.utils import (
    IGNORE_TOKEN_INDEX,
    DataModuleChoice,
    TuningModeChoice,
    ensure_directory,
)
from synthetic_data.tasks import RunMode
from synthetic_data.utils import EvalDataModeChoice
from synthetic_data.generation import RemoteModel

MOCK_LLAMA = "qgallouedec/tiny-LlamaForCausalLM-3"
LLAMA_3_2_1B = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_3_2_3B_BASE = "meta-llama/Llama-3.2-3B"
LLAMA_3_1_8B = "meta-llama/Llama-3.1-8B-Instruct"
SMOL_LM_135M = "HuggingFaceTB/SmolLM2-135M-Instruct"
# NOTE that mistral doesn't allow using system prompts, so it must be set to None.
MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
MINISTRAL_8B = "mistralai/Ministral-8B-Instruct-2410"

QWEN_2_0_5_B = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_2_1_5_B = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_2_5_3B = "Qwen/Qwen2.5-3B-Instruct"

DataCollatorChoice = Literal["basic", "chat"]
ModelFamily = Literal["qwen", "mistral", "other"]


@dataclass
class WrapperConfig:
    # Model & Adapter Configuration
    model_id_or_path: str = LLAMA_3_2_1B
    adapter_path: Optional[str] = None
    grpo_beta: float = 0.0

    # Experiment / Environment Settings
    wandb_project_name: str = "codecontests-llama-3b"
    run_suffix: Optional[str] = None
    special_tokens: Optional[List[str]] = None

    # Data & Evaluation Configuration
    data_module_choice: DataModuleChoice = "conversation"
    dataset_path: Optional[str] = None
    eval_data_mode: EvalDataModeChoice = "random"
    # Max samples to use for training
    max_samples: Optional[int] = None
    max_eval_dataset_size: Optional[int] = None
    eval_steps: int = 500
    save_steps: int = 1000

    # Prompt & Sequence Lengths
    max_sequence_length: int = 1512  # sequence length for trimming completions
    max_prompt_length: int = 1024
    max_eval_sample_length: int = 512
    max_completion_length: int = 200

    warmup_steps: int = 1000

    # Training Parameters
    train_batch_size: int = 4
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    n_epochs: int = 1
    train_on_inputs: bool = False

    # Optimization & Scheduling
    learning_rate: float = 5e-5
    max_grad_norm: float = 0.3
    lr_scheduler: SchedulerType = SchedulerType.CONSTANT
    optimizer: str = OptimizerNames.ADAMW_8BIT.value
    neftune_noise_alpha: Optional[float] = None
    dpo_beta: float = 0.1

    # Tuning / LoRA Configuration
    tuning_mode: TuningModeChoice = "sft"
    use_lora: bool = False
    lora_rank: int = 256
    lora_dropout: float = 0.05
    lora_alpha: int = 128
    logprob_precompute_batch_size: int = 16

    # GRPO
    num_generations: int = 1
    judge_model: RemoteModel = "gemini-2.0-flash"

    @property
    def model_family(self) -> ModelFamily:
        if self.model_id_or_path.startswith("mistralai"):
            return "mistral"
        elif self.model_id_or_path.startswith("Qwen"):
            return "qwen"
        return "other"

    @property
    def using_mistral(self) -> bool:
        return self.model_family == "mistral"


@dataclass
class DatasetConfig(WrapperConfig):
    chat_template_path: str | None = None
    run_mode: RunMode = "cli"


def class_name_to_underscore(cls):
    class_name = cls.__name__  # Get the class name
    underscore_case = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
    return underscore_case


class SmDataset(pl.LightningDataModule):
    def __init__(self, tokenizer: PreTrainedTokenizer, config: DatasetConfig):
        super().__init__()

        self.config = config

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.tokenizer = tokenizer
        self.num_workers = 1 if config.run_mode == "notebook" else None
        current_dir = Path().resolve().name
        self.prefix = "/"
        if current_dir == "notebooks":
            self.prefix = "../"
        self.cache_dir = (
            f"{self.prefix}dataset_caches/{class_name_to_underscore(self.__class__)}"
        )
        self.input_column, self.target_column = "context", "fields"
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.dataset_name: str | None = None

    def init_dataset(self):
        """
        Cannot call load_dataset as that will shadow the load_dataset function from datasets
        """
        # Load dataset and split
        assert self.config.dataset_path is not None, "Input dataset name must be set"
        local_dataset_location = f"{self.prefix}{self.config.dataset_path}"
        if os.path.exists(local_dataset_location):
            logger.info(f"Loading local dataset from {local_dataset_location}")
            dataset = Dataset.from_parquet(local_dataset_location)
        else:
            logger.info(
                f"Dataset not found at expected local location {local_dataset_location}, loading from remote: {self.config.dataset_path}"
            )
            dataset = load_dataset(self.config.dataset_path)
            if isinstance(dataset, DatasetDict):
                dataset = dataset["train"]
        if self.config.max_samples:
            logger.info(f"Selecting {self.config.max_samples} samples")
            dataset = dataset.select(range(self.config.max_samples))  # type: ignore
        dataset = dataset.train_test_split(test_size=0.1)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]
        self.custom_template = None
        if self.config.dataset_path is not None:
            chat_template_path = (
                f"{self.prefix}chat_templates/{self.config.chat_template_path}.jinja"
            )
            logger.info(f"Loading custom chat template: {chat_template_path}")
            with open(chat_template_path) as f:
                self.custom_template = f.read()

    def setup(self, stage: Optional[str] = None):
        logger.info(f"Loading dataset for stage {stage}")
        ensure_directory(self.cache_dir, clear=False)
        use_cache = not self.config.run_mode == "notebook"
        logger.info(
            f"Processing dataset for stage {stage}, workers: {self.num_workers}, cache dir {self.cache_dir}, using cache: {use_cache}"
        )

        self.init_dataset()

        assert self.train_dataset is not None
        assert self.val_dataset is not None
        train_steps_per_epoch = len(self.train_dataset) // self.config.train_batch_size
        logger.info(
            f"Train dataset samples: {len(self.train_dataset)} Val dataset samples: {len(self.val_dataset)} Train steps per epoch: {train_steps_per_epoch}"
        )

        self.train_dataset = self.train_dataset.map(
            self.process_samples_batch,
            batched=True,
            load_from_cache_file=use_cache,
            cache_file_name=f"{self.cache_dir}/training.parquet",
            num_proc=self.num_workers,
        )

        self.val_dataset = self.val_dataset.map(
            self.process_samples_batch,
            batched=True,
            load_from_cache_file=use_cache,
            cache_file_name=f"{self.cache_dir}/validation.parquet",
            num_proc=self.num_workers,
        )
        self.post_setup()

    def post_setup(self):
        pass

    def process_samples_batch(self, examples: dict):
        return self._tokenize(examples[self.input_column], examples[self.target_column])

    def _tokenize(self, inputs: List[str], labels: List[str]) -> dict:
        """
        Basic tokenizing function. Inputs are samples from the dataset, and labels are the target values.
        """
        inputs_tokenized = self.tokenizer(
            inputs,
            max_length=self.config.max_sequence_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels_tokenized = self.tokenizer(
            labels,
            max_length=self.config.max_sequence_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": inputs_tokenized["input_ids"],
            "attention_mask": inputs_tokenized["attention_mask"],
            "labels": labels_tokenized["input_ids"],
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.config.train_batch_size,
            num_workers=self.num_workers if self.num_workers else 0,
        )  # type: ignore

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=8, num_workers=self.num_workers)  # type: ignore

    def visualize_sample(self, input_dict: dict[str, torch.Tensor | list]) -> Text:
        """Visualize a sample from the dataset."""
        input_ids, labels = input_dict["input_ids"], input_dict["labels"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        if isinstance(labels, list):
            labels = torch.tensor(labels)
        input_ids = input_ids.squeeze().tolist()
        labels = labels.squeeze().tolist()

        rich_text = Text()

        for token, label in zip(input_ids, labels):
            decoded = self.tokenizer.decode(token)
            if label == 0 or label == IGNORE_TOKEN_INDEX:
                rich_text.append(decoded, style="bright_red")
            else:
                rich_text.append(decoded, style="bright_green")
        return rich_text

    def reward_functions(self) -> list[RewardFunc]:
        return []
