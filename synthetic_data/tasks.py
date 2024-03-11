from abc import ABC
from enum import Enum
from typing import Dict, List, Optional

from synthetic_data.utils import DatasetTaskFormat, SeedDataFormat


class SyntheticDataTask(ABC):

    seed_data_format: SeedDataFormat
    dataset_task_format: DatasetTaskFormat = DatasetTaskFormat.SFT
    
    # Only used for the toolformer DPO dataset. # TODO remove this when that project is finished.
    seed_data_uses_conversation_format: bool = False

    # Name for the dataset used to cache the seed data.
    # Once all the seed data is generated, this dataset will be used to cache the seed data.
    dpo_task_cache_dataset_name: Optional[str] = None

    seed_data_location: str
    output_dataset_name: str
    output_dataset_org: str

    empty_dataset_format: Dict[str, List]

    def prompt_template(self) -> str:
        """
        Prompt template to use for generating data.
        """
        raise NotImplementedError

    def validation_function(self, dataset_row: Dict) -> bool:
        """
        Return whether a dataset row is valid.
        """
        return True

    def scoring_function(self, dataset_row: Dict) -> float:
        """
        Score a single completion.
        """
        return 1


class PromptUpsample(SyntheticDataTask):

    seed_data_format = SeedDataFormat.TSV
    seed_data_location = "gs://openai-datasets/prompt-upsample/seed-data.tsv"
    output_dataset_name = "prompt-upsample"
    output_dataset_org = "openai"

    empty_dataset_format = {
        "Prompt": [],
        "Category": [],
        "Upsampled": [],
    }

    def prompt_template(self) -> str:
        return "Upsample the following prompt:"

    def validation_function(self, dataset_row: Dict) -> bool:
        return len(dataset_row["upsampled_prompt"]) > 0


class Toolformer(SyntheticDataTask):

    seed_data_format = SeedDataFormat.SYNTHETIC
    seed_data_location = "seed_data_files/domain_specific_tasks.csv"

    dataset_format = DatasetTaskFormat.DPO

    dpo_task_cache_dataset_name = "synthetic-toolformer-dpo-pairs"
    output_dataset_org = "roborovski"

    empty_dataset_format = {
        "system": [],
        "question": [],
        "chosen": [],
        "rejected": [],
    }
