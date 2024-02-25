from dataclasses import dataclass
import os
from typing import Dict, List

from enum import Enum
import pandas as pd
from datasets import Dataset, load_dataset
import fire
from huggingface_hub import login
from dotenv import dotenv_values
from synthetic_data.generation import (
    GenerationWrapper,
    VLLMWrapper,
    OpenAIGenerationWrapper,
    upload_dataset,
)

from synthetic_data.prompts import format_dalle_prompt_template


class GenerationSource(str, Enum):
    OPENAI = "openai"
    VLLM = "vllm"


class DataSourceFormat(Enum):
    TSV = "tsv"
    HF_DATASET = "hf_dataset"


class DatasetFormat(Enum):
    DPO = "dpo"
    UPSAMPLE = "upsample"


@dataclass
class Config:
    dataset_format: DatasetFormat
    seed_data_format: DataSourceFormat
    # Either the name of the HF dataset or the path to the CSV file
    seed_data_location: str
    output_dataset_name: str
    n_epochs: int = 10


CONFIGS = {
    "tool_usage": Config(
        dataset_format=DatasetFormat.DPO,
        seed_data_format=DataSourceFormat.HF_DATASET,
        seed_data_location="glaiveai/glaive-function-calling-v2",
        output_dataset_name="roborovski/glaive-tool-usage-dpo",
    ),
    "upsample": Config(
        dataset_format=DatasetFormat.UPSAMPLE,
        seed_data_format=DataSourceFormat.TSV,
        seed_data_location="data/PartiPrompts.tsv",
        output_dataset_name="roborovski/upsampled-prompts-parti",
    ),
}


def main(
    upload_every: int = 500,
    batch_size: int = 16,
    restart: bool = False,
    generation_source: GenerationSource = GenerationSource.OPENAI,
    config_name: str = "tool_usage",
):

    config = CONFIGS[config_name]

    print("Logging into the Hub...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv = dotenv_values(os.path.join(current_dir, ".env"))
    hf_token, oai_token = dotenv["HF_TOKEN"], dotenv["OAI_TOKEN"]
    print(f"Logging in with token: {hf_token}")
    login(token=hf_token, add_to_git_credential=True)

    model_wrapper: GenerationWrapper = (
        OpenAIGenerationWrapper(oai_token)
        if generation_source == GenerationSource.OPENAI == "openai"
        else VLLMWrapper()
    )

    print("Loading existing data...")
    if restart:
        hf_dataset = Dataset.from_dict({"Prompt": [], "Category": [], "Upsampled": []})
    else:
        hf_dataset: Dataset = load_dataset(hf_dataset_name, split="train")  # type: ignore

    print("Loading new prompts...")
    parti_prompts: pd.DataFrame = pd.read_csv("data/PartiPrompts.tsv", sep="\t")

    new_dataset_rows: List[Dict] = []

    # initial test upload before loading the pipeline
    upload_dataset(hf_dataset, config.output_dataset_name, new_dataset_rows)

    print("Upsampling captions...")
    for epoch in range(config.n_epochs):
        for i in range(0, len(parti_prompts), batch_size):
            batch = parti_prompts.iloc[i : i + batch_size]
            prompts, categories_batch = batch["Prompt"], batch["Category"]
            full_conversations_batch = []

            for prompt in prompts.to_list():
                conversation = format_dalle_prompt_template(prompt)

                full_conversation_formatted: str = model_wrapper.tokenizer.apply_chat_template(  # type: ignore
                    conversation, tokenize=False, add_generation_prompt=True
                )
                full_conversations_batch.append(full_conversation_formatted)

            outputs = model_wrapper.generate(full_conversations_batch)

            for category, original_prompt, output in zip(
                categories_batch, prompts, outputs  # type: ignore
            ):
                upsampled = output.outputs[0].text
                new_dataset_rows.append(
                    {
                        "Prompt": original_prompt,
                        "Category": category,
                        "Upsampled": upsampled,
                    }
                )

                print(
                    f"Epoch: {epoch} idx: {i} ({category}): {original_prompt} -> {upsampled}"
                )

            if i % upload_every == 0:
                print(f"Upsampled {i} prompts")
                upload_dataset(hf_dataset, config.output_dataset_name, new_dataset_rows)

        upload_dataset(hf_dataset, config.output_dataset_name, new_dataset_rows)


if __name__ == "__main__":
    fire.Fire(main)
