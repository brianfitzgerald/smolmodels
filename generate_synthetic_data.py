from dataclasses import dataclass
import os
from typing import Dict, List, cast
import asyncio

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
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from synthetic_data.prompts import format_dalle_prompt_template


class GenerationSource(str, Enum):
    OPENAI = "openai"
    VLLM = "vllm"


class DataSourceFormat(Enum):
    TSV = "tsv"
    HF_DATASET = "hf_dataset"


class DatasetTask(Enum):
    TOOL_USAGE_DPO = "dpo"
    PROMPT_UPSAMPLE = "upsample"


@dataclass
class Config:
    dataset_task: DatasetTask
    seed_data_format: DataSourceFormat
    # Either the name of the HF dataset or the path to the CSV file
    seed_data_location: str
    output_dataset_name: str
    n_epochs: int = 10


CONFIGS = {
    "tool_usage": Config(
        dataset_task=DatasetTask.TOOL_USAGE_DPO,
        seed_data_format=DataSourceFormat.HF_DATASET,
        seed_data_location="glaiveai/glaive-function-calling-v2",
        output_dataset_name="roborovski/glaive-tool-usage-dpo",
    ),
    "upsample": Config(
        dataset_task=DatasetTask.PROMPT_UPSAMPLE,
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
        output_dataset = Dataset.from_dict(
            {"Prompt": [], "Category": [], "Upsampled": []}
        )
    else:
        output_dataset = cast(
            Dataset, load_dataset(config.output_dataset_name, split="train")
        )

    input_dataset: Dataset
    if config.seed_data_format == DataSourceFormat.HF_DATASET:
        input_dataset = cast(
            Dataset, load_dataset(config.seed_data_location, split="train")
        )
    elif config.seed_data_format == DataSourceFormat.TSV:
        input_dataset = cast(
            Dataset, load_dataset("tsv", data_files=config.seed_data_location)
        )

    new_dataset_rows: List[Dict] = []

    # initial test upload before loading the pipeline
    upload_dataset(output_dataset, config.output_dataset_name, new_dataset_rows)

    print("Running...")
    for epoch in range(config.n_epochs):
        for i, batch in enumerate(input_dataset.iter(batch_size=batch_size)):
            prompts, categories_batch = batch["Prompt"], batch["Category"] # type: ignore
            full_conversations_batch: List[List[ChatCompletionMessageParam]] = []

            for prompt in prompts.to_list():
                if config.dataset_task == DatasetTask.TOOL_USAGE_DPO:
                    conversation = format_dalle_prompt_template(prompt)
                elif config.dataset_task == DatasetTask.PROMPT_UPSAMPLE:
                    conversation = format_dalle_prompt_template(prompt)

                full_conversation_formatted = (
                    model_wrapper.tokenizer.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=True
                    )
                )
                full_conversations_batch.append(
                    cast(List[ChatCompletionMessageParam], full_conversation_formatted)
                )

            outputs = asyncio.run(model_wrapper.generate(full_conversations_batch))

            for category, original_prompt, output in zip(
                categories_batch, prompts, outputs
            ):
                new_dataset_rows.append(
                    {
                        "Prompt": original_prompt,
                        "Category": category,
                        "Upsampled": output,
                    }
                )

                print(
                    f"Epoch: {epoch} idx: {i} ({category}): {original_prompt} -> {output}"
                )

            if i % upload_every == 0:
                print(f"Upsampled {i} prompts")
                upload_dataset(
                    output_dataset, config.output_dataset_name, new_dataset_rows
                )

        upload_dataset(output_dataset, config.output_dataset_name, new_dataset_rows)


if __name__ == "__main__":
    fire.Fire(main)
