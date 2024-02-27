from dataclasses import dataclass
import os
import re
from typing import Dict, List, cast
import asyncio

from enum import Enum
from datasets import Dataset, load_dataset
import fire
from huggingface_hub import login
from dotenv import dotenv_values
from tabulate import tabulate
from synthetic_data.conversion import chatml_to_conversation
from synthetic_data.generation import (
    SHAREGPT_TO_OPENAI_ROLE,
    Conversation,
    GenerationWrapper,
    VLLMWrapper,
    OpenAIGenerationWrapper,
    upload_dataset,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from synthetic_data.prompts import format_dalle_prompt_template
from synthetic_data.utils import print_conversations_table


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

def clean_message(message: str) -> str:
    """
    Clean up spaces, tabs, and newlines in a message, so the JSON is formatted nicely.
    """
    message = message.strip()
    message = re.sub(r'\n+|\t+', '', message)
    message = re.sub(r'\s+', ' ', message)
    return message



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
    hf_token, oai_token = dotenv["HF_TOKEN"], dotenv["OPENAI_API_KEY"]
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

    print("Running...")
    for epoch in range(config.n_epochs):
        for i, batch in enumerate(input_dataset.iter(batch_size=batch_size)):
            if config.dataset_task == DatasetTask.PROMPT_UPSAMPLE:
                prompts, categories_batch = batch["Prompt"], batch["Category"]  # type: ignore
                full_conversations_batch: List[Conversation] = [
                    format_dalle_prompt_template(prompt) for prompt in prompts
                ]
                completions = asyncio.run(
                    model_wrapper.generate(full_conversations_batch)
                )

                for category, original_prompt, output in zip(
                    categories_batch, prompts, completions
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
            elif config.dataset_task == DatasetTask.TOOL_USAGE_DPO:
                full_conversations_batch: List[List[ChatCompletionMessageParam]] = []

                glaive_conversations = [chatml_to_conversation(chat, system) for chat, system in zip(batch["chat"], batch["system"])]  # type: ignore
                system_prompts = [
                    [clean_message(conv[0]["value"])] for conv in glaive_conversations
                ]
                for prompt in system_prompts:
                    print(f"\n{prompt}")
                completion_conversations: List[Conversation] = []
                for conversation in glaive_conversations:
                    completion_conv = []
                    for msg in conversation:
                        if msg["from"] == "gpt":
                            break
                        completion_conv.append(
                            {
                                "role": SHAREGPT_TO_OPENAI_ROLE[msg["from"]],
                                "content": msg["value"],
                            }
                        )
                    completion_conversations.append(completion_conv)

                print(f"Generating {len(completion_conversations)} completions for batch {i}...")
                completions = asyncio.run(
                    model_wrapper.generate(completion_conversations)
                )

                for completion, glaive_conversation in zip(
                    completions, glaive_conversations
                ):
                    system_msg = glaive_conversation[0]["value"]
                    user_msg = glaive_conversation[1]["value"]
                    accepted_msg = clean_message(glaive_conversation[-1]["value"])
                    rejected_msg = completion
                    new_dataset_rows.append(
                        {
                            "system": system_msg,
                            "question": user_msg,
                            "chosen": accepted_msg,
                            "rejected": rejected_msg,
                        }
                    )

                print_conversations_table(new_dataset_rows[:-8])

            if i % upload_every == 0 and i > 0:
                print(f"Upsampled {i} prompts")
                upload_dataset(
                    output_dataset, config.output_dataset_name, new_dataset_rows
                )

        upload_dataset(output_dataset, config.output_dataset_name, new_dataset_rows)


if __name__ == "__main__":
    fire.Fire(main)