import asyncio
import os
import traceback
from typing import Dict, List, cast

import fire
from datasets import Dataset, load_dataset
from datasets.data_files import EmptyDatasetError
from dotenv import dotenv_values
from huggingface_hub import login
from synthetic_data.tasks import (
    PromptUpsample,
    SyntheticDataTask,
    Toolformer,
    Toolformer,
)

from synthetic_data.generation import (
    GenerationWrapper,
    GroqGenerationWrapper,
    OpenAIGenerationWrapper,
    OpenRouterGenerationWrapper,
    VLLMWrapper,
    upload_dataset,
)
from synthetic_data.utils import (
    DatasetTaskFormat,
    GenerationSource,
    SeedDataFormat,
    assert_valid_python_code,
    extract_toolformer_dpo_row,
    print_result_dicts,
)

DATA_TASKS: Dict[str, type[SyntheticDataTask]] = {
    "toolformer": Toolformer,
    "prompt_upsample": PromptUpsample,
}

MODEL_WRAPPER_CLASSES = {
    GenerationSource.OPENAI: OpenAIGenerationWrapper,
    GenerationSource.VLLM: VLLMWrapper,
    GenerationSource.OPENROUTER: OpenRouterGenerationWrapper,
    GenerationSource.GROQ: GroqGenerationWrapper,
}


def main(
    # n batches
    upload_every: int = 10,
    batch_size: int = 16,
    restart: bool = False,
    generate_dpo_pairs: bool = False,
    generation_source: GenerationSource = GenerationSource.OPENROUTER,
    task_name: str = "toolformer_pairs",
    **kwargs,
):
    """
    Generate synthetic preference data from a given dataset.
    Inputs a seed dataset, that is either given from a CSV or HF dataset,
    or generated from a synthetic source, such as a list of subjects.

    generate_dpo_negative_pair - If true, generate a negative pair for the DPO dataset.
    """
    assert not kwargs, f"Unrecognized arguments: {kwargs}"

    task = DATA_TASKS[task_name]()

    if generate_dpo_pairs and task.dataset_task_format != DatasetTaskFormat.DPO:
        raise ValueError("generate_pairs is only supported for DPO tasks.")

    print("Logging into the Hub...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv = dotenv_values(os.path.join(current_dir, ".env"))
    hf_token = dotenv["HF_TOKEN"]
    print(f"Logging in with token: {hf_token}")
    login(token=hf_token, add_to_git_credential=True)

    model_wrapper: GenerationWrapper = MODEL_WRAPPER_CLASSES[generation_source](dotenv)

    print("Loading output dataset...")
    if restart:
        output_dataset = Dataset.from_dict(task.empty_dataset_format)
    else:
        try:
            output_dataset = cast(
                Dataset,
                load_dataset(
                    f"{task.output_dataset_org}/{task.output_dataset_name}",
                    split="train",
                ),
            )
        except (EmptyDatasetError, ValueError):
            print("No existing dataset found, starting from scratch...")
            output_dataset = Dataset.from_dict(task.empty_dataset_format)

    print("Loading input dataset...")
    input_dataset: Dataset
    if task.seed_data_format == SeedDataFormat.HF_DATASET:
        assert task.seed_data_location
        input_dataset = cast(
            Dataset, load_dataset(task.seed_data_location, split="train")
        )
    elif task.seed_data_format == SeedDataFormat.TSV:
        input_dataset = cast(
            Dataset, load_dataset("tsv", data_files=task.seed_data_location)
        )
    elif task.seed_data_format == SeedDataFormat.SYNTHETIC and task.dpo_task_cache_dataset_name:
        input_dataset = cast(
            Dataset, load_dataset(task.dpo_task_cache_dataset_name, split="train")
        )

    new_dataset_rows: List[Dict] = []
    print("Running...")

    if generate_dpo_pairs:
        # Generate negative pairs for toolformer
        # task, definition, tool_call, call_result, agent_output
        for batch_idx, batch in enumerate(
            input_dataset.iter(batch_size=batch_size)
        ):
            full_conversations_batch = []
            new_rows_batch = []
            original_rows = []

            print(f"Generating {len(full_conversations_batch)} completions...")
            completions = asyncio.run(
                model_wrapper.generate(full_conversations_batch)
            )

            for j, completion in enumerate(completions):
                try:
                    row = extract_toolformer_dpo_row(completion, original_rows[j])

                    assert_valid_python_code(row.tool_call_accepted)
                    assert_valid_python_code(row.tool_call_rejected)

                    row_dict = row.__dict__
                    new_rows_batch.append(row_dict)
                except Exception as e:
                    traceback.print_exc()
                    continue
            print_result_dicts(new_rows_batch)
            new_dataset_rows.extend(new_rows_batch)
            if batch_idx % upload_every == 0 and batch_idx > 0:
                upload_dataset(
                    output_dataset, task.output_dataset_name, new_dataset_rows
                )

if __name__ == "__main__":
    fire.Fire(main)
