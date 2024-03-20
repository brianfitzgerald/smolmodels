import asyncio
import os
from typing import Dict, List, Optional, cast
import pandas as pd
import asyncio
from synthetic_data.tasks import (
    FamousFiguresLabeling,
    SafetyLabeling,
    SyntheticDataTask,
)


import fire
from datasets import Dataset, load_dataset
from datasets.data_files import EmptyDatasetError
from dotenv import dotenv_values
from huggingface_hub import login
from synthetic_data.tasks import (
    PromptUpsample,
    PromptRewritingDPO,
    SyntheticDataTask,
    Toolformer,
    SyntheticToolCalls,
    save_dataset,
)

from synthetic_data.generation import (
    GenerationWrapper,
    GroqGenerationWrapper,
    OpenAIGenerationWrapper,
    OpenRouterGenerationWrapper,
    VLLMWrapper,
)
from synthetic_data.utils import (
    DatasetTaskFormat,
    GenerationSource,
    DatasetFormat,
    print_result_dicts,
)

DATA_TASKS: Dict[str, type[SyntheticDataTask]] = {
    "toolformer": Toolformer,
    "prompt_upsample": PromptUpsample,
    "synthetic_tool_calls": SyntheticToolCalls,
    "prompt_rewriting_dpo": PromptRewritingDPO,
    "safety_labeling": SafetyLabeling,
    "famous_figures_labeling": FamousFiguresLabeling,
}

MODEL_WRAPPER_CLASSES = {
    GenerationSource.OPENAI: OpenAIGenerationWrapper,
    GenerationSource.VLLM: VLLMWrapper,
    GenerationSource.OPENROUTER: OpenRouterGenerationWrapper,
    GenerationSource.GROQ: GroqGenerationWrapper,
}


def main(
    # n batches
    save_every: int = 10,
    batch_size: int = 19,
    restart: bool = False,
    pairs: bool = False,
    resume_input_position: bool = True,
    generation_source: GenerationSource = GenerationSource.OPENROUTER,
    task_name: str = "famous_figures_labeling",
    n_epochs: int = 10,
    **kwargs,
):
    """
    Generate synthetic preference data from a given dataset.
    Inputs a seed dataset, that is either given from a CSV or HF dataset,
    or generated from a synthetic source, such as a list of subjects.

    Each task follows the format of:
    - Format conversations to generate completions from seed data
    - Generate completions using the LLM, then format them into dataset rows

    For DPO tasks, we then:
    - Format conversation to generate negative completions from the generated data, possibly performing dropout or perturbation of input data
    - Generate completions and format negative and positive pairs into a single dataset row
    So that each output row contains a prompt, and two completions, one positive and one negative.

    Arguments:
    - save_every: Upload or save the dataset every n batches.
    - batch_size: The batch size to use for generation.
    - restart: Restart the dataset from scratch.
    - pairs: Generate DPO pairs.
    - resume_input_position: Resume generating from the current row in the input dataset, based on the output dataset length.
    """
    assert not kwargs, f"Unrecognized arguments: {kwargs}"

    task = DATA_TASKS[task_name]()

    if pairs and task.dataset_task_format != DatasetTaskFormat.DPO:
        raise ValueError("generate_pairs is only supported for DPO tasks.")

    print("Logging into the Hub...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv = dotenv_values(os.path.join(current_dir, ".env"))
    hf_token = dotenv["HF_TOKEN"]
    print(f"Logging in with token: {hf_token}")
    login(token=hf_token, add_to_git_credential=True)

    model_wrapper: GenerationWrapper = MODEL_WRAPPER_CLASSES[generation_source](dotenv)

    empty_dataset_format = (
        task.empty_dpo_dataset_format if pairs else task.empty_dataset_format
    )

    print("Loading output dataset...")
    if restart:
        output_dataset = Dataset.from_dict(empty_dataset_format)
    elif task.output_data_format == DatasetFormat.HF_DATASET:
        try:
            output_dataset = cast(
                Dataset,
                load_dataset(
                    f"{task.dataset_org}/{task.output_data_name}",
                    split="train",
                ),
            )
            # TODO filter for rows that don't need completion
        except (EmptyDatasetError, ValueError):
            print("No existing dataset found, starting from scratch...")
            output_dataset = Dataset.from_dict(empty_dataset_format)
    else:
        output_dataset = Dataset.from_dict(empty_dataset_format)

    input_dataset: Dataset
    input_dataset_location: Optional[str] = None
    if task.dpo_seed_cache_dataset_name and pairs:
        input_dataset_location = (
            f"{task.dataset_org}/{task.dpo_seed_cache_dataset_name}"
        )
    elif task.seed_data_format == DatasetFormat.HF_DATASET:
        input_dataset_location = f"{task.dataset_org}/{task.seed_data_location}"
    elif task.seed_data_format in (DatasetFormat.CSV, DatasetFormat.PARQUET):
        input_dataset_location = task.seed_data_location
    assert input_dataset_location, "No input dataset location provided."

    print(
        f"Loading input dataset: {input_dataset_location}, format: {task.seed_data_format.value}"
    )
    split = "train"
    if (
        task.seed_data_format in (DatasetFormat.HF_DATASET, DatasetFormat.SYNTHETIC)
        or pairs
    ):
        if len(output_dataset) > 0 and resume_input_position:
            split = f"train[{len(output_dataset)}:]"
        input_dataset = cast(Dataset, load_dataset(input_dataset_location, split=split))
    elif task.seed_data_format == DatasetFormat.CSV:
        if input_dataset_location.endswith(".tsv"):
            seed_data = pd.read_csv(
                input_dataset_location, sep="\t", on_bad_lines="skip"
            )
        else:
            seed_data = pd.read_csv(input_dataset_location, on_bad_lines="skip")
        input_dataset = Dataset.from_pandas(seed_data)
    elif task.seed_data_format == DatasetFormat.PARQUET:
        seed_data = pd.read_parquet(input_dataset_location)
        input_dataset = Dataset.from_pandas(seed_data)
    else:
        raise ValueError(f"Unrecognized seed_data_format: {task.seed_data_format}")

    input_dataset = task.preprocess_dataset(input_dataset)

    print(f"Input dataset length: {len(input_dataset)} output: {len(output_dataset)}")
    new_dataset_rows: List[Dict] = []
    print("Running...")

    # Test save
    save_dataset(output_dataset, task, [])

    total_batches = len(input_dataset) // batch_size

    for epoch in range(n_epochs):
        # Generate negative pairs for DPO
        if pairs:
            for batch_idx, batch in enumerate(
                input_dataset.iter(batch_size=batch_size)
            ):
                batch = cast(Dict, batch)
                full_conversations_batch = task.format_dpo_input_conversations(batch)

                print(f"Generating {len(full_conversations_batch)} completions...")
                completions = asyncio.run(
                    model_wrapper.generate(full_conversations_batch)
                )

                output_rows_batch = task.get_dpo_dataset_output_batch(completions)
                print_result_dicts(output_rows_batch)
                new_dataset_rows.extend(output_rows_batch)
                if batch_idx % save_every == 0 and batch_idx > 0:
                    save_dataset(output_dataset, task, new_dataset_rows)
        else:
            for batch_idx, batch in enumerate(
                input_dataset.iter(batch_size=batch_size)
            ):
                batch = cast(Dict, batch)
                full_conversations_batch = task.format_seed_input_conversation(batch)

                print(
                    f"Epoch {epoch}, batch {batch_idx}/{total_batches}: generating {len(full_conversations_batch)} completions ({len(new_dataset_rows)}/{len(input_dataset)} total rows)"
                )
                completions = asyncio.run(
                    model_wrapper.generate(full_conversations_batch)
                )

                output_rows_batch = task.get_seed_dataset_output_rows_batch(completions)
                print_result_dicts(output_rows_batch)
                new_dataset_rows.extend(output_rows_batch)
                if batch_idx % save_every == 0 and batch_idx > 0:
                    save_dataset(output_dataset, task, new_dataset_rows)


if __name__ == "__main__":
    fire.Fire(main)
