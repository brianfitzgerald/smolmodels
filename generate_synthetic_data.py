import asyncio
import os
import random
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, cast

import fire
import pandas as pd
from datasets import Dataset, load_dataset
from datasets.data_files import EmptyDatasetError
from dotenv import dotenv_values
from huggingface_hub import login
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from synthetic_data.tasks import (
    PromptUpsample,
    SyntheticDataTask,
    Toolformer,
    Toolformer,
)

from synthetic_data.conversion import chatml_to_conversation
from synthetic_data.generation import (
    SHAREGPT_TO_OPENAI_ROLE,
    Conversation,
    GenerationWrapper,
    GroqGenerationWrapper,
    OpenAIGenerationWrapper,
    OpenRouterGenerationWrapper,
    VLLMWrapper,
    upload_dataset,
)
from synthetic_data.prompts import (
    TOOL_USE_CATEGORIES,
    format_dalle_prompt_template,
    get_tool_usage_prompt,
    get_toolformer_dpo_negative_completion_prompt,
    get_toolformer_prompt,
)
from synthetic_data.utils import (
    DatasetTaskFormat,
    GenerationSource,
    SeedDataFormat,
    ToolFormerDPORow,
    ToolFormerRow,
    assert_valid_python_code,
    clean_message,
    extract_tool_usage_dpo_row,
    extract_toolformer_dpo_row,
    extract_toolformer_row,
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
    n_epochs: int = 10,
    restart: bool = False,
    generate_pairs: bool = False,
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

    if generate_pairs and task.dataset_task_format != DatasetTaskFormat.DPO:
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

    new_dataset_rows: List[Dict] = []
    print("Running...")

    if generate_pairs:
            # Generate negative pairs for toolformer
            # task, definition, tool_call, call_result, agent_output
            for batch_idx, batch in enumerate(
                input_dataset.iter(batch_size=batch_size)
            ):
                full_conversations_batch = []
                new_rows_batch = []
                original_rows = []
                # TODO chance of dropping out tool definition
                for conversation in batch["conversations"]:  # type: ignore
                    messages = [message["content"] for message in conversation]
                    question, tool_call, call_result, agent_output = messages
                    original_row = ToolFormerRow(
                        question=question,
                        call_result=call_result,
                        tool_call=tool_call,
                        agent_output=agent_output,
                    )
                    original_rows.append(original_row)
                    conversation = get_toolformer_dpo_negative_completion_prompt(
                        question
                    )
                    full_conversations_batch.append(conversation)

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

        else:
            raise ValueError(
                "generate_dpo_negative_pairs is only supported for TOOLFORMER."
            )
    else:
        for epoch_idx in range(n_epochs):

            # Seed dataset, so generate the prompt and negative sample
            prompt_conversations: List[Conversation] = []
            if task:
                seed_data = pd.read_csv(task.seed_data_location, on_bad_lines="skip")
                num_batches = len(seed_data) // batch_size + 1

                # Iterate through batches
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(seed_data))
                    seed_data_batch = seed_data.iloc[start_idx:end_idx]

                    for _, seed_data_row in seed_data_batch.iterrows():
                        prompt_conversations.append(
                            get_tool_usage_prompt(
                                seed_data_row["Category"],
                                seed_data_row["Task"],
                            )
                        )
            elif config.dataset_task == DatasetTask.TOOLFORMER:
                # Iterate through batches
                random_categories = random.sample(
                    TOOL_USE_CATEGORIES * batch_size, batch_size
                )
                for category in random_categories:
                    prompt_conversations.append(get_toolformer_prompt(category))

            # Generate completions

            try:
                print(
                    f"Generating {len(prompt_conversations)} completions for epoch {epoch_idx}..."
                )
                completions = asyncio.run(model_wrapper.generate(prompt_conversations))
            except Exception as e:
                print(f"Error generating completions: {e}")
                continue

            new_rows_batch = []
            for completion in completions:
                try:
                    if config.dataset_task == DatasetTask.TOOL_USAGE_DPO:
                        row = extract_tool_usage_dpo_row(completion)

                        assert_valid_python_code(row.definition)
                        assert_valid_python_code(row.call_result)
                        assert_valid_python_code(row.tool_call)

                        new_rows_batch.append(
                            {
                                "tool": row.definition,
                                "question": row.task,
                                "call_result": row.call_result,
                                "tool_call": row.tool_call,
                                "agent_output": row.agent_output,
                            }
                        )
                    elif config.dataset_task == DatasetTask.TOOLFORMER:
                        row = extract_toolformer_row(completion)
                        reformatted_conversation = [
                            {"role": "user", "content": row.question},
                            {"role": "tool", "content": row.tool_call},
                            {"role": "tool", "content": row.call_result},
                            {"role": "assistant", "content": row.agent_output},
                        ]
                        # TODO validate output
                        new_rows_batch.append(
                            {
                                "conversations": reformatted_conversation,
                            }
                        )
                except Exception as e:
                    traceback.print_exc()
                    continue

            dict_of_steps = [
                {
                    f"step_{index}": value["content"]
                    for index, value in enumerate(row["conversations"])
                }
                for row in new_rows_batch
            ]
            print_result_dicts(dict_of_steps)
            new_dataset_rows.extend(new_rows_batch)

            if epoch_idx % upload_every == 0 and epoch_idx > 0:
                upload_dataset(
                    output_dataset, config.output_dataset_name, new_dataset_rows
                )

            # Generate completions for existing prompts
            else:
                for i, batch in enumerate(input_dataset.iter(batch_size=batch_size)):
                    new_rows_batch = []
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
                            new_rows_batch.append(
                                {
                                    "Prompt": original_prompt,
                                    "Category": category,
                                    "Upsampled": output,
                                }
                            )

                            print(
                                f"Epoch: {epoch_idx} idx: {i} ({category}): {original_prompt} -> {output}"
                            )
                    elif config.dataset_task == DatasetTask.TOOL_USAGE_DPO:
                        full_conversations_batch: List[
                            List[ChatCompletionMessageParam]
                        ] = []

                        glaive_conversations = [chatml_to_conversation(chat, system) for chat, system in zip(batch["chat"], batch["system"])]  # type: ignore
                        prompt_conversations: List[Conversation] = []
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
                            prompt_conversations.append(completion_conv)

                        print(
                            f"Generating {len(prompt_conversations)} completions for batch {i}..."
                        )
                        completions = asyncio.run(
                            model_wrapper.generate(prompt_conversations)
                        )

                        for completion, glaive_conversation in zip(
                            completions, glaive_conversations
                        ):

                            system_msg, user_msg, accepted_msg, rejected_msg = (
                                "",
                                "",
                                "",
                                "",
                            )
                            for msg in glaive_conversation:
                                role, content = msg["from"], msg["value"]
                                if role == "system":
                                    system_msg = clean_message(content)
                                if role == "human":
                                    user_msg = clean_message(content)
                                if role == "gpt":
                                    accepted_msg = clean_message(content)
                                    rejected_msg = completion
                                    break
                            new_rows_batch.append(
                                {
                                    "system": system_msg,
                                    "question": user_msg,
                                    "chosen": accepted_msg,
                                    "rejected": rejected_msg,
                                }
                            )

                        print_result_dicts(new_rows_batch)
                        new_dataset_rows.extend(new_rows_batch)

            if i % upload_every == 0 and i > 0:
                upload_dataset(
                    output_dataset, config.output_dataset_name, new_dataset_rows
                )


if __name__ == "__main__":
    fire.Fire(main)
