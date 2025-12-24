import json
import random
from typing import Dict, List

from loguru import logger
from pydantic import ValidationError

from synthetic_data.prompts import (
    format_dalle_prompt_template,
    format_entity_extraction_conversation_template,
    format_goody_prompt_template,
    get_toolformer_dpo_negative_completion_prompt,
)
from synthetic_data.tasks import BaseTaskV1
from synthetic_data.tools import (
    DROPOUT_TYPES_TOOLFORMER,
    get_tool_descriptions,
)
from synthetic_data.utils import (
    Conversation,
    DatasetFormat,
    ExtractiveQARow,
    ToolFormerDPORow,
    ToolFormerRow,
    extract_code_block,
    get_matches,
    is_valid_python,
)


class DPOTask(BaseTaskV1):
    n_dpo_completions: int = 2


class PromptUpsample(BaseTaskV1):
    seed_data_format = DatasetFormat.TSV
    seed_data_location = "gs://openai-datasets/prompt-upsample/seed-data.tsv"
    output_dataset_name = "prompt-upsample"
    output_dataset_org = "openai"

    dataset_columns = ["Prompt", "Category", "Upsampled"]

    def format_input_conversation(self, batch: List[Dict]) -> List[Conversation]:
        prompts = [item["Prompt"] for item in batch]
        return [format_dalle_prompt_template(prompt) for prompt in prompts]


class Toolformer(DPOTask):
    seed_data_format = DatasetFormat.SYNTHETIC
    seed_data_location = "seed_data_files/domain_specific_tasks.csv"

    output_dataset_name = "synthetic-toolformer-dpo"
    dataset_org = "roborovski"

    dataset_columns = ["system", "question", "chosen", "rejected"]

    original_rows_batch: List[ToolFormerRow] = []

    def format_output_rows(
        self, completions: List[str], input_rows: List[Dict]
    ) -> List[Dict]:
        row = None
        for completion in completions:
            question, tool_call, call_result, agent_output = get_matches(completion)
            row = ToolFormerRow(question, tool_call, call_result, agent_output)

            assert is_valid_python(row.call_result)
            assert is_valid_python(row.tool_call)

        assert row is not None

        return [
            {
                "call_result": row.call_result,
                "tool_call": row.tool_call,
                "agent_output": row.agent_output,
            }
        ]

    # TODO re add the original row to the input conv
    def format_input_conversation(self, batch: List[Dict]) -> List[Conversation]:
        conversations = [item["conversations"] for item in batch]

        dropout_types_batch = random.choices(
            DROPOUT_TYPES_TOOLFORMER, k=len(conversations)
        )

        conversations_batch: List[Conversation] = []
        original_rows_batch: List[ToolFormerRow] = []
        for i, conversation in enumerate(conversations):
            messages = [message["content"] for message in conversation]
            for _ in range(self.n_dpo_completions):
                question, tool_call, call_result, agent_output = messages
                original_row = ToolFormerRow(
                    question=question,
                    call_result=call_result,
                    tool_call=tool_call,
                    agent_output=agent_output,
                )
                tool_descriptions = get_tool_descriptions(dropout_types_batch[i])
                conversation = get_toolformer_dpo_negative_completion_prompt(
                    question, tool_descriptions
                )
                conversations_batch.append(conversation)
                original_rows_batch.append(original_row)
        self.original_rows_batch = original_rows_batch
        return conversations_batch

    def get_dpo_dataset_output_batch(self, completions_batch: List[str]) -> List[Dict]:
        # Get completions for each prompt, rank, and choos the 2 highest
        new_rows_batch = []
        dpo_rows_batch: List[ToolFormerRow] = []
        for completion, original_row in zip(
            completions_batch, self.original_rows_batch
        ):
            try:
                tool_call, call_result, agent_output = get_matches(completion)

                dpo_row = ToolFormerRow(
                    original_row.question,
                    call_result,
                    tool_call,
                    agent_output,
                )

                dpo_rows_batch.append(dpo_row)

                if dpo_row.tool_call == original_row.tool_call:
                    continue

                output_row = ToolFormerDPORow(
                    question=original_row.question,
                    call_result_accepted=original_row.call_result,
                    tool_call_accepted=original_row.tool_call,
                    agent_output_accepted=original_row.agent_output,
                    call_result_rejected=call_result,
                    tool_call_rejected=tool_call,
                    agent_output_rejected=agent_output,
                )

                row_dict = output_row.__dict__
                new_rows_batch.append(row_dict)
            except Exception as e:
                logger.error(f"Error in parsing completion: {e}")
                continue

        return new_rows_batch


class SquadExtractiveQA(BaseTaskV1):
    """
    Performs the following steps:
    - Generate JSON struct from the prompt.
    - Dropout certain fields from the JSON struct.
    - Save the original JSON struct and the dropout JSON struct.
    """

    seed_data_format = DatasetFormat.HF_DATASET
    seed_data_location = "rajpurkar/squad_v2"
    output_dataset_name = "squad-extractive-qa"
    output_dataset_org = "roborovski"

    dataset_columns = ["id", "context", "json_schema", "fields"]

    def format_input_conversation(self, batch: list[dict]) -> list[Conversation]:
        prompts = [item["context"] for item in batch]
        self.contexts = [item["context"] for item in batch]
        return [
            format_entity_extraction_conversation_template(prompt) for prompt in prompts
        ]

    def format_output_rows(
        self, completions: List[str], input_rows: list[dict]
    ) -> List[Dict]:
        parsed_rows: List[ExtractiveQARow] = []
        for i, completion in enumerate(completions):
            blocks = extract_code_block(completion)
            if len(blocks) != 2:
                logger.warning(f"Could not extract JSON from completion: {completion}")
                continue
            json_data, json_query = blocks
            if not isinstance(json_data, dict) or not isinstance(json_query, dict):
                logger.warning(f"Invalid JSON schema for completion: {completion}")
                continue
            field_names = set(json_data.keys()).union(json_query.keys())

            if len(field_names) == 0:
                logger.warning(f"Empty JSON schema for completion: {completion}")
                continue

            try:
                qa_row = ExtractiveQARow(
                    self.contexts[i],
                    json_query,
                    json_data,
                )
            except ValidationError as e:
                logger.warning(f"Error in formatting completion: {e}")
                continue
            parsed_rows.append(qa_row)

        out_rows = []
        for row in parsed_rows:
            row = row.__dict__
            for key in ["json_query", "json_data"]:
                row[key] = json.dumps(row[key])
            out_rows.append(row)
        return out_rows


class Goody2(BaseTaskV1):
    seed_data_format = DatasetFormat.HF_DATASET
    seed_data_location = "yahma/alpaca-cleaned"
    dataset_columns = ["instruction", "response"]
    output_dataset_name = "open-goody2"

    def format_input_conversation(self, batch: list[dict]) -> list[Conversation]:
        self.instructions = [item["instruction"] for item in batch]
        return [
            format_goody_prompt_template(instruction)
            for instruction in self.instructions
        ]

    def format_output_rows(
        self, completions: List[str], input_rows: list[dict]
    ) -> List[Dict]:
        res = []
        for completion, instruction in zip(completions, self.instructions):
            res.append(
                {
                    "instruction": instruction,
                    "response": completion,
                }
            )
        return res
