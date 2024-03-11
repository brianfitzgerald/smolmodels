from abc import ABC
from enum import Enum
import random
from typing import Dict, List, Optional
from synthetic_data.prompts import (
    TOOL_USE_CATEGORIES,
    format_dalle_prompt_template,
    get_toolformer_dpo_negative_completion_prompt,
    get_toolformer_prompt,
)

from synthetic_data.utils import (
    Conversation,
    DatasetTaskFormat,
    SeedDataFormat,
    ToolFormerRow,
    assert_valid_python_code,
    extract_tool_usage_dpo_row,
)


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

    def get_seed_input_conversation(self, batch: Dict) -> List[Conversation]:
        """
        Prompt template to use for generating initial seed data.
        """
        raise NotImplementedError

    def post_generation_format_row(self, completion: str) -> Dict:
        """
        Take the completed conversation and format it into the final dataset format.
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

    def __init__(self) -> None:
        super().__init__()
        self.original_rows = []

    def get_seed_input_conversation(self, batch: Dict) -> List[Conversation]:
        prompts, categories_batch = batch["Prompt"], batch["Category"]
        return [
            format_dalle_prompt_template(prompt) for prompt in prompts
        ]

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

    def get_seed_input_conversation(self, batch_size: int) -> List[Conversation]:
        prompt_conversations: List[Conversation] = []
        random_categories = random.sample(TOOL_USE_CATEGORIES * batch_size, batch_size)
        for category in random_categories:
            prompt_conversations.append(get_toolformer_prompt(category))
        return prompt_conversations

    def get_seed_input_conversation(self, batch: Dict) -> List[Conversation]:
        # TODO chance of dropping out tool definition
        conversations_batch = []
        for conversation in batch["conversations"]:
            messages = [message["content"] for message in conversation]
            question, tool_call, call_result, agent_output = messages
            original_row = ToolFormerRow(
                question=question,
                call_result=call_result,
                tool_call=tool_call,
                agent_output=agent_output,
            )
            conversation = get_toolformer_dpo_negative_completion_prompt(question)
            conversations_batch.append(conversation)
        return conversations_batch

    def post_generation_format_row(self, completion: str) -> Dict:
        row = extract_tool_usage_dpo_row(completion)

        assert_valid_python_code(row.definition)
        assert_valid_python_code(row.call_result)
        assert_valid_python_code(row.tool_call)

        return {
            "tool": row.definition,
            "question": row.task,
            "call_result": row.call_result,
            "tool_call": row.tool_call,
            "agent_output": row.agent_output,
        }


class GlaiveDPO(SyntheticDataTask):

    def get_seed_input_conversation(self, batch: Dict) -> List[Conversation]:
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
        return prompt_conversations

    def _clean_glaive_conversation(self, conversation: List[Dict]) -> List[Dict]:
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

        return [
            {"role": SHAREGPT_TO_OPENAI_ROLE[msg["from"]], "content": msg["value"]}
            for msg in conversation
        ]