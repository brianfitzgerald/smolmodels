from abc import ABC
from enum import Enum
import random
from typing import Dict, List, Optional
from synthetic_data.conversion import chatml_to_conversation
from synthetic_data.generation import SHAREGPT_TO_OPENAI_ROLE
from synthetic_data.prompts import (
    TOOL_USE_CATEGORIES,
    format_dalle_prompt_template,
    get_toolformer_dpo_negative_completion_prompt,
    get_toolformer_prompt,
)
from synthetic_data.tools import TOOL_FUNCTIONS

from synthetic_data.utils import (
    Conversation,
    DatasetTaskFormat,
    SeedDataFormat,
    ToolFormerDPORow,
    ToolFormerRow,
    ToolUsageDPORow,
    get_fn_call_metadata,
    is_valid_python,
    clean_message,
    get_matches,
)


class SyntheticDataTask(ABC):

    seed_data_format: SeedDataFormat
    dataset_task_format: DatasetTaskFormat = DatasetTaskFormat.SFT

    # Only used for the toolformer DPO dataset. # TODO remove this when that project is finished.
    seed_data_uses_conversation_format: bool = False

    # Name for the dataset used to cache the seed data.
    # Once all the seed data is generated, this dataset will be used to cache the seed data.
    dpo_seed_cache_dataset_name: Optional[str] = None

    no_dpo_completions: int = 2

    seed_data_location: str
    output_dataset_name: str
    dataset_org: str

    empty_dataset_format: Dict[str, List]

    def format_seed_input_conversation(self, batch: Dict) -> List[Conversation]:
        """
        Prompt template to use for generating initial seed data.
        """
        raise NotImplementedError

    def get_seed_dataset_output_row(self, completion: str) -> Dict:
        """
        Take the completed conversation and format it into the final dataset format.
        """
        raise NotImplementedError

    def format_dpo_input_conversations(self, batch: Dict) -> List[Conversation]:
        """
        Format conversation for DPO completion
        """
        raise NotImplementedError

    def get_dpo_dataset_output_batch(self, completion: List[str]) -> List[Dict]:
        """
        Take the completed conversation and format it into the final dataset format.
        """
        raise NotImplementedError


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

    def format_seed_input_conversation(self, batch: Dict) -> List[Conversation]:
        prompts = batch["Prompt"]
        return [format_dalle_prompt_template(prompt) for prompt in prompts]


class Toolformer(SyntheticDataTask):

    seed_data_format = SeedDataFormat.SYNTHETIC
    seed_data_location = "seed_data_files/domain_specific_tasks.csv"

    dataset_task_format = DatasetTaskFormat.DPO

    dpo_seed_cache_dataset_name = "synthetic-toolformer-sharegpt"
    no_dpo_completions = 3

    output_dataset_name = "synthetic-toolformer-dpo"
    dataset_org = "roborovski"

    empty_dataset_format = {
        "system": [],
        "question": [],
        "chosen": [],
        "rejected": [],
    }

    original_rows_batch: List[ToolFormerRow] = []

    def format_seed_input_conversation(self, batch_size: int) -> List[Conversation]:
        prompt_conversations: List[Conversation] = []
        random_categories = random.sample(TOOL_USE_CATEGORIES * batch_size, batch_size)
        for category in random_categories:
            prompt_conversations.append(get_toolformer_prompt(category))
        return prompt_conversations

    def get_seed_dataset_output_row(self, completions_batch: List[str]) -> Dict:
        for completion in completions_batch:
            question, tool_call, call_result, agent_output = get_matches(completion)
            row = ToolFormerRow(question, tool_call, call_result, agent_output)

            is_valid_python(row.call_result)
            is_valid_python(row.tool_call)

        return {
            "call_result": row.call_result,
            "tool_call": row.tool_call,
            "agent_output": row.agent_output,
        }

    def format_dpo_input_conversations(self, batch: Dict) -> List[Conversation]:
        # TODO chance of dropping out tool definition
        conversations_batch: List[Conversation] = []
        original_rows_batch: List[ToolFormerRow] = []
        for i, conversation in enumerate(batch["conversations"]):
            messages = [message["content"] for message in conversation]
            for _ in range(self.no_dpo_completions):
                question, tool_call, call_result, agent_output = messages
                original_row = ToolFormerRow(
                    question=question,
                    call_result=call_result,
                    tool_call=tool_call,
                    agent_output=agent_output,
                )
                conversation = get_toolformer_dpo_negative_completion_prompt(question)
                conversations_batch.append(conversation)
                original_rows_batch.append(original_row)
        self.original_rows_batch = original_rows_batch
        return conversations_batch

    def _score_dpo_completion(self, row: ToolFormerRow) -> float:
        score = 0
        try:
            fn_call = row.tool_call.replace("`", "").strip()
            fn_call = get_fn_call_metadata(fn_call)
            result = TOOL_FUNCTIONS[fn_call.fn_name](*fn_call.parameters)
        except Exception as e:
            print(f"Error in completion {row.question}: {e}")
            return 0
        if str(result) == row.call_result:
            score += 0.3
        if str(result) in row.agent_output:
            score += 0.2
        return score

    def get_dpo_dataset_output_batch(self, completions_batch: List[str]) -> List[Dict]:
        new_rows_batch = []

        # Get completions for each prompt, rank, and choos the 2 highest
        for i in range(0, len(completions_batch), self.no_dpo_completions):
            completion_batch = completions_batch[i : i + self.no_dpo_completions]
            original_rows_batch = self.original_rows_batch[
                i : i + self.no_dpo_completions
            ]
            dpo_rows_batch = []
            try:
                for completion, original_row in zip(
                    completion_batch, original_rows_batch
                ):
                    tool_call, call_result, agent_output = get_matches(completion)
                    dpo_row = ToolFormerRow(
                        original_row.question,
                        call_result,
                        tool_call,
                        agent_output,
                    )
                    dpo_rows_batch.append(dpo_row)
            except Exception as e:
                print(f"Error in completion {i}: {e}")
                continue

            row_scores = [self._score_dpo_completion(row) for row in dpo_rows_batch]

            print(f"Scores: {row_scores}")

            top_2_dpo_rows: List[ToolFormerRow] = sorted(
                dpo_rows_batch, key=self._score_dpo_completion, reverse=True
            )[:2]

            output_row = ToolFormerDPORow(
                question=top_2_dpo_rows[0].question,
                call_result_accepted=top_2_dpo_rows[0].call_result,
                tool_call_accepted=top_2_dpo_rows[0].tool_call,
                agent_output_accepted=top_2_dpo_rows[0].agent_output,
                call_result_rejected=top_2_dpo_rows[1].call_result,
                tool_call_rejected=top_2_dpo_rows[1].tool_call,
                agent_output_rejected=top_2_dpo_rows[1].agent_output,
            )

            row_dict = output_row.__dict__
            new_rows_batch.append(row_dict)

        return new_rows_batch


class GlaiveDPO(SyntheticDataTask):

    def format_seed_input_conversation(self, batch: Dict) -> List[Conversation]:
        glaive_conversations = [
            chatml_to_conversation(chat, system)
            for chat, system in zip(batch["chat"], batch["system"])
        ]
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

    def _clean_glaive_conversation(
        self, completions: List[str], glaive_conversations: List[Dict]
    ) -> List[Dict]:
        new_rows_batch = []
        for completion, glaive_conversation in zip(completions, glaive_conversations):

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

        # TODO fix
        return [
            {"role": SHAREGPT_TO_OPENAI_ROLE[msg["from"]], "content": msg["value"]}
            for msg in new_rows_batch
        ]
