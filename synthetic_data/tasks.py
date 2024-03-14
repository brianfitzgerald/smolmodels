from abc import ABC
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
from synthetic_data.tools import (
    DROPOUT_TYPES,
    TOOL_FUNCTIONS,
    get_fn_call_metadata,
    get_tool_descriptions,
)

from synthetic_data.utils import (
    Conversation,
    DatasetTaskFormat,
    SeedDataFormat,
    SyntheticToolCallRow,
    SyntheticToolCallDPORow,
    ToolFormerDPORow,
    ToolFormerRow,
    is_valid_python,
    clean_message,
    get_matches,
)


class SyntheticDataTask(ABC):

    seed_data_format: SeedDataFormat = SeedDataFormat.SYNTHETIC
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
        tool_descriptions = get_tool_descriptions()
        for category in random_categories:
            prompt_conversations.append(
                get_toolformer_prompt(category, tool_descriptions)
            )
        return prompt_conversations

    def get_seed_dataset_output_row(self, completions_batch: List[str]) -> Dict:
        for completion in completions_batch:
            question, tool_call, call_result, agent_output = get_matches(completion)
            row = ToolFormerRow(question, tool_call, call_result, agent_output)

            assert is_valid_python(row.call_result)
            assert is_valid_python(row.tool_call)

        return {
            "call_result": row.call_result,
            "tool_call": row.tool_call,
            "agent_output": row.agent_output,
        }

    def format_dpo_input_conversations(self, batch: Dict) -> List[Conversation]:

        conversations = batch["conversations"]

        dropout_types_batch = random.choices(DROPOUT_TYPES, k=len(conversations))

        conversations_batch: List[Conversation] = []
        original_rows_batch: List[ToolFormerRow] = []
        for i, conversation in enumerate(conversations):
            messages = [message["content"] for message in conversation]
            for _ in range(self.no_dpo_completions):
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
                print(f"Error in parsing completion: {e}")
                continue

        return new_rows_batch


class SyntheticToolCalls(SyntheticDataTask):

    dataset_task_format = DatasetTaskFormat.DPO
    seed_data_format = SeedDataFormat.SYNTHETIC

    dataset_org = "roborovski"
    dpo_seed_cache_dataset_name = "synthetic-tool-calls"

    output_dataset_name = "synthetic-tool-calls-dpo-pairs"

    empty_dataset_format = {
        "tool": [],
        "question": [],
        "call_result": [],
        "tool_call": [],
    }

    original_rows_batch: List[SyntheticToolCallRow] = []

    def format_dpo_input_conversations(self, batch: Dict) -> List[Conversation]:

        n_samples = len(batch["tool"])
        dropout_types_batch = random.choices(DROPOUT_TYPES, k=n_samples)

        conversations_batch: List[Conversation] = []
        original_rows_batch: List[SyntheticToolCallRow] = []
        for i in range(n_samples):
            for _ in range(self.no_dpo_completions):
                question = batch["question"][i]
                original_row = SyntheticToolCallRow(
                    tool=batch["tool"][i],
                    question=question,
                    tool_call=batch["tool_call"][i],
                    call_result=batch["call_result"][i],
                    agent_output=batch["agent_output"][i],
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
        dpo_rows_batch: List[SyntheticToolCallDPORow] = []
        for completion, original_row in zip(
            completions_batch, self.original_rows_batch
        ):
            try:
                tool_call, call_result, agent_output = get_matches(completion)

                dpo_rows_batch.append(dpo_row)

                if (
                    tool_call == original_row.tool_call
                    and agent_output == original_row.agent_output
                ):
                    continue

                dpo_row = SyntheticToolCallDPORow(
                    original_row.tool,
                    original_row.question,
                    original_row.tool_call,
                    original_row.call_result,
                    original_row.agent_output,
                    tool_call,
                    call_result,
                    agent_output,
                )
                row_dict = dpo_row.__dict__
                new_rows_batch.append(row_dict)

            except Exception as e:
                print(f"Error in parsing completion: {e}")
                continue

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
