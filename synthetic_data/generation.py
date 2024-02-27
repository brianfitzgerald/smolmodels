from abc import ABC, abstractmethod
import openai
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion
from typing import List, Dict, Optional, TypedDict
from datasets import Dataset, concatenate_datasets
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import asyncio

Conversation = List[ChatCompletionMessageParam]
ShareGPTConversation = List[Dict[str, str]]

SHAREGPT_TO_OPENAI_ROLE = {
    "system": "system",
    "human": "user",
    "gpt": "assistant",
}


def upload_dataset(
    hf_dataset: Dataset, dataset_name: str, new_dataset_rows: List[Dict]
):
    dataset_new_rows = Dataset.from_list(new_dataset_rows)
    dataset_new_rows.to_csv("upsampled_new_prompts.csv")

    concat_dataset = concatenate_datasets([hf_dataset, dataset_new_rows])

    print(f"Uploading {len(new_dataset_rows)} new prompts to the Hub...")
    concat_dataset.push_to_hub(dataset_name)


class GenerationWrapper(ABC):
    """
    Abstract method for various ways of generating data.
    """

    tokenizer: PreTrainedTokenizerBase

    @abstractmethod
    async def generate(self, conversations: List[Conversation]):
        pass


class VLLMWrapper(GenerationWrapper):

    def __init__(self):

        from vllm import LLM, SamplingParams  # type: ignore

        self.sampling_params = SamplingParams(
            temperature=0.7, top_p=0.95, max_tokens=256
        )
        print("Loading local pipeline...")
        self.model = LLM(model="HuggingFaceH4/zephyr-7b-beta", dtype="auto")
        print("Pipeline loaded.")
        self.tokenizer = self.model.get_tokenizer()

    async def generate(self, conversations: List[Conversation]):
        full_conversation_formatted = self.tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=True  # type: ignore
        )
        return self.model.generate(full_conversation_formatted, self.sampling_params)


class OpenAIGenerationWrapper(GenerationWrapper):

    def __init__(self, api_key: Optional[str]):
        if api_key is None:
            raise ValueError("OpenAI API key is required for OpenAIGenerationWrapper")
        self.oai_client = openai.AsyncOpenAI(api_key=api_key)

    async def generate(self, conversations: List[Conversation]) -> List[str]:
        completion_requests = []
        for conversation in conversations:
            request = self.oai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=conversation,
                temperature=0,
                max_tokens=1024,
            )
            completion_requests.append(request)
        results: List[ChatCompletion] = await asyncio.gather(*completion_requests)
        completions = [
            result.choices[0].message.content
            for result in results
            if result.choices[0].message.content is not None
        ]
        return completions