from abc import ABC, abstractmethod
import openai
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from typing import List, Dict
from datasets import Dataset, concatenate_datasets
from tokenizers import Tokenizer


def upload_dataset(
    hf_dataset: Dataset, hf_dataset_name: str, new_dataset_rows: List[Dict]
):
    dataset_new_rows = Dataset.from_list(new_dataset_rows)
    dataset_new_rows.to_csv("upsampled_new_prompts.csv")

    concat_dataset = concatenate_datasets([hf_dataset, dataset_new_rows])

    print(f"Uploading {len(new_dataset_rows)} new prompts to the Hub...")
    concat_dataset.push_to_hub(hf_dataset_name)


class GenerationWrapper(ABC):
    """
    Abstract method for various ways of generating data.
    """

    tokenizer: Tokenizer

    @abstractmethod
    async def generate(self, messages: List[ChatCompletionMessageParam]):
        pass


class VLLMWrapper(GenerationWrapper):

    def __init__(self):

        from vllm import LLM, SamplingParams # type: ignore

        self.sampling_params = SamplingParams(
            temperature=0.7, top_p=0.95, max_tokens=256
        )
        print("Loading local pipeline...")
        self.model = LLM(model="HuggingFaceH4/zephyr-7b-beta", dtype="auto")
        print("Pipeline loaded.")
        self.tokenizer = self.model.get_tokenizer()

    async def generate(self, messages: List[ChatCompletionMessageParam]):
        return self.model.generate(messages, self.sampling_params)


class OpenAIGenerationWrapper(GenerationWrapper):

    def __init__(self):
        self.oai_client = openai.AsyncOpenAI()

    async def generate(self, messages: List[ChatCompletionMessageParam]):
        result = await self.oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
        )
        return result.choices[0].message.content
