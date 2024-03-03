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
    dataset_new_rows.to_csv(f"dataset_samples/{dataset_name}.csv")

    concat_dataset = concatenate_datasets([hf_dataset, dataset_new_rows])

    print(f"Uploading {len(new_dataset_rows)} new rows to the Hub...")
    concat_dataset.push_to_hub(dataset_name)


class GenerationWrapper(ABC):
    """
    Abstract method for various ways of generating data.
    """

    tokenizer: PreTrainedTokenizerBase

    @abstractmethod
    async def generate(self, conversations: List[Conversation]) -> List[str]:
        pass


class VLLMWrapper(GenerationWrapper):

    def __init__(self, dotenv: Dict[str, str]):

        from vllm import LLM, SamplingParams  # type: ignore

        self.sampling_params = SamplingParams(
            temperature=0.7, top_p=0.95, max_tokens=128
        )
        print("Loading local pipeline...")
        self.model = LLM(model="HuggingFaceH4/zephyr-7b-beta", dtype="auto")
        print("Pipeline loaded.")
        self.tokenizer = self.model.get_tokenizer()

    async def generate(self, conversations: List[Conversation]) -> List[str]:
        convs_formatted: str = [self.tokenizer.apply_chat_template(c, tokenize=False) for c in conversations]  # type: ignore
        responses = self.model.generate(convs_formatted, self.sampling_params)
        responses_text = [r.outputs[0].text for r in responses]
        responses_text = [r.replace("<|assistant|>", " ") for r in responses_text]
        return responses_text


class OpenAIGenerationWrapper(GenerationWrapper):

    def __init__(self, dotenv: Dict[str, str]):
        api_key = dotenv.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OPENAI_API_KEY is required for OpenAIGenerationWrapper")
        self.oai_client = openai.AsyncOpenAI(api_key=api_key)
        self.model_name = "gpt-3.5-turbo"

    async def generate(self, conversations: List[Conversation]) -> List[str]:
        completion_requests = []
        for conversation in conversations:
            request = self.oai_client.chat.completions.create(
                model=self.model_name,
                messages=conversation,
                temperature=0,
                max_tokens=512,
            )
            completion_requests.append(request)
        results: List[ChatCompletion] = await asyncio.gather(*completion_requests)
        completions = [
            result.choices[0].message.content
            for result in results
            if result.choices[0].message.content is not None
        ]
        return completions


class OpenRouterGenerationWrapper(OpenAIGenerationWrapper):

    def __init__(self, dotenv: Dict[str, str]):
        api_key = dotenv.get("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouterGenerationWrapper")
        self.oai_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model_name = "nousresearch/nous-hermes-2-mixtral-8x7b-dpo"
