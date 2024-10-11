import asyncio
from fire import Fire
from dataclasses import dataclass
from typing import Dict, List, cast
from synthetic_data.generation import (
    GenerationWrapper,
    OpenAIGenerationWrapper,
)
import os
from dotenv import dotenv_values

from rich import print as rprint

from datasets import load_dataset, Dataset

from synthetic_data.tasks import DollyEntityExtraction, HumanEval, BaseTask
from synthetic_data.utils import Conversation
from evaluation.code_execution import evaluate_sample


@dataclass
class ModelConfig:
    name: str
    wrapper: GenerationWrapper


@dataclass
class EvalTask:
    name: str
    dataset_uri: str
    task_class: type[BaseTask]


current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv: Dict[str, str] = dotenv_values(os.path.join(current_dir, ".env"))  # type: ignore


MODEL_CONFIGS = [
    ModelConfig(name="GPT-4o", wrapper=OpenAIGenerationWrapper(dotenv)),
]


async def sample_worker(model_config: ModelConfig, prompt: List[Conversation], sample: Dict):
    out = await model_config.wrapper.generate([prompt]) # type: ignore
    return out, sample


def _list_of_dicts_to_dict_of_lists(dict_of_lists):
    return [dict(zip(dict_of_lists, t)) for t in zip(*dict_of_lists.values())]


TASKS = [
    EvalTask(
        "dolly-entity-extraction",
        "roborovski/dolly-entity-extraction",
        DollyEntityExtraction,
    ),
    EvalTask("humaneval", "openai/openai_humaneval", HumanEval),
]


async def main(max_concurrent: int = 4, task_name: str = "humaneval"):
    eval_task = next(t for t in TASKS if t.name == task_name)
    task = eval_task.task_class()

    dataset = cast(Dataset, load_dataset(eval_task.dataset_uri))["test"]

    all_futures = []

    for batch in dataset.iter(batch_size=max_concurrent):  # type: ignore
        samples_batch = _list_of_dicts_to_dict_of_lists(batch)
        prompts_batch = [
            task.format_inference_conversation(sample)
            for sample in samples_batch 
        ]
        for model_config in MODEL_CONFIGS:
            for prompt, sample in zip(prompts_batch, samples_batch):
                all_futures.append(
                    asyncio.create_task(sample_worker(model_config, prompt, sample))
                )

        results = await asyncio.gather(*all_futures)
        for result, sample in results:
            for generated in result:
                rprint(f"Function: {sample['entry_point']}")
                rprint(f"Canonical solution:\n {sample['canonical_solution']}")
                rprint(f"Generated solution:\n {generated}")
                rprint(f"Tests: {sample['test']}")
                evaluation_results = evaluate_sample(
                    sample["prompt"],
                    generated.replace("```", "").replace("python", ""),
                    sample["test"],
                    sample["entry_point"],
                )
                rprint(f"Evaluation results: {evaluation_results}")


Fire(main)
