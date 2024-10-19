import asyncio
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, cast

from datasets import Dataset, load_dataset
from dotenv import dotenv_values
from fire import Fire
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress
from rich.syntax import Syntax

from evaluation.code_execution import evaluate_sample, print_code_snippet
from synthetic_data.generation import GeminiWrapper, GenerationWrapper
from synthetic_data.tasks import BaseTask, DollyEntityExtraction, HumanEval
from synthetic_data.utils import Conversation, lddl


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
    # ModelConfig(name="GPT-4o", wrapper=OpenAIGenerationWrapper(dotenv)),
    ModelConfig(
        name="Gemini 1.5 Flash 8b",
        wrapper=GeminiWrapper(
            "gemini-1.5-flash-8b",
            system_instruction="If asked to generate source code, always generate the code within a source block without any surrounding text.",
        ),
    ),
]


async def sample_worker(
    model_config: ModelConfig, prompt: List[Conversation], sample: Dict
):
    out = await model_config.wrapper.generate([prompt])  # type: ignore
    return out, sample


TASKS = [
    EvalTask(
        "dolly-entity-extraction",
        "roborovski/dolly-entity-extraction",
        DollyEntityExtraction,
    ),
    EvalTask("humaneval", "openai/openai_humaneval", HumanEval),
]


def _print_test_results(err: Optional[str], results: List[bool], console: Console):
    result_str = "Results: "
    if err:
        result_str += f"[red]Execution error: {err}[/red]"
    else:
        passed = sum(results)
        total = len(results)
        result_str += f"{passed}/{total} tests passed"
    console.print(result_str)


@dataclass
class EvalResult:
    prompt: str
    generated: str
    test: str
    entry_point: str
    err: Optional[str]
    evaluation_results: List[bool]


async def main(max_concurrent: int = 16, task_name: str = "humaneval"):
    eval_task = next(t for t in TASKS if t.name == task_name)
    task = eval_task.task_class()

    dataset = cast(Dataset, load_dataset(eval_task.dataset_uri))["test"]

    console = Console()

    eval_results: List[EvalResult] = []

    with Progress() as progress:
        prog_task = progress.add_task("Evaluating", total=len(dataset))
        for batch in dataset.iter(batch_size=max_concurrent):  # type: ignore
            all_futures = []
            samples_batch = lddl(batch)
            prompts_batch = [
                task.format_inference_conversation(sample) for sample in samples_batch
            ]
            for model_config in MODEL_CONFIGS:
                for prompt, sample in zip(prompts_batch, samples_batch):
                    all_futures.append(
                        asyncio.create_task(sample_worker(model_config, prompt, sample))
                    )

            results = await asyncio.gather(*all_futures)
            for result, sample in results:
                for generated in result:
                    console.print(f"Function: {sample['entry_point']}")
                    console.print(f"Canonical solution:")
                    print_code_snippet(sample["canonical_solution"], console)
                    generated_code = generated.replace("```", "").replace("python", "")
                    err, evaluation_results = evaluate_sample(
                        sample["prompt"],
                        generated_code,
                        sample["test"],
                        sample["entry_point"],
                    )
                    console.print(f"Generated solution:")
                    print_code_snippet(generated_code, console)
                    console.print(f"Test code:")
                    print_code_snippet(sample["test"], console)
                    _print_test_results(err, evaluation_results, console)
                    console.print("=" * console.size.width)
                    progress.advance(prog_task, 1)
                    eval_results.append(
                        EvalResult(
                            sample["prompt"],
                            generated_code,
                            sample["test"],
                            sample["entry_point"],
                            err,
                            evaluation_results,
                        )
                    )

    n_all_tests_passed = sum(
        sum(res.evaluation_results) == len(res.evaluation_results)
        for res in eval_results
    )
    n_tests_passed = sum(sum(res.evaluation_results) for res in eval_results)
    total_n_tests = sum(len(res.evaluation_results) for res in eval_results)
    console.print(
        f"Samples where all tests passed: {n_all_tests_passed}/{len(eval_results)}"
    )
    console.print(f"Total tests passed: {n_tests_passed}/{total_n_tests}")


Fire(main)
