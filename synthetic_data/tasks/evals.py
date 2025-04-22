import asyncio
from evaluation.code_execution import EvalResult, EvalTask
from synthetic_data.generation import GenerationWrapper, get_generation_wrapper
from synthetic_data.utils import Conversation, flatten_list, ldictl
from datasets import Dataset
import json
import re
from loguru import logger

from synthetic_data.writing_judge import CreativeWritingBench


def parse_scores(judge_model_response):
    scores = {}

    # Parse scores using regex
    score_pattern = r"(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)"
    matches = re.findall(score_pattern, judge_model_response)

    for match in matches:
        metric_name = match[0].strip()
        score = float(match[1])
        scores[metric_name] = score

    return scores


class EQBenchWriting(EvalTask):
    def __init__(self):
        super().__init__(
            name="eq_bench_writing",
        )
        self.judge_model = get_generation_wrapper("gpt-4o")
        self.n_iters = 5
        self.judge = CreativeWritingBench("cli")

    def load_task_data(self) -> Dataset:
        questions: dict = {}
        with open(
            "./data/creative_writing_prompts_v2.2.json", "r", encoding="utf-8"
        ) as f:
            questions = json.load(f)

        questions_items_list = list(questions.values())
        questions_items_list = ldictl(questions_items_list)
        return Dataset.from_dict(questions_items_list)

    async def run_eval(  # type: ignore
        self, batch: dict, generation_wrapper: GenerationWrapper
    ) -> list[EvalResult]:
        # Run inference to get completions
        writing_prompt: list[str] = batch["writing_prompt"]
        seed_modifiers: list[list[str]] = batch["seed_modifiers"]

        async def process_iteration(i: int) -> list[EvalResult]:
            # TODO check that seed is correct
            modified_prompts = [
                prompt.replace("<SEED>", seed[j])
                for j, (prompt, seed) in enumerate(
                    zip(writing_prompt, zip(*seed_modifiers))
                )
            ]
            writing_convs: list[Conversation] = [
                [{"role": "user", "content": prompt}] for prompt in modified_prompts
            ]
            logger.info("Generating samples")
            generated_samples = await generation_wrapper.generate(
                writing_convs,
            )
            judge_convs: list[Conversation] = [
                [
                    {
                        "role": "user",
                        "content": self.judge.format_prompt(sample, model_response),
                    }
                ]
                for sample, model_response in zip(writing_prompt, generated_samples)
            ]
            logger.info("Judging samples")
            judge_samples = await self.judge_model.generate(
                judge_convs,
            )
            iteration_scores = [parse_scores(sample) for sample in judge_samples]
            return [
                EvalResult(
                    prompt=writing_prompt[i],
                    model_response=generated_samples[i],
                    scores=iteration_scores[i],
                )
                for i in range(self.n_iters)
            ]

        # Run all iterations in parallel
        results = await asyncio.gather(
            *(process_iteration(i) for i in range(self.n_iters))
        )

        return flatten_list(results)
