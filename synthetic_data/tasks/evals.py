from evaluation.code_execution import EvalTask
from synthetic_data.generation import GenerationWrapper, get_generation_wrapper
from synthetic_data.tasks.writing import format_eq_bench_scoring_prompt
from synthetic_data.utils import Conversation, ldictl
from datasets import Dataset
import json
import re


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

    def load_task_data(self) -> Dataset:
        questions: dict = {}
        with open(
            "../data/creative_writing_prompts_v2.2.json", "r", encoding="utf-8"
        ) as f:
            questions = json.load(f)

        questions_items_list = list(questions.values())
        questions_items_list = ldictl(questions_items_list)
        return Dataset.from_dict(questions_items_list)

    async def run_eval(self, batch: dict, generation_wrapper: GenerationWrapper):
        # Run inference to get completions
        writing_prompt: list[str] = batch["writing_prompt"]
        for i, seed in enumerate(batch["seed_modifiers"]):
            writing_prompt[i] = writing_prompt[i].replace("<SEED>", seed)
        writing_convs: list[Conversation] = [
            [{"role": "user", "content": prompt} for prompt in writing_prompt]
        ]
        generated_samples = await generation_wrapper.generate(
            writing_convs,
        )

        judge_convs = [
            format_eq_bench_scoring_prompt(sample) for sample in generated_samples
        ]
        judge_samples = await self.judge_model.generate(
            judge_convs,
        )
        judge_scores = [parse_scores(sample) for sample in judge_samples]

        return judge_scores
