from evaluation.code_execution import EvalTask
from synthetic_data.generation import GenerationWrapper
from datasets import Dataset, load_dataset
import json


class EQBenchWriting(EvalTask):
    def __init__(self):
        super().__init__(
            name="eq_bench_writing",
        )

    def load_task_data(self) -> Dataset:
        questions: dict = {}
        with open(
            "data/creative_writing_prompts_v2.2.json", "r", encoding="utf-8"
        ) as f:
            questions = json.load(f)

        questions_ls = list(questions.values())
        return Dataset.from_dict(questions_ls)

    async def run_eval(self, generation_wrapper: GenerationWrapper):
        pass
