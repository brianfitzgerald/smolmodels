import asyncio
from evaluation.code_execution import EvalTask
from synthetic_data.generation import GenerationWrapper, get_generation_wrapper
from synthetic_data.tasks.writing import format_eq_bench_scoring_prompt
from synthetic_data.utils import Conversation, ldictl
from datasets import Dataset
import json
import re
from loguru import logger


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


neg_criteria = [
    "melodramatic",
    "shallow resolution",
    "unearned resolution",
    "simplistic moralizing",
    "shallow optimism",
    "forced optimism",
    "trite",
    "overwrought",
    "amateurish",
    "contrived",
    "uninspiring",
    "characters are too good",
    "incongruent ending positivity",
    "unearned transformations",
    "profundity over-reach",
    "amateurish descriptives",
    "clunky asides and interruptive sentence structures",
    "stilted dialogue",
    "tit-for-tat dialogue",
    "purple prose",
    "unsurprising or uncreative",
    "tell-don't-show",
    "weak dialogue",
    "meandering",
]


def get_total_score(scores: dict[str, float], RELATIVE_SCORING=False):
    if not scores:
        print("! No scores were parseable")
        return
    scoresum = 0
    for criteria, score in scores.items():
        criteria_lower = criteria.lower().strip()
        if RELATIVE_SCORING:
            if any(neg_criterion in criteria_lower for neg_criterion in neg_criteria):
                scoresum += ((-1 * score) + 10) / 2
            else:
                scoresum += (score + 10) / 2
        else:
            if any(neg_criterion in criteria_lower for neg_criterion in neg_criteria):
                scoresum += 10 - score
            else:
                scoresum += score
    score = round(10 * scoresum / len(scores))
    logger.info(f"This question score: {score}")
    return score


def calculate_creative_writing_score(
    iterations: list[dict[str, dict]],
):
    RELATIVE_SCORING = False
    prompt_scores = []  # List to hold total scores for each prompt
    iteration_averages = []  # To hold the average scores of the best half of each iteration

    for run_iter in iterations:
        for scores in run_iter.values():
            scoresum = 0

            for criteria, score in scores.items():
                criteria_lower = criteria.lower().strip()
                if RELATIVE_SCORING:
                    if any(
                        neg_criterion in criteria_lower
                        for neg_criterion in neg_criteria
                    ):
                        scoresum += ((-1 * score) + 10) / 2
                    else:
                        scoresum += (score + 10) / 2
                else:
                    if any(
                        neg_criterion in criteria_lower
                        for neg_criterion in neg_criteria
                    ):
                        scoresum += 10 - score
                    else:
                        scoresum += score
            if len(scores):
                prompt_scores.append(scoresum / len(scores))

        if len(prompt_scores) > 10:
            iteration_average = sum(prompt_scores) / len(prompt_scores)
            iteration_averages.append(iteration_average)

    # Average of iteration averages
    if iteration_averages:
        creative_writing_averaged_score = sum(iteration_averages) / len(
            iteration_averages
        )
    else:
        creative_writing_averaged_score = 0

    return round(10 * creative_writing_averaged_score, 2)


class EQBenchWriting(EvalTask):
    def __init__(self):
        super().__init__(
            name="eq_bench_writing",
        )
        self.judge_model = get_generation_wrapper("gpt-4o")
        self.n_iters = 5

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
        seed_modifiers: list[list[str]] = batch["seed_modifiers"]

        async def process_iteration(i: int) -> list[dict[str, float]]:
            logger.info(f"Iteration {i}")
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
            logger.info(f"writing_convs: {writing_convs}")
            generated_samples = await generation_wrapper.generate(
                writing_convs,
            )
            logger.info(f"generated_samples: {generated_samples}")
            judge_convs = [
                format_eq_bench_scoring_prompt(sample) for sample in generated_samples
            ]
            judge_samples = await self.judge_model.generate(
                judge_convs,
            )
            iteration_scores = [parse_scores(sample) for sample in judge_samples]
            logger.info(f"iteration_scores: {iteration_scores}")
            return iteration_scores

        # Run all iterations in parallel
        judge_scores = await asyncio.gather(
            *(process_iteration(i) for i in range(self.n_iters))
        )

        return judge_scores
