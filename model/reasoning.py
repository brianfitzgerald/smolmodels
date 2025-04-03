from typing import Optional
from datasets import load_dataset, Dataset
from transformers.trainer_callback import TrainerCallback
import re
from loguru import logger
import pandas as pd
from gyms.twenty_questions.env import GUESSER_PROMPT
from trl_wrapper.wrapper_config import SmDataset
from trl.trainer.grpo_trainer import RewardFunc
import itertools

# Based on:
# https://colab.research.google.com/drive/1bfhs1FMLW3FGa8ydvkOZyBNxLYOu0Hev?usp=sharing#scrollTo=ybtxR89X1YJq
# https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb


# Reward functions
def math_correctness_reward_func(prompts, completions, **kwargs) -> list[float]:
    model_generations = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in model_generations]
    extracted_responses_hash = [extract_hash_answer(r) for r in model_generations]
    answer = kwargs["answer"]
    rewards = [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
    rewards_hash = [
        2.0 if r == a else 0.0 for r, a in zip(extracted_responses_hash, answer)
    ]
    rewards_max = [max(r, h) for r, h in zip(rewards, rewards_hash)]
    logger.info(
        f"XML rewards: {rewards} | Hash rewards: {rewards_hash} | Max rewards: {rewards_max}"
    )
    return rewards_max


def int_reward_func(prompts, completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = [0.5 if r.isdigit() else 0.0 for r in extracted_responses]
    logger.info(f"Integer rewards: {rewards}")
    return rewards


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(prompts, completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    rewards = [count_xml(c) for c in contents]
    # clamp values to [0, 1]
    rewards = [max(0.0, min(1.0, r)) for r in rewards]
    logger.info(f"XML count rewards: {rewards}")
    return rewards


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")


# Define a custom callback class for evaluation.
class EvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_examples, device):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_examples = eval_examples
        self.device = device

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore
        if state.global_step % args.eval_steps == 0:  # type: ignore
            logger.info(f"\nEvaluating at step {state.global_step}:")
        return control


def soft_format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has the right format, with variable spacing."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    rewards = [0.5 if match else 0.0 for match in matches]
    logger.info(f"Soft format rewards: {rewards}")
    return rewards


def strict_format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has the right format, with strict spacing."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    rewards = [0.5 if match else 0.0 for match in matches]
    logger.info(f"Strict format rewards: {rewards}")
    return rewards


class GSM8KDataModule(SmDataset):
    def init_dataset(self):
        dataset = get_gsm8k_questions()
        dataset = dataset.train_test_split(test_size=0.1)
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

    def setup(self, stage: Optional[str] = None):
        self.init_dataset()

    def reward_functions(self) -> list[RewardFunc]:
        return [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            math_correctness_reward_func,
        ]


def _twenty_q_map(example: dict) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": GUESSER_PROMPT},
        ],
        "answer": example["metadata"]["word"][0],
    }


def parse_groups(input_string) -> list[list[str]]:
    # Find all occurrences of text within <group>...</group>
    group_contents = re.findall(r"<group>(.*?)</group>", input_string, re.DOTALL)

    groups = []
    for content in group_contents:
        # Split on commas and trim each word
        words = [word.strip() for word in content.split(",") if word.strip()]
        groups.append(words)

    return groups


def score_connections(solution_groups, submitted_groups):
    hard_score = 0
    correct_group_indices = []  # Track indices of correctly solved solution groups.

    solution_sets = [set(group) for group in solution_groups]
    solved = set()

    for submitted in submitted_groups:
        submitted_set = set(submitted)
        for index, correct_set in enumerate(solution_sets):
            if submitted_set == correct_set and index not in solved:
                hard_score += 1
                correct_group_indices.append(index)
                solved.add(index)
                break
    return float(hard_score)


def score_connections_soft(solution_groups, submitted_groups):
    total_score = 0
    # Convert each solution group to a set for easier comparisons.
    solution_sets = [set(group) for group in solution_groups]

    # For each solution group, determine the best (largest) number of correct items found
    # among all submitted groups.
    for sol_set in solution_sets:
        best_match_count = 0
        for submitted in submitted_groups:
            submitted_set = set(submitted)
            match_count = len(sol_set.intersection(submitted_set))
            best_match_count = max(best_match_count, match_count)
        total_score += best_match_count * 0.25

    return total_score


def _generations(completions: list[dict]) -> list[str]:
    return [completion[0]["content"] for completion in completions]


def connections_soft_group_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Reward the num correct in each group."""
    model_generations = _generations(completions)
    groups = [parse_groups(r) for r in model_generations]
    scores = [score_connections_soft(kwargs["answer"], g) for g in groups]
    logger.info(f"Connections group soft scores: {scores}")
    scores = [s * 2 for s in scores]  # Scale the score to be between 0 and 5
    scores = [max(0.0, min(5.0, s)) for s in scores]  # Clamp to [0, 5]
    return scores


def connections_hard_group_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Reward whether each group is correct as a whole or not."""
    model_generations = _generations(completions)
    print("Generations:")
    for g in model_generations:
        print("-" * 80)
        print(g)
    generation_groups = [parse_groups(r) for r in model_generations]
    scores = [score_connections(kwargs["answer"], g) for g in generation_groups]
    scores = [s * 5 for s in scores]  # Scale the score to be between 0 and 5
    scores = [max(0.0, min(5.0, s)) for s in scores]  # Clamp to [0, 5]
    logger.info(f"Connections group hard scores: {scores}")
    return scores


def group_size_reward_func(prompts, completions, **kwargs) -> list[float]:
    model_generations = _generations(completions)
    groups = [parse_groups(r) for r in model_generations]
    sizes = [len(g) for g in groups]
    rewards = [0.5 if s == 4 else 0.0 for s in sizes]
    logger.info(f"Group size rewards: {rewards}")
    return rewards


def n_groups_reward_func(prompts, completions, **kwargs) -> list[float]:
    model_generations = _generations(completions)
    groups = [parse_groups(r) for r in model_generations]
    rew = [0.5 if len(g) == 4 else 0.0 for g in groups]
    logger.info(f"Number of groups rewards: {rew}")
    return rew


CONNECTIONS_PROMPT = """
You are an expert puzzle solving model.
Find groups of words that are related to each other. Each group is four words long. There are exactly four groups in total.
You may only use each word in one group.
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
<group>
...
</group>
<group>
...
</group>
</answer>

# Example

User: apple, orange, banana, pear, corolla, charger,
Assistant: <reasoning>
The first group are all fruits.
The second group are all cars.
</reasoning>
<answer>
<group>apple, orange, banana, pear</group>
<group>corolla, charger</group>
</answer>

# Example

User: dog, cat, red, white,
Assistant: <reasoning>
The first group are all animals.
The second group are all colors.
</reasoning>
<answer>
<group>dog, cat</group>
<group>red, white</group>
</answer>
"""


def _connections_map(example: dict) -> dict:
    words = example["words"]
    words_formatted = ", ".join(words)
    answer = []
    answer_groups = []
    for group in example["groups"]:
        answer.append(", ".join(group["words"]))
        answer_groups.append(group["words"])
    answer_formatted = "\n".join([f"<group>{a}</group>" for a in answer])
    return {
        "prompt": [
            {
                "role": "system",
                "content": CONNECTIONS_PROMPT,
            },
            {"role": "user", "content": words_formatted},
        ],
        "answer": f"<answer>{answer_formatted}</answer>",
        "answer_groups": answer_groups,
    }


class ConnectionsDataModule(SmDataset):
    def setup(self, stage: Optional[str] = None):
        prompts_pd = pd.read_json(
            "../dataset_files/connections_prompts.jsonl", lines=True
        )
        df_groups = pd.json_normalize(prompts_pd["solution"], "groups")  # type: ignore

        groups = [
            {
                "groups": (
                    g := df_groups.sample(4, replace=False).reset_index(drop=True)
                ).to_dict(orient="records"),
                "words": list(itertools.chain.from_iterable(g["words"].dropna())),
            }
            for _ in range(100000)
        ]

        groups_pd = pd.DataFrame(groups)
        groups_dataset = Dataset.from_pandas(groups_pd)
        groups_dataset = groups_dataset.map(_connections_map)
        groups_dataset = groups_dataset.train_test_split(test_size=0.1)  # type: ignore
        self.train_dataset = groups_dataset["train"]
        self.val_dataset = groups_dataset["test"]

    def reward_functions(self) -> list[RewardFunc]:
        return [
            xmlcount_reward_func,
            strict_format_reward_func,
            connections_soft_group_reward_func,
            group_size_reward_func,
            connections_hard_group_reward_func,
            n_groups_reward_func,
        ]


GSM8K_SYSTEM_PROMPT = """
You are an expert reasoning model.
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>

# Example

User: Mr. Sam shared a certain amount of money between his two sons, Ken and Tony. If Ken got $1750, and Tony got twice as much as Ken, how much was the money shared?
Assistant: <reasoning>
Tony got twice $1750 which is 2*$1750 = 3500
The total amount shared was $1750+$3500 = 5250
</reasoning>
<answer>5250</answer>

# Example

User: Bob has a collection of marbles. If his friend Tim has 20 marbles and Bob has three times as many marbles as Tim, how many marbles do they have together?
Assistant: <reasoning>Bob has 3x20 = 60 marbles.
Together, they have 20 + 60 = 80 marbles.</reasoning>
<answer>80</answer>
"""


def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]  # type: ignore
    data = data.map(  # type: ignore
        lambda x: {  # type: ignore
            "prompt": [
                {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )  # type: ignore
    return data  # type: ignore
