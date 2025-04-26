from typing import Optional
from datasets import load_dataset, Dataset
from transformers.trainer_callback import TrainerCallback
import re
from loguru import logger
import pandas as pd
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
    if text.count("<reasoning>") == 1:
        count += 0.125
    if text.count("</reasoning>") == 1:
        count += 0.125
    if text.count("<answer>") == 1:
        count += 0.125
        count -= len(text.split("<answer>")[-1]) * 0.001
    if text.count("</answer>") == 1:
        count += 0.125
        count -= (len(text.split("<answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(prompts, completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    rewards = [count_xml(c) for c in contents]
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
    pattern = r"^<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    rewards = [0.25 if match else 0.0 for match in matches]
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


def parse_groups(input_string) -> list[list[str]]:
    # Find all occurrences of text within <group>...</group>
    group_contents = re.findall(r"<group>(.*?)</group>", input_string, re.DOTALL)

    groups = []
    for content in group_contents:
        # Split on commas and trim each word
        words = [word.strip() for word in content.split(",") if word.strip()]
        groups.append(words)

    return groups


def score_connections_hard(
    solution_groups: list[list[str]], submitted_groups: list[list[str]]
):
    """Return the number of correct groups."""
    hard_score = 0
    correct_group_indices = []  # Track indices of correctly solved solution groups.

    solution_set = [set(group) for group in solution_groups]
    solved = set()

    if len(submitted_groups) > N_GROUPS:
        return 0.0

    for submitted_group in submitted_groups:
        for i, correct_set in enumerate(solution_set):
            if set(submitted_group) == correct_set and i not in solved:
                hard_score += 1
                correct_group_indices.append(i)
                solved.add(i)
                break

    if len(submitted_groups) == 0:
        return 0.0
    return float(hard_score) / len(submitted_groups)


def score_connections_soft(
    solution_groups: list[list[str]], submitted_groups: list[list[str]]
):
    """Return the best match count for each solution group."""
    solution_sets = [set(group) for group in solution_groups]
    submitted_sets = [
        set(group) for group in submitted_groups if len(group) == GROUP_SIZE
    ]

    if len(submitted_sets) > N_GROUPS:
        return 0.0

    if len(submitted_groups) == 0 or len(solution_groups) == 0:
        return 0.0

    # Get highest match count for each solution group
    best_match_counts = []
    if submitted_sets:
        for sol_set in solution_sets:
            if submitted_sets:
                best_match_counts.append(
                    max(
                        len(sol_set.intersection(submitted))
                        for submitted in submitted_sets
                    )
                )
            else:
                best_match_counts.append(0)
    else:
        best_match_counts = [0] * len(solution_sets)
    return float(sum(best_match_counts) / len(solution_groups)) / len(submitted_groups)


def _generations(completions: list[dict]) -> list[str]:
    return [completion[0]["content"] for completion in completions]


def _user_messages(prompts: list[dict]) -> list[str]:
    idx = 1 if prompts[0][0]["role"] == "system" else 0
    return [p[idx]["content"] for p in prompts]


def soft_group_reward(prompts, completions, **kwargs) -> list[float]:
    """Reward the number of correct groups."""
    model_generations = _generations(completions)
    generation_groups = [parse_groups(r) for r in model_generations]
    scores = [
        score_connections_soft(answers, groups)
        for answers, groups in zip(kwargs["answer_groups"], generation_groups)
    ]
    logger.info(f"Soft accuracy scores: {scores}")
    return scores


def hard_group_reward(prompts, completions, **kwargs) -> list[float]:
    """Reward whether each group is correct as a whole or not."""
    model_generations = _generations(completions)
    generation_groups = [parse_groups(r) for r in model_generations]
    scores = [
        score_connections_hard(answers, groups)
        for answers, groups in zip(kwargs["answer_groups"], generation_groups)
    ]
    logger.info(f"Hard accuracy scores: {scores}")
    return scores


def logger_reward(prompts, completions, **kwargs) -> list[float]:
    model_generations = _generations(completions)
    prompts = _user_messages(prompts)
    for i, (g, p) in enumerate(zip(model_generations, prompts)):
        logger.info("=" * 40)
        logger.info(f"Prompt {i}: " + "-" * 20)
        logger.info(p)
        logger.info("Generation: " + "-" * 20)
        logger.info(g)
        if "answer" in kwargs:
            logger.info("Answer: " + "-" * 20)
            logger.info(kwargs["answer"][i])
    return [0.0] * len(completions)


N_GROUPS = 4
GROUP_SIZE = 4


def group_size_reward(prompts, completions, **kwargs) -> list[float]:
    model_generations = _generations(completions)
    groups = [parse_groups(r) for r in model_generations]
    sizes = [[len(s) for s in s] for s in groups]
    rewards = []
    for group_lens in sizes:
        sample_reward = 0.0
        for group_len in group_lens:
            if group_len == N_GROUPS:
                sample_reward += 1.0
        if len(group_lens) > 0:
            rewards.append(sample_reward / len(group_lens))
        else:
            rewards.append(0.0)
    logger.info(f"Group size rewards: {rewards}")

    rewards = [max(r, 0.0) for r in rewards]
    return rewards


def n_groups_reward(prompts, completions, **kwargs) -> list[float]:
    model_generations = _generations(completions)
    groups = [parse_groups(r) for r in model_generations]
    rewards = [1 if len(g) == N_GROUPS else 0.0 for g in groups]
    logger.info(f"Number of groups rewards: {rewards}")
    return rewards


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

User: candle, crayon, honeycomb, seal, defense, excuse, out, reason, kettles, mittens, raindrops, whiskers, canine, fang, molar, tusk
Assistant: <reasoning>
I'll start with breaking down the reasoning. For Group 1, the words related to wax: candle, crayon, honeycomb, seal — connecting to wax in various forms, such as crayon and candles being wax-based.
For Group 2, “My Favorite Things” connects to lyrics mentioning kettles, mittens, raindrops, whiskers.
Group 3 relates to teeth, covering canine, fang, molar, tusk. Group 4 involves “no” related phrases like “no excuse,” “no defense.”
</reasoning>
<answer>
<group> candle, crayon, honeycomb, seal</group>
<group> kettles, mittens, raindrops, whiskers</group>
<group> canine, fang, molar, tusk</group>
<group> defense, excuse, out, reason</group>
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
                    g := df_groups.sample(N_GROUPS, replace=False).reset_index(
                        drop=True
                    )
                ).to_dict(orient="records"),
                "words": list(itertools.chain.from_iterable(g["words"].dropna())),
            }
            for _ in range(50000)
        ]

        groups_pd = pd.DataFrame(groups)
        groups_dataset = Dataset.from_pandas(groups_pd)
        groups_dataset.shuffle(seed=42)
        groups_dataset = groups_dataset.map(_connections_map)
        groups_dataset = groups_dataset.train_test_split(test_size=0.1)  # type: ignore
        self.train_dataset = groups_dataset["train"]
        self.val_dataset = groups_dataset["test"]

    def reward_functions(self) -> list[RewardFunc]:
        return [
            xmlcount_reward_func,
            strict_format_reward_func,
            soft_group_reward,
            group_size_reward,
            hard_group_reward,
            n_groups_reward,
            logger_reward,
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
