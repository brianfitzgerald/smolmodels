from typing import Optional
from datasets import load_dataset, Dataset, DatasetDict
from transformers.trainer_callback import TrainerCallback
import re
from loguru import logger
import pandas as pd
from gyms.twenty_questions.env import GUESSER_PROMPT
from trl_wrapper.wrapper_config import SmDataset
from trl.trainer.grpo_trainer import RewardFunc

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


def connections_reward_func(prompts, completions, **kwargs) -> list[float]:
    model_generations = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in model_generations]
    return [0.0]


class TwentyQDataModule(SmDataset):
    def init_dataset(self):
        dataset: Dataset = load_dataset("roborovski/twenty_questions")["train"]  # type: ignore
        dataset = dataset.map(_twenty_q_map)  # type: ignore
        dataset: DatasetDict = dataset.train_test_split(test_size=0.1)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

    def setup(self, stage: Optional[str] = None):
        self.init_dataset()

    def reward_functions(self) -> list[RewardFunc]:
        return [
            xmlcount_reward_func,
            strict_format_reward_func,
            int_reward_func,
            connections_reward_func,
        ]


CONNECTIONS_PROMPT = """
You are an expert puzzle solving model.
Find groups of words that are related to each other, and return the answer in the following format:
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
    for group in example["solution"]["groups"]:
        answer.append(", ".join(group["words"]))
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
    }


class ConnectionsDataModule(SmDataset):
    def init_dataset(self):
        dataset = Dataset.from_pandas(
            pd.read_json("../dataset_files/connections_prompts.jsonl", lines=True)
        )
        dataset = dataset.map(_connections_map)  # type: ignore
        dataset = dataset.train_test_split(test_size=0.1)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

    def setup(self, stage: Optional[str] = None):
        self.init_dataset()

    def reward_functions(self) -> list[RewardFunc]:
        return [
            xmlcount_reward_func,
            strict_format_reward_func,
            int_reward_func,
        ]


SYSTEM_PROMPT = """
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
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )  # type: ignore
    return data  # type: ignore
