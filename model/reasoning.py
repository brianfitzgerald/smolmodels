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

    @classmethod
    def reward_functions(cls) -> list[RewardFunc]:
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
    submitted_sets = [set(group) for group in submitted_groups]

    # Get highest match count for each solution group
    best_match_counts = [
        max(len(sol_set.intersection(submitted)) for submitted in submitted_sets)
        for sol_set in solution_sets
    ]
    return float(sum(best_match_counts) / len(solution_groups)) / len(submitted_groups)


def _generations(completions: list[dict]) -> list[str]:
    return [completion[0]["content"] for completion in completions]


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
    prompts = [prompt[1]["content"] for prompt in prompts]
    answers = kwargs["answer"]
    logger.info("Generations:")
    for g, p, a in zip(model_generations, prompts, answers):
        logger.info("=" * 40)
        logger.info("Prompt: " + "-" * 20)
        logger.info(p)
        logger.info("Generation: " + "-" * 20)
        logger.info(g)
        logger.info("Answer: " + "-" * 20)
        logger.info(a)
    return [0.0] * len(completions)


def group_size_reward(prompts, completions, **kwargs) -> list[float]:
    model_generations = _generations(completions)
    groups = [parse_groups(r) for r in model_generations]
    sizes = [[len(s) for s in s] for s in groups]
    rewards = []
    for group_lens in sizes:
        sample_reward = 0.0
        for group_len in group_lens:
            if group_len == 4:
                sample_reward += 1.0
        if len(group_lens) > 0:
            rewards.append(sample_reward / len(group_lens))
        else:
            rewards.append(0.0)
    logger.info(f"Group size rewards: {rewards}")

    # scale to be between 0 and 0.25
    rewards = [r / 4 if r > 0 else 0.0 for r in rewards]
    return rewards


def n_groups_reward(prompts, completions, **kwargs) -> list[float]:
    model_generations = _generations(completions)
    groups = [parse_groups(r) for r in model_generations]
    rew = [0.25 if len(g) == 4 else 0.0 for g in groups]
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

User: bird, curry, james, jordan, book, cactus, hedgehog, skeleton, equal, even, fair, just, agency, company, enterprise, firm
Assistant: <reasoning>
Okay, here's how I'm thinking about this. I have 16 words and need to group them into four groups. Hmm, I'm seeing that “Bird,” “Curry,” “James,” and “Jordan” are all basketball player nicknames, so that's one group.
Now, the next group: “Book,” “Agency,” “Company,” “Enterprise,” and “Firm.” These all seem linked with business/agencies, but I need only four. Maybe “Agency,” “Company,” “Enterprise,” and “Firm”?
Now I just need to figure out the rest of the groupings!
Okay, let's break this down systematically. For Group 1, the basketball player nicknames: “Bird,” “Curry,” “James,” and “Jordan.” They all fit here.
For Group 2, it's clear that “Company,” “Enterprise,” “Firm,” and “Agency” are all synonyms for “business.”
Group 3 seems straightforward: “Equal,” “Even,” “Fair,” and “Just” all reflect synonyms for “fair.”
Now, Group 4: “Book,” “Cactus,” “Hedgehog,” and “Skeleton.” Hmm, these all connect through “spine” in some way—“book spine,” “cactus spine,” “hedgehog spine,” and “skeleton spine” makes sense!
Okay, let's go back to the core groupings. Group 1: “Bird,” “Curry,” “James,” “Jordan” — basketball players. Group 2: “Agency,” “Company,” “Enterprise,” “Firm” — business synonyms. Group 3: “Equal,” “Even,” “Fair,” “Just” — synonyms for “fair.” That leaves Group 4, which I'm thinking includes “Book,” “Cactus,” “Hedgehog,” and “Skeleton.” These can all be followed by “spine” when used in certain contexts (e.g., “book spine,” “cactus spine,” “hedgehog spine,” “skeleton spine”). This seems like a solid grouping!
</reasoning>
<answer>
<group>bird, curry, james, jordan</group>
<group>book, agency, company, enterprise, firm</group>
<group>equal, even, fair, just</group>
<group>cactus, hedgehog, skeleton</group>
</answer>

# Example

User: candle, crayon, honeycomb, seal, defense, excuse, out, reason, kettles, mittens, raindrops, whiskers, canine, fang, molar, tusk
Assistant: <reasoning>
Alright, I'm tasked with finding four groups of related words. Let's start by listing the words: candle, crayon, honeycomb, seal, defense, excuse, out, reason, kettles, mittens, raindrops, whiskers, canine, fang, molar, tusk.
I wonder if “teeth” could form a group — molar, fang, tusk, canine definitely seem related. Keep going, looking for other links.
I'll need to keep the groups to four words each. Let me map out these connections now.
I see “canine”, “fang”, “molar”, and “tusk” are related to teeth, forming one group. Now, I have “candle”, “crayon”, “honeycomb”, and “seal” left. I think these can be linked to “bee” — like “beecandle”, but that doesn't fully work.
Then there's “raindrops”, “whiskers”, “kettles”, and “mittens”, which seems to fit well with lyrics from “The Sound of Music”.
Finally, “defense”, “excuse”, “out”, and “reason” could fit with “no” preceding them, like “no defense”.
I've identified a few clear groups: For “teeth”, the group is “canine”, “fang”, “molar”, “tusk”. For “wax”, it pairs “candle”, “crayon”, “honeycomb”, “seal”. Then, there's the song-inspired group: “raindrops”, “whiskers”, “kettles”, “mittens”. That leaves “defense”, “excuse”, “out”, and “reason”. They all fit with the phrase “no”, like “no defense”, “no excuse”, “no reason”. I'm settling on these four groups, based on straightforward connections and some memorable associations. Great!
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
                    g := df_groups.sample(4, replace=False).reset_index(drop=True)
                ).to_dict(orient="records"),
                "words": list(itertools.chain.from_iterable(g["words"].dropna())),
            }
            for _ in range(100000)
        ]

        groups_pd = pd.DataFrame(groups)
        groups_dataset = Dataset.from_pandas(groups_pd)
        groups_dataset.shuffle(seed=42)
        groups_dataset = groups_dataset.map(_connections_map)
        groups_dataset = groups_dataset.train_test_split(test_size=0.1)  # type: ignore
        self.train_dataset = groups_dataset["train"]
        self.val_dataset = groups_dataset["test"]

    @classmethod
    def reward_functions(cls) -> list[RewardFunc]:
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
