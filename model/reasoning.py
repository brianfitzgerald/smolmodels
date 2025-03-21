from torch.nn.utils.rnn import pad_sequence
import re
from loguru import logger
import torch
from transformers import (
    PreTrainedTokenizerBase,
)
from datasets import load_dataset

# https://github.com/aburkov/theLMbook/blob/main/GRPO_Qwen_0_5_Instruct.ipynb


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


VERBOSE = False


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    if VERBOSE:
        logger.info(
            "-" * 20,
            f"Question:\n{q}",
            f"\nAnswer:\n{answer[0]}",
            f"\nResponse:\n{responses[0]}",
            f"\nExtracted:\n{extracted_responses[0]}",
        )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


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


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def _extract_last_number(text):
    """
    Extracts the last number from text if it's properly separated.
    The number must be at the end and separated by space or = sign.
    Ignores $ and % signs.
    Returns None if no valid number is found.
    """
    import re

    # Remove $ and % signs
    text = text.replace("$", "").replace("%", "")

    # Look for numbers that are:
    # - preceded by space or = or start of string (via \b or ^)
    # - followed by end of string or space
    pattern = r"(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$"
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def extract_answer_from_model_output(text):
    """
    Extracts the value from the last <answer> tag in the text.
    Returns None if no valid answer is found.
    """
    # Split on <answer> and take everything after the last occurrence
    parts = text.split("<answer>")
    if len(parts) < 2:  # No <answer> tag found
        return None

    last_part = parts[-1]

    # Extract content up to </answer>
    if "</answer>" not in last_part:
        return None

    answer = last_part.split("</answer>")[0].strip()
    return None if answer == "..." else answer


def _extract_single_number(text):
    """
    Extracts a single number from text if exactly one exists,
    otherwise returns None.
    """
    import re

    numbers = re.findall(r"-?\d*\.?\d+", text)
    return float(numbers[0]) if len(numbers) == 1 else None


def correctness_reward(prompts, completions, answer, **kwargs):
    """
    Assigns a reward based on the correctness of the model's answer.
    Also logs detailed metrics about the response.
    """

    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_answer_from_model_output(r) for r in responses]

    rewards = []
    for r, a in zip(extracted, answer):
        if r == a:  # Exact match case
            rewards.append(2.0)
        else:
            # Try numeric equivalence
            r_num = _extract_single_number(str(r))
            a_num = _extract_single_number(str(a))
            if r_num is not None and a_num is not None and r_num == a_num:
                rewards.append(1.5)
            else:
                rewards.append(0.0)

    # Log completion lengths
    completion_lengths = [len(response.split()) for response in responses]
    logger.info(f"Completion lengths: {completion_lengths}")
    return rewards


def format_reward(completions, **kwargs):
    """
    Assigns a reward for adhering to the desired XML format.
    Also logs detailed format compliance metrics.
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    format_scores = []

    for response in responses:
        score = 0.0
        if "<reasoning>" in response:
            score += 0.20
        if "</reasoning>" in response:
            score += 0.20
        if "<answer>" in response:
            score += 0.20
        if "</answer>" in response:
            score += 0.20
        rewards.append(score)
        format_scores.append(score)

    return rewards


def prepare_dataset(split="train"):
    """Load and prepare the GSM8K dataset for training."""
    data = load_dataset("openai/gsm8k", "main")[split]
    formatted_data = []

    for example in data:
        formatted_example = {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "answer": extract_answer_from_dataset(example["answer"]),
        }
        formatted_data.append(formatted_example)

    return formatted_data


def build_prompt(messages):
    """
    Build a single prompt string from a list of messages.
    Each message is expected to be a dictionary with 'role' and 'content' keys.
    This function concatenates all message contents, preserving the training format.
    """
    return "\n".join([msg["content"].strip() for msg in messages])


class ChatDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        inputs = []
        labels = []
        for example in batch:
            # Here we assume the last message is the target (assistant's output)
            prompt = build_prompt(example["messages"][:-1])
            target = example["messages"][-1]["content"]

            # Concatenate prompt and target (add a newline between them)
            full_text = prompt + "\n" + target
            tokenized = self.tokenizer(
                full_text, truncation=True, max_length=self.max_length
            )
            input_ids = torch.tensor(tokenized["input_ids"])
            inputs.append(input_ids)
            # You can choose to set labels equal to input_ids, or modify as needed.
            labels.append(input_ids)

        inputs_padded = pad_sequence(
            inputs,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,  # type: ignore
        )
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": inputs_padded, "labels": labels_padded}
