from __future__ import annotations

import nltk.downloader
from synthetic_data.tasks import BaseTask
from typing import Optional
import random
import nltk
import re
from typing import Literal
from loguru import logger

from gyms.twenty_questions.data import WordVariants, get_default_word_list
from gyms.utils import TextEnv
from synthetic_data.generation import (
    GenWrapperArgs,
    GenerationWrapper,
    get_generation_wrapper,
)
from synthetic_data.utils import Conversation, DatasetFormat


GUESSER_PROMPT = """
You are an AI agent playing the role of the guesser in a game of Twenty Questions. Your goal is to guess the secret object, person, or concept that the other player (the answerer) is thinking of by asking up to 20 yes-or-no questions.

You have {remaining_questions} questions left to ask.

Based on the information you've gathered so far, think carefully about what question would be most helpful in narrowing down the possible answers. Your question should be specific and designed to eliminate as many possibilities as possible.

If you believe you know the answer before using all 20 questions, you may make a guess instead of asking another question. However, be sure you have enough information to make an educated guess.

When formulating your next question or making a guess, follow these steps:

1. Analyze the previous questions and answers to identify patterns or key information.
2. Consider what categories or characteristics you still need to clarify.
3. Formulate a clear, concise yes-or-no question that will provide valuable information.

Present your output in the following format:

<thought>
[Your reasoning process for choosing the next question or making a guess]
</thought>

<output>
[If asking a question]: Question: [Your yes-or-no question]
[If making a guess]: Final Guess: [Your guess of the secret object, person, or concept]
</output>

Remember, your goal is to guess the correct answer in as few questions as possible.
"""

ORACLE_PROMPT = """
You are the oracle in a game of 20 Questions. Your role is to think of a secret answer and respond to the player's guesses. The secret answer has already been chosen for you.
If the player asks a question, you should respond with a yes-or-no answer.

The secret answer is:
<secret_answer>
{secret_answer}
</secret_answer>

When the player makes a guess, you should compare it to the secret answer. If the guess matches the secret answer exactly (ignoring case), the player has won. If the guess is incorrect, you should not provide any information about how close or far off the guess is.

Respond to the player's guess in the following format:
1. If the guess is correct, respond with: "Congratulations! You've guessed correctly. The answer was indeed [secret answer]."
2. If the player asked a question, respond with: "Yes" or "No"

Please provide your response to the player's guess inside <response> tags.
"""


def _conv_template_guesser(conversation: Conversation, remaining_questions: int):
    return [
        {
            "role": "system",
            "content": GUESSER_PROMPT.format(remaining_questions=remaining_questions),
        },
        *conversation,
    ]


def _conv_template_oracle(word_var: WordVariants, conversation: Conversation):
    return [
        {"role": "system", "content": ORACLE_PROMPT.format(secret_answer=word_var)},
        *conversation,
    ]


TwentyQuestionsRole = Literal["guesser", "oracle"]


class OutputParseError(Exception):
    """Custom exception for when parsing fails."""

    pass


def parse_oracle_output(message: str) -> str:
    response_match = re.search(r"<response>(.*?)</response>", message, re.DOTALL)
    if not response_match:
        return ""

    response = response_match.group(1).strip()

    if response.lower() == "yes":
        return "Yes"
    elif response.lower() == "no":
        return "No"

    return response


def extract_from_pattern(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_guesser_output(message: str) -> tuple[str, bool]:
    content = message.replace("Guesser:", "").strip()

    # First, try to extract content within <output> tags
    output_match = re.search(r"<output>(.*?)</output>", content, re.DOTALL)
    if output_match:
        content = output_match.group(1).strip()

    # Define patterns for question and final guess
    question_pattern = r"Question:\s*(.*?)(?:\n|$)"
    guess_pattern = r"Final Guess:\s*(.*?)(?:\n|$)"
    final_answer_pattern = r"Final Answer:\s*(.*?)(?:\n|$)"
    is_question_pattern = r"^is it .*?\?$"

    # Check for question and final guess patterns
    question = extract_from_pattern(content, question_pattern)
    if question:
        return question, False

    guess = extract_from_pattern(content, guess_pattern)
    if guess:
        return guess, True

    final_answer = extract_from_pattern(content, final_answer_pattern)
    if final_answer:
        return final_answer.strip(), True

    # Check for "is it" questions
    if re.search(is_question_pattern, content.lower()):
        # Preserve the original case for the question
        return content, False

    # If no explicit tags, check for "question:" prefix
    if "question:" in content.lower():
        # Preserve the original case for the question
        question_part = content.split(":", 1)[1].strip() if ":" in content else content
        return question_part, False

    return content, False


def did_win(word_var: WordVariants, question: str):
    for word in word_var.words:
        if word.lower() in question.lower():
            return True
    return False


class TwentyQuestionsTask(BaseTask):
    seed_data_format = DatasetFormat.NONE
    output_dataset_name = "twenty_questions"
    output_dataset_org = "roborovski"
    output_dataset_format = DatasetFormat.HF_DATASET

    dataset_columns = ["conversation"]


class TwentyQuestionsPolicyEnvironment(TextEnv):
    """
    Environment for generating synthetic preference data for the 20 questions game.
    On each step:
    1. The generator generates a query about the answer.
    2. The oracle answers the question.
    """

    def __init__(
        self,
        generator: GenerationWrapper,
        seed: int,
        n_steps: int = 20,
    ):
        nltk.download("punkt_tab")
        nltk.download("averaged_perceptron_tagger_eng")
        self.guesser_model = generator
        self.word_list = get_default_word_list()
        self.n_steps = n_steps
        self.current_role: TwentyQuestionsRole = "guesser"
        self.seed = seed
        self.oracle_model = get_generation_wrapper(
            "gpt-4o-mini",
            args_override=GenWrapperArgs(stop=["</output>"], seed=self.seed),
        )

        self.random = random.Random(self.seed)
        self.step_count = 0
        self.curr_word: Optional[WordVariants] = None
        self.conversation: Conversation = []
        self.task = TwentyQuestionsTask(run_mode="modal")
        self.run_metadata: dict = {}

    async def step(self) -> bool:
        assert self.curr_word is not None, "call env.reset() first."
        query_conv: Conversation = (
            _conv_template_guesser(self.conversation, self.n_steps - self.step_count)
            if self.current_role == "guesser"
            else _conv_template_oracle(self.curr_word, self.conversation)
        )
        if self.current_role == "oracle":
            response = await self.oracle_model.generate([query_conv])
        else:
            response = await self.guesser_model.generate([query_conv])
        response_to_add = response[0]
        logger.info(f"Raw response: {response_to_add}")
        try:
            if self.current_role == "oracle":
                formatted_output = parse_oracle_output(response_to_add)
                formatted_output = f"Oracle: {formatted_output}"
                if len(formatted_output) == 0:
                    raise OutputParseError(
                        "No output could be extracted from the oracle's response"
                    )
            elif self.current_role == "guesser":
                formatted_output, is_final_guess = parse_guesser_output(response_to_add)
                if is_final_guess:
                    if did_win(self.curr_word, response_to_add):
                        self.run_metadata["win"] = True
                    return True
                else:
                    self.step_count += 1
                if len(formatted_output) == 0:
                    raise OutputParseError(
                        "No output could be extracted from the guesser's response"
                    )
                formatted_output = f"Guesser: {formatted_output}"
            else:
                raise ValueError(f"Unknown role: {self.current_role}")
        except OutputParseError as e:
            logger.error(f"Error parsing output: {e}")
            self.run_metadata["succeeded"] = False
            return True

        logger.info(
            f"{self.current_role} output for {self.step_count}/{self.n_steps}: {formatted_output}"
        )
        self.conversation.append({"role": "assistant", "content": response_to_add})

        if did_win(self.curr_word, response_to_add):
            self.run_metadata["win"] = True
            return True

        self.current_role = "oracle" if self.current_role == "guesser" else "guesser"

        if self.step_count == self.n_steps:
            logger.info("The word was {}", self.curr_word[0])

        # Return a reward of 0.0 and done flag based on conversation length
        return self.step_count >= self.n_steps

    def reset(self):
        self.step_count = 0
        self.conversation = []
        self.run_metadata: dict = {}

        word_ind = self.seed % len(self.word_list)
        self.curr_word = self.word_list[word_ind]

        logger.info(f"Word to guess: {self.curr_word[0]} seed: {self.seed}")

        self.run_metadata["word"] = self.curr_word.json()
        self.run_metadata["n_steps"] = self.n_steps
        self.run_metadata["seed"] = self.seed
        self.run_metadata["win"] = False
        self.run_metadata["succeeded"] = True
