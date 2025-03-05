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
from synthetic_data.generation import GenerationWrapper
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
    is_question_pattern = r"(is it .*?)\?$"

    # Check for question and final guess patterns
    question = extract_from_pattern(content, question_pattern)
    if question:
        return question, False

    guess = extract_from_pattern(content, guess_pattern)
    if guess:
        return guess, True

    is_question = re.search(is_question_pattern, content, re.DOTALL)
    if is_question:
        return content, False

    # If no explicit tags, check for "question:" prefix
    if "question:" in content.lower():
        question = content.lower().split("question:")[1].strip()
        return question, False

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
    output_dataset_format = DatasetFormat.PARQUET

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
        n_steps: int = 20,
    ):
        nltk.download("punkt_tab")
        nltk.download("averaged_perceptron_tagger_eng")
        self.generator = generator
        self.word_list = get_default_word_list()
        self.n_steps = n_steps
        self.current_role: TwentyQuestionsRole = "guesser"

        self.random = random.Random(None)
        self.step_count = 0
        self.curr_word: Optional[WordVariants] = None
        self.conversation: Conversation = []
        self.task = TwentyQuestionsTask()

    async def step(self):
        assert self.curr_word is not None, "call env.reset() first."
        query_conv: Conversation = (
            _conv_template_guesser(self.conversation, self.n_steps - self.step_count)
            if self.current_role == "guesser"
            else _conv_template_oracle(self.curr_word, self.conversation)
        )
        response = await self.generator.generate([query_conv])
        response_to_add = response[0]
        if self.current_role == "oracle":
            formatted_output = parse_oracle_output(response_to_add)
            formatted_output = f"Oracle: {formatted_output}"
            if len(formatted_output) == 0:
                raise ValueError(
                    "No output could be extracted from the oracle's response"
                )
        elif self.current_role == "guesser":
            logger.info(f"Raw response: {response_to_add}")
            formatted_output, is_final_guess = parse_guesser_output(response_to_add)
            if is_final_guess:
                if did_win(self.curr_word, response_to_add):
                    reward = 1.0
                else:
                    reward = 0.0
                return reward, True
            else:
                self.step_count += 1
            if len(formatted_output) == 0:
                raise ValueError(
                    "No output could be extracted from the guesser's response"
                )
            formatted_output = f"Guesser: {formatted_output}"
        else:
            raise ValueError(f"Unknown role: {self.current_role}")

        logger.info(f"Added output: {formatted_output}")
        self.conversation.append({"role": "assistant", "content": response_to_add})

        if did_win(self.curr_word, response_to_add):
            return 1.0, True

        self.current_role = "oracle" if self.current_role == "guesser" else "guesser"

        print(self.step_count, self.n_steps)
        if self.step_count == self.n_steps:
            logger.info("The word was {}", self.curr_word[0])

        # Return a reward of 0.0 and done flag based on conversation length
        done = self.step_count >= self.n_steps
        return 0.0, done

    def reset(self, seed: Optional[int] = None):
        self.step_count = 0
        self.conversation = []
        if seed is not None:
            self.random = random.Random(seed)

        if seed is not None:
            word_ind = seed % len(self.word_list)
            self.curr_word = self.word_list[word_ind]
        else:
            self.curr_word = self.random.choice(self.word_list)

        logger.info(f"Word to guess: {self.curr_word[0]}")
