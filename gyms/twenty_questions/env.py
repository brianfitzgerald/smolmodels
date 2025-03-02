from __future__ import annotations
from pprint import pprint
from typing import Dict, Optional
import random
import nltk
import re
from typing import Literal
from loguru import logger

from gyms.twenty_questions.data import WordVariants, get_default_word_list
from gyms.utils import TextEnv
from synthetic_data.generation import GenerationWrapper
from synthetic_data.utils import Conversation


def is_done(word_var: WordVariants, question: str):
    # cut out punctuations at the end
    while len(question) > 0 and not question[-1].isalpha():
        question = question[:-1]

    if len(question) == 0:
        return False

    question_pos = nltk.pos_tag(nltk.word_tokenize(question.lower()))

    # ignore these nouns when checking for extra words
    ignores = {"object", "something", "type", "kind"}
    for pos_list in word_var.pos_tags:
        for w, _ in pos_list:
            ignores.add(w)

    # check for extra words
    for q_i in range(len(question_pos)):
        q_i_word, q_i_pos = question_pos[q_i]
        # check if the current word is a noun that shouldn't be ignored
        if q_i_pos[:2] == "NN" and q_i_word not in ignores:
            # if it's a counter word that comes before "of", also ignore it
            if q_i + 1 < len(question_pos) and question_pos[q_i + 1][0] == "of":
                continue
            # extra word found
            return False

    # check for the actual word at the end of the question
    for word_pos in word_var.pos_tags:
        if len(word_pos) > len(question_pos):
            continue

        all_same = True
        for (var_i_word, _), (q_i_word, _) in zip(
            word_pos, question_pos[-len(word_pos) :]
        ):
            if var_i_word != q_i_word:
                all_same = False
                break
        if all_same:
            return True

    return False


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
You are the oracle in a game of 20 Questions. Your role is to think of a secret answer and respond to the player's guesses. The secret answer has already been chosen for you. Your task is to determine if the player has guessed correctly and respond appropriately.

The secret answer is:
<secret_answer>
{secret_answer}
</secret_answer>

When the player makes a guess, you should compare it to the secret answer. If the guess matches the secret answer exactly (ignoring case), the player has won. If the guess is incorrect, you should not provide any information about how close or far off the guess is.

Respond to the player's guess in the following format:
1. If the guess is correct, respond with: "Congratulations! You've guessed correctly. The answer was indeed [secret answer]."
2. If the guess is incorrect, respond with: "I'm sorry, that's not correct. Would you like to guess again?"

Do not provide any additional hints or information about the secret answer. Your response should be concise and limited to the formats provided above.

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
        max_conversation_length: int = 20,
    ):
        self.generator = generator
        self.word_list = get_default_word_list()
        self.max_conversation_length = max_conversation_length
        self.current_role: TwentyQuestionsRole = "guesser"
        nltk.download("punkt_tab")
        nltk.download("averaged_perceptron_tagger_eng")

        self.random = random.Random(None)
        self.step_count = 0
        self.curr_word: Optional[WordVariants] = None
        self.conversation: Conversation = []

    async def step(self):
        assert self.curr_word is not None, "call env.reset() first."
        self.step_count += 1
        query_conv: Conversation = (
            _conv_template_guesser(
                self.conversation, self.max_conversation_length - self.step_count
            )
            if self.current_role == "guesser"
            else _conv_template_oracle(self.curr_word, self.conversation)
        )
        logger.info(pprint(query_conv))
        response = await self.generator.generate([query_conv])
        logger.info(pprint(response))
        self.conversation.append({"role": "assistant", "content": response[0]})

        self.current_role = "oracle" if self.current_role == "guesser" else "guesser"

        if self.step_count == self.max_conversation_length:
            logger.info("The word was {}", self.curr_word[0])

        # Return a reward of 0.0 and done flag based on conversation length
        done = self.step_count >= self.max_conversation_length
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


def parse_oracle_output(message: str) -> str:
    response_match = re.search(r"<response>(.*?)</response>", message, re.DOTALL)
    if not response_match:
        return ""

    return response_match.group(1).strip()


def parse_guesser_output(message: str) -> tuple[str, bool]:
    output_match = re.search(r"<output>(.*?)</output>", message, re.DOTALL)
    if not output_match:
        return "", False

    output_content = output_match.group(1).strip()

    question_match = re.search(r"Question:\s*(.*?)(?:\n|$)", output_content, re.DOTALL)
    if question_match:
        return question_match.group(1).strip(), False

    guess_match = re.search(r"Final Guess:\s*(.*?)(?:\n|$)", output_content, re.DOTALL)
    if guess_match:
        return guess_match.group(1).strip(), True

    return "", False
