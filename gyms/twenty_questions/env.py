from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import random
import nltk

from gyms.twenty_questions.data import WordVariants
from gyms.utils import Text, TextEnv, TextHistory, TextTrajectory
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


GUESSER_PROMPT = "Welcome to the game of Twenty Questions! Your objective is to guess what the object is within twenty questions. At every turn, you will have the oppurnity to ask a yes/no question, and receive an answer from the oracle. You can ask tweny questions but must ask as few questions as possible. "


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
        word_list: List[WordVariants],
        max_conversation_length: int = 20,
    ):
        self.generator = generator
        self.word_list = word_list
        self.max_conversation_length = max_conversation_length

        self.random = random.Random(None)
        self.count = 0
        self.curr_word: Optional[WordVariants] = None

    async def step(self, conversation: Conversation) -> tuple[float, bool]:
        assert self.curr_word is not None, "call env.reset() first."
        self.count += 1
        query_conv: Conversation = [
            {"role": "system", "content": "Questions:"},
        ]
        answer = await self.generator.generate([query_conv])
        answer_text = Text(answer[0], is_action=False)

        if self.count == self.max_conversation_length:
            print("The word was", self.curr_word[0])
        return

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.count = 0
        if self.curr_word is not None:
            print("The word was ", self.curr_word)
            print("Next word...")
        if seed is not None:
            self.random = random.Random(seed)

        if options is None:
            options = {}
        if seed is not None:
            word_ind = seed % len(self.word_list)
            self.curr_word = self.word_list[word_ind]
        else:
            self.curr_word = self.random.choice(self.word_list)
