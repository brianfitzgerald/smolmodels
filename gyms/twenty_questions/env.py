from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import random
import nltk

from gyms.twenty_questions.data import WordVariants
from gyms.utils import INITIAL_STR, Text, TextEnv, TextHistory, TextTrajectory
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


def create_trajectory_from_history(
    word_var: WordVariants,
    text_history: TextHistory,
    max_conversation_len: int = 20,
) -> TextTrajectory:
    """Create a TextTrajectory from a TextHistory"""
    assert len(text_history) % 2 == 1, (
        "TextHistory should be [initial str, question1, answer1, ..., questionN, answerN]."
    )
    assert all(question_text.is_action for question_text in text_history[1::2]), (
        "All questions should be actions."
    )
    assert all(not answer_text.is_action for answer_text in text_history[0::2]), (
        "All answers should not be actions."
    )
    # subtract 1 because of the starting text, then text_history contains pairs of questions and answers
    conversation_len = (len(text_history) - 1) // 2
    assert conversation_len <= max_conversation_len, (
        f"Conversation is too long {conversation_len}. Max should be {max_conversation_len}."
    )

    reward: List[float] = []
    for text in text_history:
        if text.is_action:
            reward.append(-1.0)
        else:
            reward.append(0.0)

    if len(text_history) < 2:
        done = False
    else:
        last_question = text_history[-2].text.strip()
        last_answer = text_history[-1].text.strip()
        word_guessed = last_answer == "Yes." and is_done(word_var, last_question)

        done = word_guessed or conversation_len == max_conversation_len

        if word_guessed:
            reward[-2] = 0.0

    return TextTrajectory(text_history, tuple(reward), done)


class TwentyQuestionsPolicyEnvironment(TextEnv):
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

    async def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        assert text_history[-1].is_action
        assert self.curr_word is not None, "call env.reset() first."
        self.count += 1
        question = text_history[-1].text.strip()
        query_conv: Conversation = [
            {"role": "system", "content": "Questions:"},
        ]
        answer = await self.generator.generate([query_conv])
        answer_text = Text(answer[0], is_action=False)

        trajectory = create_trajectory_from_history(
            self.curr_word, text_history + (answer_text,), self.max_conversation_length
        )
        if self.count == self.max_conversation_length:
            print("The word was", self.curr_word[0])
        return trajectory.text_history, trajectory.reward[-2], trajectory.done

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> TextHistory:
        self.count = 0
        if self.curr_word is not None:
            print("The word was ", self.curr_word)
            print("Next word...")
        if seed is not None:
            self.random = random.Random(seed)

        if options is None:
            options = {}
        deterministic = options.get("deterministic", False)

        if deterministic:
            assert seed is not None, (
                "In deterministic mode, the seed specifies which word to use."
            )
            word_ind = seed % len(self.word_list)
            self.curr_word = self.word_list[word_ind]
        else:
            self.curr_word = self.random.choice(self.word_list)

        return (Text(INITIAL_STR, is_action=False),)
