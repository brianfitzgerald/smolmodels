from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List
from copy import deepcopy
from dataclasses import dataclass
import random

from gyms.twenty_questions.data import WordVariants
from synthetic_data.generation import GenerationWrapper


@dataclass(frozen=True)
class Text:
    text: str
    is_action: bool


TextHistory = Tuple[Text, ...]


class TextEnv(ABC):
    @abstractmethod
    def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        pass

    @abstractmethod
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> TextHistory:
        pass

    def close(self) -> None:
        pass

    def copy(self) -> TextEnv:
        return deepcopy(self)


INITIAL_STR = "Questions:\n"


class TwentyQuestionsPolicyEnvironment(TextEnv):
    def __init__(
        self,
        oracle: GenerationWrapper,
        word_list: List[WordVariants],
        max_conversation_length: int = 20,
    ):
        self.oracle = oracle
        self.word_list = word_list
        self.max_conversation_length = max_conversation_length

        self.random = random.Random(None)
        self.count = 0
        self.curr_word: Optional[WordVariants] = None

    def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        assert text_history[-1].is_action
        assert self.curr_word is not None, "call env.reset() first."
        self.count += 1
        question = text_history[-1].text.strip()
        answer = self.oracle.generate_answers(self.curr_word, question)
        answer_text = Text(answer + "\n", is_action=False)

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
