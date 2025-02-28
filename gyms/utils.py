from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List, Union
from copy import deepcopy
from dataclasses import dataclass
import random
import nltk

from synthetic_data.utils import Conversation


@dataclass(frozen=True)
class Text:
    text: str
    is_action: bool


TextHistory = Tuple[Text, ...]


@dataclass(frozen=True)
class TextTrajectory:
    text_history: TextHistory
    reward: Tuple[float, ...]
    done: bool

    def __post_init__(self):
        assert len(self.reward) == len(self.text_history), (
            "reward is needed for each text"
        )
        assert all(
            [
                r == 0.0
                for r, t in zip(self.reward, self.text_history)
                if not t.is_action
            ]
        ), "reward for non-actions texts should be 0.0"


class TextEnv(ABC):
    @abstractmethod
    async def step(self, conversation: Conversation) -> Tuple[float, bool]:
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        pass

    def close(self) -> None:
        pass

    def copy(self) -> TextEnv:
        return deepcopy(self)
