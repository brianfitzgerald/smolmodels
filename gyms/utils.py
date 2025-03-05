from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from synthetic_data.generation import GenerationWrapper
from synthetic_data.tasks import BaseTask
from synthetic_data.utils import Conversation


class TextEnv(ABC):
    task: BaseTask
    conversation: Conversation
    run_metadata: dict

    def __init__(self, generation_wrapper: GenerationWrapper):
        self.generation_wrapper = generation_wrapper

    @abstractmethod
    async def step(self) -> bool:
        """
        Perform a single step in the environment.
        Returns a boolean indicating if the episode is done.
        """
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None):
        """
        Reset the environment to the initial state.
        """
        pass
