from __future__ import annotations
from abc import ABC, abstractmethod

from synthetic_data.generation import GenerationWrapper
from synthetic_data.tasks import BaseTask
from synthetic_data.utils import Conversation


class TextEnv(ABC):
    task: BaseTask
    conversation: Conversation
    run_metadata: dict
    seed: int

    def __init__(self, generation_wrapper: GenerationWrapper, seed: int):
        self.generation_wrapper = generation_wrapper
        self.seed = seed

    @abstractmethod
    async def step(self) -> bool:
        """
        Perform a single step in the environment.
        Returns a boolean indicating if the episode is done.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the environment to the initial state.
        """
        pass
