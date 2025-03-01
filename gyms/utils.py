from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from synthetic_data.generation import GenerationWrapper


class TextEnv(ABC):
    def __init__(self, generation_wrapper: GenerationWrapper):
        self.generation_wrapper = generation_wrapper

    @abstractmethod
    async def step(self) -> Tuple[float, bool]:
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None):
        pass
