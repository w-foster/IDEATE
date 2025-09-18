from abc import ABC, abstractmethod
from typing import Dict, Tuple

from new_core.models.diffusion_prompt import DiffusionPrompt

class IImageGenerator(ABC):
    @abstractmethod
    async def generate(self, prompt: DiffusionPrompt) -> Tuple[str, Dict]:  #TODO: make more explicit return type
        ...