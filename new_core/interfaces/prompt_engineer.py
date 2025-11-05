from abc import ABC, abstractmethod
from new_core.models.idea import Idea
from new_core.models.diffusion_prompt import DiffusionPrompt
from new_core.models.task_context import TaskContext

class IPromptEngineer(ABC):
    @abstractmethod
    async def idea_to_prompt(self, task_context: TaskContext, idea: Idea) -> DiffusionPrompt:
        ...