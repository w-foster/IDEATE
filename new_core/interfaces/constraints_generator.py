from abc import ABC, abstractmethod

from new_core.models.task_context import TaskContext
from new_core.models.task_constraints import TaskConstraints

class IConstraintsGenerator(ABC):
    @abstractmethod
    async def generate_constraints_for_task(
        self,
        task_context: TaskContext
    ) -> TaskConstraints:
        ...