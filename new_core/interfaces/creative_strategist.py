from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from new_core.models.archive_feedback import ArchiveFeedback
from new_core.models.creative_strategy import CreativeStrategy
from new_core.models.task_constraints import TaskConstraints
from new_core.models.task_context import TaskContext


class ICreativeStrategist(ABC):
    @abstractmethod
    async def generate_strategy_from_task(self, task_context: TaskContext, task_constraints: TaskConstraints) -> CreativeStrategy:
        ...

    @abstractmethod
    async def refine_existing_strategy(
        self, 
        task_context: TaskContext, 
        task_constraints: TaskConstraints,
        strategy: CreativeStrategy, 
        feedback: ArchiveFeedback
    ) -> CreativeStrategy:
        ... 


