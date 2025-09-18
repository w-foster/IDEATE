from abc import ABC, abstractmethod
from typing import List, Optional
from new_core.interfaces.archive_store import IArchiveStore
from new_core.models.creative_strategy import CreativeStrategy
from new_core.models.task_context import TaskContext
from new_core.models.image_solution import ImageSolution
from new_core.models.idea import Idea


class IIdeator(ABC):
    @abstractmethod
    async def ideate(
        self, 
        task_context: TaskContext, 
        parent_solutions: Optional[List[ImageSolution]], 
        strategy: CreativeStrategy,
        archive: IArchiveStore
    ) -> Idea:
        ...