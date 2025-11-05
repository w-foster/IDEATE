from abc import ABC, abstractmethod

from new_core.interfaces.archive_store import IArchiveStore
from new_core.models.archive_addition_decision import ArchiveAdditionDecision
from new_core.models.image_solution import ImageSolution
from new_core.models.run_config import RunConfig
from new_core.models.task_context import TaskContext


class IArchiveAdditionPolicy(ABC):
    @abstractmethod
    async def decide(
        self,
        task_context: TaskContext,
        run_config: RunConfig,
        archive: IArchiveStore,
        new_solution: ImageSolution,
        randomise_order_for_llm: bool
    ) -> ArchiveAdditionDecision:
        ...