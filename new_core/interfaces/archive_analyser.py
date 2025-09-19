from abc import ABC, abstractmethod

from new_core.models.archive_feedback import ArchiveFeedback
from new_core.models.creative_strategy import CreativeStrategy
from new_core.interfaces.archive_store import IArchiveStore
from new_core.models.run_config import RunConfig


class IArchiveAnalyser(ABC):
    @abstractmethod
    async def generate_feedback(self, run_config: RunConfig, archive: IArchiveStore, current_strategy: CreativeStrategy) -> ArchiveFeedback:
        ...