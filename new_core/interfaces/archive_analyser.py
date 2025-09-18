from abc import ABC, abstractmethod

from new_core.models.archive_feedback import ArchiveFeedback
from new_core.models.creative_strategy import CreativeStrategy
from new_core.interfaces.archive_store import IArchiveStore


class IArchiveAnalyser(ABC):
    @abstractmethod
    async def generate_feedback(self, archive: IArchiveStore, current_strategy: CreativeStrategy) -> ArchiveFeedback:
        ...