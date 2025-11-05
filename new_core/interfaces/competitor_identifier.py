from abc import ABC, abstractmethod
from typing import Optional

from new_core.models.image_solution import ImageSolution
from new_core.interfaces.archive_store import IArchiveStore

class ICompetitorIdentifier(ABC):
    @abstractmethod
    async def identify_competitor_or_none(self, new_solution: ImageSolution, archive: IArchiveStore) -> Optional[ImageSolution]:
        ...