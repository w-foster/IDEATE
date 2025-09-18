from abc import ABC, abstractmethod
from typing import List

from new_core.interfaces.archive_store import IArchiveStore
from new_core.models.image_solution import ImageSolution

class IParentSelector(ABC):
    @abstractmethod
    async def select(self, archive: IArchiveStore, num_parents: int) -> List[ImageSolution]:
        ...