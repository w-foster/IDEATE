from typing import List
import random

from new_core.interfaces.archive_store import IArchiveStore
from new_core.interfaces.parent_selector import IParentSelector
from new_core.models.image_solution import ImageSolution


class RandomSelector(IParentSelector):
    async def select(self, archive: IArchiveStore, num_parents: int) -> List[ImageSolution]:
        if num_parents <= 0:
            return []
        
        candidates = list(archive.all())
        if not candidates:
            return []

        k = min(num_parents, len(candidates))
        return random.sample(candidates, k)  #TODO: add a global seed for stuff like this. maybe via env var

