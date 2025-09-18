from typing import Dict, Iterable, Optional
from new_core.interfaces.archive_store import IArchiveStore
from new_core.models.image_solution import ImageSolution


class InMemoryImageArchiveStore(IArchiveStore):
    def __init__(self, max_capacity: int) -> None:
        self._max_capacity = max_capacity
        self._solutions: Dict[str, ImageSolution] = {}

    def add(self, sol: ImageSolution) -> None:
        # allow replace if id already exists; otherwise enforce capacity
        if sol.id not in self._solutions and self.is_full():
            raise ValueError("archive full")
        self._solutions[sol.id] = sol

    def remove(self, sol_id: str) -> None:
        self._solutions.pop(sol_id, None)

    def get(self, sol_id: str) -> Optional[ImageSolution]:
        return self._solutions.get(sol_id)

    def all(self) -> Iterable[ImageSolution]:
        # return a snapshot list to decouple callers from internal dict
        return list(self._solutions.values())

    def size(self) -> int:
        return len(self._solutions)

    def is_full(self) -> bool:
        return self.size() >= self._max_capacity