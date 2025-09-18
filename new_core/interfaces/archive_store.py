from abc import ABC, abstractmethod
from typing import Optional, Iterable
from new_core.models.image_solution import ImageSolution

class IArchiveStore(ABC):
    @abstractmethod
    def add(self, sol: ImageSolution) -> None:
        ...

    @abstractmethod
    def remove(self, sol_id: str) -> None:
        ...

    @abstractmethod
    def get(self, sol_id: str) -> Optional[ImageSolution]:
        ...

    @abstractmethod
    def all(self) -> Iterable[ImageSolution]:
        ...

    @abstractmethod
    def size(self) -> int:
        ...

    @abstractmethod
    def is_full(self) -> bool:
        ...

    @abstractmethod
    def is_empty(self) -> bool:
        ...