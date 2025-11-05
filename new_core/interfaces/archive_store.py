from abc import ABC, abstractmethod
from typing import Optional, Iterable
from new_core.models.archive_addition_decision import ArchiveAdditionDecision
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

    # non-policy convenience; wrapper around add and remove 
    def apply_decision(self, new_solution: ImageSolution, decision: ArchiveAdditionDecision) -> bool:
        """Applies ArchiveAdditionDecision; returns True if new solution was added, False otherwise"""
        if decision.add_new_solution:
            if decision.remove_id:
                self.remove(decision.remove_id)
            self.add(new_solution)
            return True
        return False