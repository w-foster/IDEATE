from abc import ABC, abstractmethod
from new_core.models.image_solution import ImageSolution
from new_core.models.task_context import TaskContext

class IPairwiseEvaluator(ABC):
    @abstractmethod
    async def choose_winner(self, task_context: TaskContext, solution_one: ImageSolution, solution_two: ImageSolution) -> ImageSolution:
        ...