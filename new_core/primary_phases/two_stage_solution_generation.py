


from typing import List, Optional
from new_core.interfaces.archive_store import IArchiveStore
from new_core.interfaces.high_level.solution_generator import ISolutionGenerator
from new_core.interfaces.ideator import IIdeator
from new_core.interfaces.image_generator import IImageGenerator
from new_core.interfaces.prompt_engineer import IPromptEngineer
from new_core.models.creative_strategy import CreativeStrategy
from new_core.models.diffusion_prompt import DiffusionPrompt
from new_core.models.idea import Idea
from new_core.models.image_solution import ImageSolution
from new_core.models.solution_generation_result import SolutionGenerationResult
from new_core.models.task_context import TaskContext


class TwoStageSolutionGeneration(ISolutionGenerator):
    def __init__(
        self,
        ideator: IIdeator,
        prompt_engineer: IPromptEngineer,
        image_generator: IImageGenerator
    ) -> None:
        self._ideator = ideator
        self._prompt_engineer = prompt_engineer
        self._image_generator = image_generator

    async def generate(
        self, 
        task_context: TaskContext, 
        parent_solutions: Optional[List[ImageSolution]], 
        strategy: CreativeStrategy,
        archive: IArchiveStore
    ) -> SolutionGenerationResult:

        new_idea: Idea = await self._ideator.ideate(
            task_context=task_context,
            parent_solutions=parent_solutions,
            strategy=strategy,
            archive=archive
        )

        new_prompt: DiffusionPrompt = await self._prompt_engineer.idea_to_prompt(
            task_context=task_context,
            idea=new_idea
        )

        sample_path, gen_metadata = await self._image_generator.generate(
            prompt=new_prompt
        )

        return SolutionGenerationResult(
            idea=new_idea,
            prompt=new_prompt,
            sample_path=sample_path,
            gen_metadata=gen_metadata
        )






