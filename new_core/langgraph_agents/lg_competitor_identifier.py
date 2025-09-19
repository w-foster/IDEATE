

from typing import Optional, cast
from new_core.interfaces.archive_store import IArchiveStore
from new_core.interfaces.competitor_identifier import ICompetitorIdentifier
from new_core.models.ai_model_spec import AIModelSpec
from new_core.models.image_solution import ImageSolution

from langgraphs.competitor_identification.competitor_identification_graph import (
    compile_graph as compile_identify_most_similar_graph,
    OverallCompetitorIdentificationState
)

from langgraphs.novelty_checking.novelty_check_graph import (
    compile_graph as compile_identify_too_similar_or_none_graph,
    NoveltyCheckState
)
from new_core.models.run_config import RunConfig
from new_core.models.task_context import TaskContext


class LGCompetitorIdentifier(ICompetitorIdentifier):
    def __init__(self, ai_model_spec: AIModelSpec):
        self._ai_model_spec = ai_model_spec
        self._identify_most_similar_solution_graph = compile_identify_most_similar_graph()
        self._identify_too_similar_solution_or_none_graph = compile_identify_too_similar_or_none_graph()

    async def identify_competitor_or_none(self, task_context: TaskContext, run_config: RunConfig, new_solution: ImageSolution, archive: IArchiveStore) -> Optional[ImageSolution]:
        if archive.is_empty():
            return None
        
        if archive.is_full():
            competitor = await self._identify_most_similar(task_context, run_config, new_solution, archive)
            return competitor

        # else, archive not full
        optional_competitor: ImageSolution | None = await self._identify_too_similar_or_none(task_context, run_config, new_solution, archive)
        return optional_competitor

    
    async def _identify_most_similar(self, task_context: TaskContext, run_config: RunConfig, new_solution: ImageSolution, archive: IArchiveStore) -> ImageSolution:
        input_state: OverallCompetitorIdentificationState = {
            "design_task": task_context.design_task,
            "domain_description": task_context.domain_description,
            "max_comparisons": run_config.max_solution_comparisons_per_call,
            "new_img_path": new_solution.img_path,
            "archive_img_paths": [sol.img_path for sol in archive.all()],
        }

        final_state = await self._identify_most_similar_solution_graph.ainvoke(input_state)
        final_state = cast(OverallCompetitorIdentificationState, final_state)

        # TODO: bit annoying cos we get back a PATH... shld refactor the LangGraphs at some point.
        # But that's part of a wider rework tbh
        if "most_similar_img_path" not in final_state:
            raise RuntimeError("key 'most_similar_img_path' missing from final state during competitor identification")

        most_similar_path = final_state["most_similar_img_path"]
        most_similar_sol = None
        for sol in archive.all():
            if sol.img_path == most_similar_path:
                most_similar_sol = sol
                break
        
        if most_similar_sol is None:
            raise RuntimeError("key 'most_similar_img_path' does not match any img_path of archive solution")

        return most_similar_sol
    
    async def _identify_too_similar_or_none(self, task_context: TaskContext, run_config: RunConfig, new_solution: ImageSolution, archive: IArchiveStore) -> Optional[ImageSolution]:
        input_state: NoveltyCheckState = {
            "design_task": task_context.design_task,
            "domain_description": task_context.domain_description,
            "branch_context": None,  # TODO: add support later
            "archive_img_paths": [sol.img_path for sol in archive.all()],
            "new_img_path": new_solution.img_path 
        }

        final_state = await self._identify_too_similar_solution_or_none_graph.ainvoke(input_state)
        final_state = cast(NoveltyCheckState, final_state)

        if "is_novel_enough" not in final_state:
            raise RuntimeError("key 'is_novel_enough' missing from final state")
        if final_state["is_novel_enough"]:
            return None
        
        if "competing_img_paths" not in final_state:
            raise RuntimeError("key 'competing_img_paths' missing from final state")
        competing_img_paths = final_state["competing_img_paths"]
        
        if competing_img_paths[0] != new_solution.img_path and competing_img_paths[1] != new_solution.img_path:
            raise RuntimeError("neither of competing image paths match the new solution's path -- something is wrong")

        # so janky... see TODO above. Shld defo rework this langgraph
        if competing_img_paths[0] != new_solution.img_path:
            for sol in archive.all():
                if sol.img_path == competing_img_paths[0]:
                    return sol
        
        if competing_img_paths[1] != new_solution.img_path:
            for sol in archive.all():
                if sol.img_path == competing_img_paths[1]:
                    return sol
