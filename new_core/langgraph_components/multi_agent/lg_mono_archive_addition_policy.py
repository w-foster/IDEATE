
from operator import add
from typing import cast
from new_core.interfaces.archive_store import IArchiveStore
from new_core.langgraph_components.utils import to_langgraph_spec
from new_core.models.ai_model_spec import AIModelSpec
from new_core.models.archive_addition_decision import ArchiveAdditionDecision
from new_core.models.image_solution import ImageSolution
from new_core.models.run_config import RunConfig
from new_core.models.task_context import TaskContext

from new_core.interfaces.high_level.archive_addition_policy import IArchiveAdditionPolicy

from langgraphs.archive_addition.archive_addition_graph import (
    compile_graph as compile_archive_addition_graph,
    ArchiveAdditionState
)

# Using monolith LangGraph
class LGMonoArchiveAdditionPolicy(IArchiveAdditionPolicy):
    def __init__(self, ai_model_spec: AIModelSpec):
        self._ai_model_spec = ai_model_spec
        self._archive_addition_graph = compile_archive_addition_graph()

    async def decide(
        self,
        task_context: TaskContext,
        run_config: RunConfig,
        archive: IArchiveStore,
        new_solution: ImageSolution,
        flip_order: bool
    ) -> ArchiveAdditionDecision:

        input_state: ArchiveAdditionState = {
            "model_spec": to_langgraph_spec(self._ai_model_spec),
            "design_task": task_context.design_task,
            "domain_description": task_context.domain_description,
            "branch_context": None,  #TODO: add support later
            "archive_full": archive.is_full(),
            "archive_img_paths": [sol.img_path for sol in archive.all()],
            "flip_order": flip_order,
            "max_comparisons_at_once": run_config.max_solution_comparisons_per_call,
            "new_img_path": new_solution.img_path,
        }

        final_state = await self._archive_addition_graph.ainvoke(input_state)
        final_state = cast(ArchiveAdditionState, final_state)
        
        if "img_to_add_path" not in final_state:
            raise RuntimeError(f"key 'img_to_add_path' missing from final state of archive addition monolith graph")
        if "img_to_remove_path" not in final_state:
            raise RuntimeError(f"key 'img_to_remove_path' missing from final state of archive addition monolith graph")
    
        to_add_path = final_state["img_to_add_path"]
        to_remove_path = final_state["img_to_remove_path"]

        add_new_solution = to_add_path is not None
        # TODO: rework langgraph to use ID stuff as well! No more of this paths stuff
        remove_id = None
        for sol in archive.all():
            if sol.img_path == to_remove_path:
                remove_id = sol.id

        decision_reasoning = {"dummy": "To be added..."}  #TODO: handle this. ideally its the LangGraph's responsibility

        return ArchiveAdditionDecision(
            add_new_solution=add_new_solution,
            remove_id=remove_id,
            decision_reasoning=decision_reasoning
        )