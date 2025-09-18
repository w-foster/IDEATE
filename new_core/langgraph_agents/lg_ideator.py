from typing import Optional, List, cast
from new_core.interfaces.archive_store import IArchiveStore
from new_core.interfaces.ideator import IIdeator
from new_core.models.creative_strategy import CreativeStrategy
from new_core.models.idea import Idea
from new_core.models.image_solution import ImageSolution
from new_core.models.task_context import TaskContext

from langgraphs.ideation.ideation_graph import (
    compile_graph as compile_ideation_graph,
    IdeationState
)


class LGIdeator(IIdeator):
    def __init__(self):
        self._ideation_graph = compile_ideation_graph()

    async def ideate(
        self, 
        task_context: TaskContext, 
        parent_solutions: Optional[List[ImageSolution]], 
        strategy: CreativeStrategy,
        archive: IArchiveStore
    ) -> Idea:

        # handle None parent_solutions --> i think langgraph needs reworking for that
        if parent_solutions is None:
            raise RuntimeError("parent solution(s) must be selected; support for 'None' will be extended later")
        
        # TODO: consider how we want stuff liek this to be done.
        # There shld defo be a big rework of the langgraphs
        # but i guess we don't rly want strong coupling from the langgraphs
        # back onto our interfaces (e.g., passing an IArchiveStore and letting it figure it out)
        # ... maybe there is a middle ground. Either way there's defo cleanup needed
        archive_ideas_except_seeds = []
        for sol in archive.all():
            is_parent = False
            for parent in parent_solutions:
                if sol.id == parent.id:
                    is_parent = True
                    break
            if is_parent:
                continue
            archive_ideas_except_seeds.append(sol.idea.text)

        input_state: IdeationState = {
            "design_task": task_context.design_task,
            "seed_ideas": [sol.idea.text for sol in parent_solutions],  # TODO: replace 'seed' with 'parent'
            "archive_ideas_except_seeds": archive_ideas_except_seeds,
            "branch_context": None,  # TODO: add support later
            "creative_strategy": strategy.text,
            "domain_descripton": task_context.domain_description,
            "new_idea": None  # TODO: change to NotRequired or whatever. But tbh thats annoying too... idk. Maybe i shld look into non-common I/O states
        }

        final_state = await self._ideation_graph.ainvoke(input_state)
        final_state = cast(IdeationState, final_state)

        if "new_idea" not in final_state:
            raise RuntimeError("key 'new_idea' missing from final state of ideation graph")
        if final_state["new_idea"] is None:
            raise RuntimeError("key 'new_idea' is 'None' in final state of ideation graph")
            
        return Idea(
            text=final_state["new_idea"]
        )
