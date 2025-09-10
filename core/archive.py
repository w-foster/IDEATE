from typing import List, Optional, cast, Tuple
from pydantic import BaseModel

from langgraphs.archive_addition.archive_addition_graph import (
    compile_graph as compile_archive_addition_graph,
    ArchiveAdditionState,
    Evaluation
)
from core.solution import Solution
from core.branch_context import BranchContext


class Archive:
    def __init__(
        self,
        design_task: str,
        domain_description: str,
        max_capacity: int,
        max_comparisons_at_once: int = 5,
        check_novelty_before_full: bool = True,
        branch_context: Optional[BranchContext] = None
    ):
        self.design_task = design_task
        self.domain_description = domain_description
        self.max_capacity = max_capacity
        self.max_comparisons_at_once = max_comparisons_at_once
        self.check_novelty_before_full = check_novelty_before_full

        self.solutions: List[Solution] = []
        self.branch_context = branch_context

        self.archive_addition_handler = compile_archive_addition_graph()

    async def add(self, new_solution: Solution, flip_order: bool) -> Tuple[bool, Optional[ArchiveAdditionState]]:
        """
        Attempt to add a new Solution.
        Returns True if added, False if rejected.
        """

        added, workflow_output = await self._handle_new_solution(new_solution, flip_order)
        return added, workflow_output


    async def _handle_new_solution(self, new_solution: Solution, flip_order: bool) -> Tuple[bool, Optional[ArchiveAdditionState]]:
        if len(self.solutions) == 0:
            self.solutions.append(new_solution)
            return True, None

        workflow_input = ArchiveAdditionState(
            design_task=self.design_task,
            domain_description=self.domain_description,
            archive_img_paths=self.get_all_image_paths(),
            archive_full=len(self.solutions) >= self.max_capacity,
            new_img_path=new_solution.img_path,
            max_comparisons_at_once=self.max_comparisons_at_once,
            flip_order=flip_order,
            branch_context=self.branch_context
        )
        workflow_output = await self.archive_addition_handler.ainvoke(workflow_input)
        workflow_output = cast(ArchiveAdditionState, workflow_output)

        if "img_to_add_path" not in workflow_output:
            raise RuntimeError("img_to_add_path key missing from workflow output")
        if "img_to_remove_path" not in workflow_output:
            raise RuntimeError("img_to_remove_path key missing from workflow output")
        

        # fix this later this is so janky and shld just be an easy O(1)
        if workflow_output["img_to_add_path"]:
            # add the new sol
            self.solutions.append(new_solution)
            # remove old sol if necessary (if there was a competition)
            if workflow_output["img_to_remove_path"]:
                for sol in self.solutions:
                    if sol.img_path == workflow_output["img_to_remove_path"]:
                        self.solutions.remove(sol)
            return True, workflow_output
        
        # new sol not added
        return False, workflow_output


    def get_all_image_paths(self) -> List[str]:
        return [sol.img_path for sol in self.solutions]

    def get_all_solutions(self) -> List[Solution]:
        return self.solutions
