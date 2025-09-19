from typing import cast

from new_core.interfaces.pairwise_evaluator import IPairwiseEvaluator
from new_core.langgraph_agents.utils import to_langgraph_spec
from new_core.models.ai_model_spec import AIModelSpec
from new_core.models.task_context import TaskContext
from new_core.models.image_solution import ImageSolution

from langgraphs.evaluation.pairwise_evaluation_graph import (
    compile_graph as compile_evaluation_graph,
    EvaluationState,
)


class LGPairwiseEvaluator(IPairwiseEvaluator):
    def __init__(self, ai_model_spec: AIModelSpec, flip_order: bool = False) -> None:
        self._ai_model_spec = ai_model_spec
        self._evaluation_graph = compile_evaluation_graph()
        self._flip_order = flip_order
    
    async def choose_winner(
        self,
        task_context: TaskContext,
        solution_one: ImageSolution,
        solution_two: ImageSolution
    ) -> ImageSolution:

        input_state: EvaluationState = {
            "model_spec": to_langgraph_spec(self._ai_model_spec),
            "design_task": task_context.design_task,
            "domain_description": task_context.domain_description,
            "img_file_names": (solution_one.img_path, solution_two.img_path),
            "flip_order": self._flip_order,
            "branch_context": None,
        }

        final_state = await self._evaluation_graph.ainvoke(input_state)
        final_state = cast(EvaluationState, final_state)

        if "winner_img_path" not in final_state:
            raise RuntimeError("key 'winner_img_path' not set in final evaluation graph state")
        winner_path = final_state["winner_img_path"]
        if winner_path == solution_one.img_path:
            return solution_one
        if winner_path == solution_two.img_path:
            return solution_two
        
        raise RuntimeError("key 'winner_img_path' is set, but does not match either provided solution")

