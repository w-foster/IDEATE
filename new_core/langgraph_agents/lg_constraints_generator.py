

from typing import cast
from new_core.interfaces.constraints_generator import IConstraintsGenerator

from langgraphs.strategising.constraints_generation_graph import (
    compile_graph as compile_constraints_generation_graph,
    ConstraintsGenerationState
)
from new_core.models.task_constraints import TaskConstraints
from new_core.models.task_context import TaskContext

class LGConstraintsGenerator(IConstraintsGenerator):
    def __init__(self, ai_model_spec: AIModelSpec):
        self._ai_model_spec = ai_model_spec
        self._constraints_generation_graph = compile_constraints_generation_graph()
    
    async def generate_constraints_for_task(
        self,
        task_context: TaskContext
    ) -> TaskConstraints:

        input_state: ConstraintsGenerationState = {
            "design_task": task_context.design_task,
            "domain_description": task_context.domain_description
        }

        final_state = await self._constraints_generation_graph.ainvoke(input_state)
        final_state = cast(ConstraintsGenerationState, final_state)

        if "generated_constraints" not in final_state:
            raise RuntimeError("key 'generated_constraints' missing from final state of constraints generation graph")
        if final_state["generated_constraints"] is None:
            raise RuntimeError("key 'generated_constraints' is 'None' in final state of constraints generation graph")

        return TaskConstraints(
            text=final_state["generated_constraints"]
        )

