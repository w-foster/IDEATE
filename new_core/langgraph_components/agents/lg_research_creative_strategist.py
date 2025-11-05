from typing import Optional, Dict, Any, cast, override
from new_core.interfaces.creative_strategist import ICreativeStrategist
from new_core.interfaces.archive_store import IArchiveStore
from new_core.langgraph_components.utils import to_langgraph_spec
from new_core.models.ai_model_spec import AIModelSpec
from new_core.models.archive_feedback import ArchiveFeedback
from new_core.models.creative_strategy import CreativeStrategy
from new_core.models.task_constraints import TaskConstraints
from new_core.models.task_context import TaskContext

from langgraphs.strategising.external_constraints_creative_strategy_graph import (
    compile_graph as compile_creative_strategy_graph,
    CreativeStrategyState
)
from langgraphs.strategising.refine_creative_strategy_graph import (
    compile_graph as compile_refine_creative_strategy_graph,
    CreativeStrategyRefinementState,
)


class LGResearchCreativeStrategist(ICreativeStrategist):
    def __init__(self, ai_model_spec: AIModelSpec, refinement_interval: int) -> None:
        self._ai_model_spec = ai_model_spec
        self._strategy_from_task_graph = compile_creative_strategy_graph()
        self._refine_strategy_graph = compile_refine_creative_strategy_graph()        
        self._refinement_interval = refinement_interval

    async def generate_strategy_from_task(self, task_context: TaskContext, task_constraints: TaskConstraints) -> CreativeStrategy:
        input_state: CreativeStrategyState = {
            "model_spec": to_langgraph_spec(self._ai_model_spec),
            "design_task": task_context.design_task,
            "domain_description": task_context.domain_description,
            "high_level_task_constraints": task_constraints.text,
        }

        final_state = await self._strategy_from_task_graph.ainvoke(input_state)
        final_state = cast(CreativeStrategyState, final_state)
        
        if "generated_strategy" not in final_state:
            raise RuntimeError("key 'generated_strategy' is missing from final state of creative strategy graph")
        return CreativeStrategy(
            text=final_state["generated_strategy"]
        )

    @override
    async def refine_existing_strategy(
        self, 
        task_context: TaskContext, 
        task_constraints: TaskConstraints,
        strategy: CreativeStrategy, 
        feedback: ArchiveFeedback, 
        archive: IArchiveStore
    ) -> CreativeStrategy:
        input_state: CreativeStrategyRefinementState = {
            "model_spec": to_langgraph_spec(self._ai_model_spec),
            "design_task": task_context.design_task,
            "domain_description": task_context.domain_description,
            "current_strategy": strategy.text,
            "high_level_guardrails": task_constraints.text,
            "archive_solutions": list(archive.all()),     # TODO: rework langgraphs to use ImageSolution??
            "num_offspring": self._refinement_interval,
            "archive_analysis": feedback.text,
            "refined_strategy": "",
            "branch_context": None
        }

        final_state = await self._refine_strategy_graph.ainvoke(input_state)
        final_state = cast(CreativeStrategyRefinementState, final_state)

        if final_state["refined_strategy"] is None or final_state["refined_strategy"] == "":
            raise RuntimeError("key 'refined_strategy' is None or '' after graph invocation completed")
        return CreativeStrategy(
            text=final_state["refined_strategy"]
        )