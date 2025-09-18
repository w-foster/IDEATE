from typing import Optional, Dict, Any, cast
from new_core.interfaces import archive_store
from new_core.interfaces.creative_strategist import ICreativeStrategist
from new_core.interfaces.archive_store import IArchiveStore
from new_core.models.archive_feedback import ArchiveFeedback
from new_core.models.creative_strategy import CreativeStrategy
from new_core.models.task_context import TaskContext

from langgraphs.strategising.creative_strategy_graph import (
    compile_graph as compile_creative_strategy_graph,
    CreativeStrategyState,
)
from langgraphs.strategising.refine_creative_strategy_graph import (
    compile_graph as compile_refine_creative_strategy_graph,
    CreativeStrategyRefinementState,
)
from new_core.models.task_guardrails import TaskGuardrails


class LGResearchCreativeStrategist(ICreativeStrategist):
    def __init__(self, refinement_interval: int, guardrails: TaskGuardrails) -> None:
        self._strategy_from_task_graph = compile_creative_strategy_graph()
        self._refine_strategy_graph = compile_refine_creative_strategy_graph()
        self._guardrails = guardrails
        self._refinement_interval = refinement_interval

    async def generate_strategy_from_task(self, task_context: TaskContext) -> CreativeStrategy:
        input_state: CreativeStrategyState = {
            "design_task": task_context.design_task,
            "domain_description": task_context.domain_description,
            "generated_strategy": None,
            "high_level_guardrails": self._guardrails.text,
            "branch_context": None    # TODO: extend support once branching is reworked
        }

        final_state = await self._strategy_from_task_graph.ainvoke(input_state)
        final_state = cast(CreativeStrategyState, final_state)
        
        if final_state["generated_strategy"] is None:
            raise RuntimeError("key 'generated_strategy' is None after graph invocation completed")
        return CreativeStrategy(
            text=final_state["generated_strategy"]
        )

    async def refine_existing_strategy(self, task_context: TaskContext, strategy: CreativeStrategy, feedback: ArchiveFeedback, archive: IArchiveStore) -> CreativeStrategy:
        input_state: CreativeStrategyRefinementState = {
            "design_task": task_context.design_task,
            "domain_description": task_context.domain_description,
            "current_strategy": strategy.text,
            "high_level_guardrails": self._guardrails.text,
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