from typing import Optional, Dict
from pydantic import BaseModel, PrivateAttr

from new_core.models.task_constraints import TaskConstraints
from new_core.models.archive_feedback import ArchiveFeedback
from new_core.models.creative_strategy import CreativeStrategy

class StrategyEntry(BaseModel):
    id: str
    strategy: CreativeStrategy
    previous_archive_analysis: Optional[ArchiveFeedback]

# TODO: add more getters/setters to emulate a private-membered dataclass
# e.g., 'increment_solutions_generated' instead of setting it to += 1 directly
class RunState(BaseModel):
    _current_generation: int = PrivateAttr(default=0)
    _solutions_generated: int = PrivateAttr(default=0)
    _task_constraints: Optional[TaskConstraints] = PrivateAttr(default=None)
    _active_strategy_id: Optional[str] = PrivateAttr(default=None)
    _strategies: Dict[str, StrategyEntry] = PrivateAttr(default_factory=dict)

    _next_strategy_seq: int = 1 

    def push_strategy_and_activate(self, strategy: CreativeStrategy, previous_archive_analysis: Optional[ArchiveFeedback] = None) -> str:
        strategy_id = f"strat_{self._next_strategy_seq:03d}"
        self._next_strategy_seq += 1
        self._strategies[strategy_id] = StrategyEntry(
            id=strategy_id, 
            strategy=strategy, 
            previous_archive_analysis=previous_archive_analysis
        )
        self._active_strategy_id = strategy_id
        return strategy_id

    # -- Setters --
    def set_task_constraints(self, constraints: TaskConstraints) -> None:
        self._task_constraints = constraints

    def increment_num_solutions_generated(self) -> None:
        self._solutions_generated += 1
    
    def increment_current_generation_num(self) -> None:
        self._current_generation += 1

    # -- Complex Getters --
    def get_active_strategy_or_throw(self) -> CreativeStrategy:
        if self._active_strategy_id not in self._strategies:
            raise RuntimeError("attempting to get active strategy but no strategy is set")
        return self._strategies[self._active_strategy_id].strategy.model_copy(deep=True)
    
    # -- Simple Getters --
    def get_strategies_map(self) -> dict[str, StrategyEntry]:
        # deep copy for immutability
        return {k: v.model_copy(deep=True) for k, v in self._strategies.items()}

    def get_active_strategy_id(self) -> Optional[str]:
        return self._active_strategy_id
    
    def get_current_generation_num(self) -> int:
        return self._current_generation
    
    def get_num_solutions_generated(self) -> int:
        return self._solutions_generated
    
    def get_task_constraints(self) -> Optional[TaskConstraints]:
        if self._task_constraints is None:
            return None
        return self._task_constraints.model_copy(deep=True)
    
    def get_next_solution_num(self) -> int:
        return self._solutions_generated + 1
    
