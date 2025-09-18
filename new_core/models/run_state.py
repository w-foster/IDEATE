from typing import Optional, Dict
from pydantic import BaseModel, Field

from new_core.models.archive_feedback import ArchiveFeedback
from new_core.models.creative_strategy import CreativeStrategy

class StrategyEntry(BaseModel):
    id: str
    strategy: CreativeStrategy
    previous_archive_analysis: Optional[ArchiveFeedback]

class RunState(BaseModel):
    current_generation: int = 0
    solutions_generated: int = 0
    current_strategy_id: Optional[str] = None
    strategies: Dict[str, StrategyEntry] = Field(default_factory=dict)
    next_strategy_seq: int = 1 

    def push_strategy_and_activate(self, strategy: CreativeStrategy, previous_archive_analysis: Optional[ArchiveFeedback] = None) -> str:
        strategy_id = f"strat_{self.next_strategy_seq:03d}"
        self.next_strategy_seq += 1
        self.strategies[strategy_id] = StrategyEntry(
            id=strategy_id, 
            strategy=strategy, 
            previous_archive_analysis=previous_archive_analysis
        )
        self.current_strategy_id = strategy_id
        return strategy_id
    
    def get_active_strategy(self) -> CreativeStrategy:
        if self.current_strategy_id not in self.strategies:
            raise RuntimeError("attempting to get active strategy but no strategy is set")
        return self.strategies[self.current_strategy_id].strategy