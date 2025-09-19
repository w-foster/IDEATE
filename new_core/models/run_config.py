from typing import Optional
from pydantic import BaseModel

class RunConfig(BaseModel):
    generations: int
    strategy_refinement_interval: int
    max_archive_capacity: int
    max_solution_comparisons_per_call: int
    offspring_per_generation: int
    parents_per_ideation: int
    initial_ideation_count: int
    randomise_order_for_llm: bool
    random_seed: Optional[int] = None


