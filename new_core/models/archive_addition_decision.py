from typing import Optional, Dict
from pydantic import BaseModel

class ArchiveAdditionDecision(BaseModel):
    add_new_solution: bool 
    remove_id: Optional[str]
    decision_reasoning: Optional[Dict[str, str]]  # can just dump this if saving? cld store e.g., evaluator reasoning, novelty check reasoning , idk