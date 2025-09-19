from typing import Dict, Any
from pydantic import BaseModel


class AIModelSpec(BaseModel):
    provider: str  #e.g., openai
    name: str      #e.g., o3
    params: Dict[str, Any] = {}  #e.g., reasoning_effort: high