from typing import Dict
from pydantic import BaseModel

class Solution(BaseModel):
    id: str
    prompt: str
    idea: str
    img_path: str
    metadata: Dict