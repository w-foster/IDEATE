from typing import Dict
from pydantic import BaseModel, Field

from new_core.models.diffusion_prompt import DiffusionPrompt
from new_core.models.idea import Idea

class ImageSolution(BaseModel):
    id: str
    prompt: DiffusionPrompt
    idea: Idea
    img_path: str
    metadata: Dict = Field(default_factory=dict)