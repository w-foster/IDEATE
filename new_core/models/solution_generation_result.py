from typing import Dict
from pydantic import BaseModel

from new_core.models.diffusion_prompt import DiffusionPrompt
from new_core.models.idea import Idea

class SolutionGenerationResult(BaseModel):
    idea: Idea
    prompt: DiffusionPrompt
    sample_path: str
    gen_metadata: Dict