from typing import cast
from new_core.interfaces.prompt_engineer import IPromptEngineer
from new_core.langgraph_agents.utils import to_langgraph_spec
from new_core.models.ai_model_spec import AIModelSpec
from new_core.models.diffusion_prompt import DiffusionPrompt
from new_core.models.idea import Idea
from new_core.models.task_context import TaskContext

from langgraphs.blueprint_engineering.blueprint_engineering_graph import (
    compile_graph as compile_prompt_engineering_graph,
    BlueprintState as PromptEngineeringState
)


class LGPromptEngineer(IPromptEngineer):
    def __init__(self, ai_model_spec: AIModelSpec):
        self._ai_model_spec = ai_model_spec
        self._prompt_engineering_graph = compile_prompt_engineering_graph()

    async def idea_to_prompt(self, task_context: TaskContext, idea: Idea) -> DiffusionPrompt:
        input_state: PromptEngineeringState = {
            "model_spec": to_langgraph_spec(self._ai_model_spec),
            "branch_context": None,  #TODO: add support later
            "design_task": task_context.design_task,
            "domain_description": task_context.domain_description,
            "guidance": PROMPT_ENGINEERING_GUIDANCE,
            "idea": idea.text,
            "blueprint": None,  #TODO: swap to NotRequired
        }

        final_state = await self._prompt_engineering_graph.ainvoke(input_state)
        final_state = cast(PromptEngineeringState, final_state)

        if final_state["blueprint"] is None:
            raise RuntimeError("key 'blueprint' is None in final state of prompt engineering graph")
        
        return DiffusionPrompt(
            text=final_state["blueprint"]
        )




PROMPT_ENGINEERING_GUIDANCE = """
Prompt-Crafting Guidelines
- You are creating a prompt to be given to an advanced, August 2025 diffusion model for image generation
- You can assume that this model is highly capable
- You should be detailed in your prompt -- the more the better; but avoid too much flowery language, be direct and explicit with what you want (remember, this is for a diffusion model not an LLM)
- NEVER put things like 'Draw XYZ' or 'Illustrate an ABC', simply state the prompt in terms of what the image should be (e.g. 'an ABC')

Be Exceptionally Specific
- Call out exactly what you want:
- Subjects & Actions: What's in the scene, who's doing what.
- Colors & Lighting: “warm sunset hues,” “dramatic backlight,” “soft depth of field.”
- Mood & Atmosphere: “cinematic,” “hyper-realistic,” “whimsical.”

Layer Composition (if relevant)
- Even with a fleshed-out idea, organizing elements helps:
- Foreground / Midground / Background cues ensure spatial clarity.

Invoke Style & Technical Details
- You **MUST** the model your medium or camera settings:
- This should be based on the following decision process:
1) Is there an explicit style (e.g. 'oil painting', 'photo', 'anime', etc.)?
a) If YES: you **MUST** clearly state that style within the prompt, and don't mix up styles (e.g., if its 'watercolour', don't add 'cinematic quality')
b) If NO: you must make a judgement, but often defaulting to photo-realism is best (e.g., for "a boat", or "a road"), unless another style is implicit (e.g., "a logo of XYZ")
- EXAMPLES:
-- “oil painting with visible brushstrokes”
-- “shot on 50mm lens at golden hour, slight film grain”
-- “vector-art style with flat colors and bold outlines”
- ADDITIONAL RULE:
-- **DO NOT** mix styles remember; do NOT add '8K' or 'cinematic shot' to a task for a 'painting of XYZ' or 'ink-wash of ABC'
-- Only add such camera-like details in the case of photo-realism

Balance Constraints vs. Creative Freedom
- Constrain key details (“emerald scales glinting,” “laser-etched steel finish”)
- Leave room for flair (“in a fantastical setting,” “with imaginative lighting”)

EXAMPLES (for reference on successful prompts -- this is just for guidance, it isn't gospel)
"a poster for a luxurious traditional japanase bath house in san francisco named "Kontext" by "Black Forest Labs", with photo. minimal, plant aesthetic. natural, sensual. elite."

"Watercolor architectural illustration, abstract, aerial view of contemporary modern zen garden with house, pond with lily pads, wooden deck, mature trees and landscaping. Soft green palette, loose brush strokes, isometric perspective, hand-drawn sketch style."

"Retro game style, woman in old school suit, upper body, true detective, detailed character, nigh sky, crimson moon silhouette, american muscle car parked on dark street in background, complex background in style of Bill Sienkiewicz and Dave McKean and Carne Griffiths, extremely detailed, mysterious, grim, provocative, thrilling, dynamic, action-packed, fallout style, vintage, game theme, masterpiece, high contrast, stark. vivid colors, 16-bit, pixelated, textured, distressed"

"a miniature world, oblique photography, macro (photography), many vehicles on a city overpass, bird's-eye view, tilted composition, simple solid background, white and light green style, white and pink cherry blossom trees, high-definition details, Canon camera, 8k+"

"noon hard light, frosted glass serum bottle with label "FLUX.1" on beige travertine block, concrete backdrop, terracotta linen draped at sharp angle, overhead sun creating precise shadow beneath, clean editorial minimalism, delicate, grainy, subtle, 8K"
"""