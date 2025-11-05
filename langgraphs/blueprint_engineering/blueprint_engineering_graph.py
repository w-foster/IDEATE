from typing import TypedDict, Dict, List, Optional, cast
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
import os
import asyncio
from rich import print 

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent, InjectedState

from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_tavily import TavilySearch

from langgraphs.blueprint_engineering import prompts
from core.branch_context import BranchContext
from langgraphs.types import LGModelSpec
from langgraphs.utils import agent_model_name


load_dotenv(find_dotenv())



class Blueprint(BaseModel):
    blueprint: str = Field(
        description="The 'blueprint' which you are refining the idea into; this could be code, a prompt, etc., depending on the domain."
    )


class BlueprintState(TypedDict):
    model_spec: LGModelSpec
    design_task: str
    domain_description: str

    guidance: str

    idea: str
    blueprint: Optional[str]
    branch_context: Optional[BranchContext]



async def start_blueprint_engineering(state: BlueprintState) -> Dict:
    return {}


async def generate_blueprint(state: BlueprintState) -> Dict:
    blueprint_engineer = create_react_agent(
        model=agent_model_name(state["model_spec"]),
        tools=[],
        prompt=prompts.create_blueprint_system_prompt(
            is_convergence_branch=state.get("branch_context") is not None
        ),
        response_format=Blueprint
    )

    user_prompt = prompts.create_user_blueprint_engineering_prompt(
        idea=state["idea"],
        design_task=state["design_task"],
        domain_description=state["domain_description"],
        guidance=state["guidance"],
        branch_context=state.get("branch_context")
    )
    input = {"messages": {"role": "user", "content": user_prompt}}

    print(f"\n\n==== Creating Prompt from Idea ====\n\n")

    raw_response = await blueprint_engineer.ainvoke(input)
    output: Blueprint = raw_response["structured_response"]

    print(f"\n\n==== CREATED PROMPT: ====\n{output.blueprint}\n")

    return {"blueprint": output.blueprint}



def compile_graph():
    builder = StateGraph(BlueprintState)

    builder.add_node("start_blueprint_engineering", start_blueprint_engineering)
    builder.add_node("generate_blueprint", generate_blueprint)

    builder.add_edge(START, "start_blueprint_engineering")
    builder.add_edge("start_blueprint_engineering", "generate_blueprint")
    builder.add_edge("generate_blueprint", END)

    return builder.compile()


# async def run():
#     graph = compile_graph()

#     input = BlueprintState(
#         design_task="an architectural style that's never been seen before",
#         domain_description="image generation with advanced diffusion model, July 2025",
#         guidance="The model being used is FLUX1.1 KONTEXT. Prompts can be long, such as a paragraph (but probably 350 words strict maximum), and should be highly detailed -- rather than leaving any ambiguity up to the model, being explicit about details will generally yield better results. Note, there are no negative prompt tags like '--no xyz', or other tags like '[...]', but you can specify if you don't want something to happen in the prompt. The prompt, then, should be highly detailed and reflect the spirit/content/semantics of the IDEA that is given, just in a way that makes sense for FLUX1.1 KONTEXT.",
#         idea="""
# MycoSonic Filigree

# Concept
# A living, lace-like façade where interwoven acoustic ribs and bioluminescent fungal nodes form a dynamic “instrument skin” that breathes wind, footsteps, and voices into cascades of light and harmonic resonance.

# Visual Vocabulary

# Filigree Lattice: Delicate, triangular acoustic ribs in deep bronze and emerald tones, forming a semi-transparent grid

# Mycelial Nodes: Hexagonal spore capsules at each rib junction, glowing in shifting bioluminescent hues

# Resonant Shadowplay: Perforations in the ribs cast moving spectrogram-like patterns of light and shadow onto surrounding surfaces

# Material & Technology

# Adaptive Acoustic Ribs: Composite polymer-metal panels with shape-memory alloy cores that adjust micro-chamber depths to retune wind-driven overtones and regulate airflow

# Biolume Spore Capsules: Genetically engineered mycelium infused with multi-phase luciferase, housed in translucent resin shells for color-cycling glow

# SporeSynth Nodes: Modular bio-electronic connectors combining piezoelectric sensors, micro-actuators, and fungal tissue—transducing ambient stimuli into synchronized waves of light and sound

# Hidden Structural Frame: A slender steel-timber hybrid skeleton that carries loads and conceals utilities behind the living filigree

# Signature Move
# SporeSynth Node
# At each intersection, the SporeSynth Node unites living fungus, acoustic metamaterial, and sensor-actuator networks. As wind stirs the lattice and visitors approach, these nodes orchestrate cascading pulses of color and harmonic resonance—transforming the entire façade into a sentient, musical organism that guides movement, heals urban noise, and illuminates the night.
# """,
#     blueprint=None,
#     branch_context=None
#     )

#     output = await graph.ainvoke(input)

#     print(output)



# if __name__ == "__main__":
#     asyncio.run(run())