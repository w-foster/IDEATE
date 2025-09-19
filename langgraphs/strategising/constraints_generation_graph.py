from typing import NotRequired, TypedDict, Dict, List, Optional, cast
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

from core import branch_context
from langgraphs.strategising import prompts
# remove this dependency eventually
from core.branch_context import BranchContext

load_dotenv(find_dotenv())

CREATIVE_STRATEGY_LLM_MODEL = os.getenv("CREATIVE_STRATEGY_LLM_MODEL") or "openai:o3"   #TODO: consider moving model names/versions away from env vars surely
# LLM_API_KEY = os.getenv("LLM_API_KEY")


class CreativeStrategy(BaseModel):
    creative_strategy: str = Field(
        description="The entire creative strategy, in detail",
        default=""
    )

class TaskConstraints(BaseModel):
    guardrails: str = Field(
        description="The list of guardrails/constraints you are outputting",
        default=""
    )

class ConstraintsGenerationState(TypedDict):
    design_task: str
    domain_description: str
    generated_constraints: NotRequired[str]



async def generate_high_level_constraints(state: ConstraintsGenerationState) -> Dict:
    constraints_generator = create_react_agent(
        model=CREATIVE_STRATEGY_LLM_MODEL,
        prompt=prompts.create_constraints_system_prompt(is_convergence_branch=False),  # TODO: support convergence later
        response_format=TaskConstraints,
        tools=[]
    )

    user_prompt: HumanMessage = prompts.create_user_constraints_request_prompt(
        domain_description=state["domain_description"],
        design_task=state["design_task"],
        branch_context=None,  # TODO: support convergence later  
    )

    input = {"messages": user_prompt}
    print(f"\n\n{input}\n\n")

    raw_response = await constraints_generator.ainvoke(input)
    print(f"\n\n{raw_response}\n\n")

    output: TaskConstraints = raw_response["structured_response"]

    return {"high_level_guardrails": output.guardrails}



def compile_graph():
    builder = StateGraph(ConstraintsGenerationState)

    builder.add_node("generate_high_level_constraints", generate_high_level_constraints)
    
    builder.add_edge(START, "generate_high_level_constraints")
    builder.add_edge("generate_high_level_constraints", END)

    return builder.compile()




async def run():
    graph = compile_graph()

    input = CreativeStrategyState(
        design_task="a road",
        domain_description="image generation via an advanced diffusion model",
        generated_strategy=None,
        high_level_guardrails=None,
        branch_context=None
    )

    output = await graph.ainvoke(input)


if __name__ == "__main__":
    asyncio.run(run())
