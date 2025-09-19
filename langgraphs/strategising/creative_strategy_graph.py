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

from langgraphs.strategising import prompts
# remove this dependency eventually
from core.branch_context import BranchContext

load_dotenv(find_dotenv())

CREATIVE_STRATEGY_LLM_MODEL = os.getenv("CREATIVE_STRATEGY_LLM_MODEL") or "openai:o3"
# LLM_API_KEY = os.getenv("LLM_API_KEY")


class CreativeStrategy(BaseModel):
    creative_strategy: str = Field(
        description="The entire creative strategy, in detail",
        default=""
    )

class StrategyGuardrails(BaseModel):
    guardrails: str = Field(
        description="The list of guardrails/constraints you are outputting",
        default=""
    )


class CreativeStrategyState(TypedDict):
    design_task: str
    domain_description: str
    generated_strategy: Optional[str]
    high_level_guardrails: Optional[str]
    branch_context: Optional[BranchContext]


async def start_strategy_generation(state: CreativeStrategyState) -> Dict:
    return {}


async def generate_high_level_guardrails(state: CreativeStrategyState) -> Dict:
    guardrails_generator = create_react_agent(
        model=CREATIVE_STRATEGY_LLM_MODEL,
        prompt=prompts.create_constraints_system_prompt(is_convergence_branch=state["branch_context"] is not None),
        response_format=StrategyGuardrails,
        tools=[]
    )

    user_prompt: HumanMessage = prompts.create_user_constraints_request_prompt(
        domain_description=state["domain_description"],
        design_task=state["design_task"],
        branch_context=state["branch_context"]
    )

    input = {"messages": user_prompt}
    print(f"\n\n{input}\n\n")

    raw_response = await guardrails_generator.ainvoke(input)
    print(f"\n\n{raw_response}\n\n")

    output: StrategyGuardrails = raw_response["structured_response"]

    return {"high_level_guardrails": output.guardrails}


async def generate_strategy(state: CreativeStrategyState) -> Dict:
    strategy_generator = create_react_agent(
        model=CREATIVE_STRATEGY_LLM_MODEL,
        prompt=prompts.create_creative_strategy_generation_system_prompt(is_convergence_branch=state["branch_context"] is not None),
        tools=[TavilySearch(max_results=10)],
        response_format=CreativeStrategy
    )

    if not state["high_level_guardrails"]:
        raise RuntimeError("Guardrails missing from Graph state at time of Creative Strategy generation.")
    
    user_prompt: HumanMessage = prompts.create_user_strategy_request_prompt(
        domain_description=state["domain_description"],
        design_task=state["design_task"],
        guardrails=state["high_level_guardrails"],
        branch_context=state["branch_context"]
    )
    #input = {"messages": {"role": "user", "content": user_prompt}}
    input = {"messages": user_prompt}
    print(f"\n\n{input}\n\n")

    raw_response = await strategy_generator.ainvoke(input)
    print(f"\n\n{raw_response}\n\n")

    output: CreativeStrategy = raw_response["structured_response"]
    return {"generated_strategy": output.creative_strategy}




def compile_graph():
    builder = StateGraph(CreativeStrategyState)

    builder.add_node("start_strategy_generation", start_strategy_generation)
    builder.add_node("generate_high_level_guardrails", generate_high_level_guardrails)
    builder.add_node("generate_strategy", generate_strategy)
    
    builder.add_edge(START, "start_strategy_generation")
    builder.add_edge("start_strategy_generation", "generate_high_level_guardrails")
    builder.add_edge("generate_high_level_guardrails", "generate_strategy")
    builder.add_edge("generate_strategy", END)

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
