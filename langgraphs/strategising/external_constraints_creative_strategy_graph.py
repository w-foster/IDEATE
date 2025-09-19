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


class CreativeStrategyState(TypedDict):
    design_task: str
    domain_description: str
    high_level_task_constraints: str

    generated_strategy: NotRequired[str]




async def generate_strategy(state: CreativeStrategyState) -> Dict:
    strategy_generator = create_react_agent(
        model=CREATIVE_STRATEGY_LLM_MODEL,
        prompt=prompts.create_creative_strategy_generation_system_prompt(is_convergence_branch=False),  #TODO: add support later
        tools=[TavilySearch(max_results=10)],
        response_format=CreativeStrategy
    )
    
    user_prompt: HumanMessage = prompts.create_user_strategy_request_prompt(
        domain_description=state["domain_description"],
        design_task=state["design_task"],
        guardrails=state["high_level_task_constraints"],
        branch_context=None  #TODO: add support later
    )
    #input = {"messages": {"role": "user", "content": user_prompt}}
    input = {"messages": user_prompt}
    print(f"\n\n{input}\n\n")

    raw_response = await strategy_generator.ainvoke(input)

    output: CreativeStrategy = raw_response["structured_response"]
    print(output.creative_strategy)
    return {"generated_strategy": output.creative_strategy}




def compile_graph():
    builder = StateGraph(CreativeStrategyState)

    builder.add_node("generate_strategy", generate_strategy)
    
    builder.add_edge(START, "generate_strategy")
    builder.add_edge("generate_strategy", END)

    return builder.compile()




# async def run():
#     graph = compile_graph()

#     input = CreativeStrategyState(
#         design_task="a road",
#         domain_description="image generation via an advanced diffusion model",
#         generated_strategy=None,
#         high_level_task_constraints=None,
#         branch_context=None
#     )

#     output = await graph.ainvoke(input)


# if __name__ == "__main__":
#     asyncio.run(run())
