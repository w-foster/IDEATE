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

# remove this dependency eventually
from core.branch_context import BranchContext
from langgraphs.ideation import prompts

load_dotenv(find_dotenv())

MODEL = os.getenv("LLM_MODEL") or "grok-3-mini"



class NewIdea(BaseModel):
    idea: str = Field(description="The one new idea you generated (just the idea)", default="")

class IdeationState(TypedDict):
    design_task: str
    domain_descripton: str

    creative_strategy: str
    archive_ideas_except_seeds: List[str]  # EXCLUDING the seed ideas

    seed_ideas: List[str]
    new_idea: Optional[str]

    branch_context: Optional[BranchContext]



async def generate_idea(state: IdeationState) -> Dict:
    ideator = create_react_agent(
        model=MODEL,
        tools=[],
        prompt=prompts.create_ideation_system_prompt(is_convergence_branch=state["branch_context"] is not None),
        response_format=NewIdea
    )

    user_prompt: HumanMessage = prompts.create_user_ideation_prompt(
        creative_strategy=state["creative_strategy"],
        design_task=state["design_task"],
        domain_description=state["domain_descripton"],
        seed_ideas=state["seed_ideas"],
        archive_ideas_except_seeds=state["archive_ideas_except_seeds"],
        branch_context=state["branch_context"]
    )
    #input = {"messages": {"role": "user", "content": user_prompt}}
    input = {"messages": user_prompt}

    print(f"\n\n==== Generating new IDEA. Prompt: ==== \n{user_prompt}\n")

    raw_response = await ideator.ainvoke(input)
    output: NewIdea = raw_response["structured_response"]
    print(f"\n\n==== NEW IDEA: ====\n{output.idea}\n")

    return {"new_idea": output.idea}



def compile_graph():
    builder = StateGraph(IdeationState)

    builder.add_node("generate_idea", generate_idea)

    builder.add_edge(START, "generate_idea")
    builder.add_edge("generate_idea", END)

    return builder.compile()


# async def run():
#     graph = compile_graph()

#     input = IdeationState(
#         design_task=
#     )

#     response = await graph.ainvoke(input)
#     print(response)


