from typing import TypedDict, Dict, List, Optional, cast, Tuple, NotRequired
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
import os
import asyncio
from rich import print 
import base64
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent, InjectedState

from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

from langgraphs.novelty_checking import prompts
from core.branch_context import BranchContext


load_dotenv(find_dotenv())

VLM_MODEL = os.getenv("VLM_MODEL") or "gpt-4o"


class NoveltyCheck(BaseModel):
    is_novel_enough: bool = Field(
        description="true/false, depending on whether or not the image is sufficiently creative/novel/interesting relative to the existing archive"
    )

    too_similar_to: Optional[int] = Field(
        description="This should either be None if the new image is sufficiently novel, OR, if not sufficiently novel, then the integer of the archive image which it is MOST SIMILAR to.",
        default=None
    )


class NoveltyCheckState(TypedDict):
    design_task: str
    domain_description: str

    archive_img_paths: List[str]
    new_img_path: str
    branch_context: Optional[BranchContext]

    result: NotRequired[NoveltyCheck]
    is_novel_enough: NotRequired[bool]
    competing_img_paths: NotRequired[Tuple[str, str]]

    


async def check_novelty(state: NoveltyCheckState):
    novelty_checker = ChatOpenAI(model=VLM_MODEL).with_structured_output(NoveltyCheck)

    system_msg = AIMessage(content=prompts.create_novelty_system_prompt(
        is_convergence_branch=state.get("branch_context") is not None
    ))
    human_msg = prompts.build_novelty_human_message(
        new_img_path=state["new_img_path"], 
        archive_img_paths=state["archive_img_paths"],
        design_task=state["design_task"],
        branch_context=state.get("branch_context")
    )

    print(f"\n\n==== CHECKING NOVELTY ====\n\n")

    raw_response = await novelty_checker.ainvoke([system_msg, human_msg])
    print(raw_response)

    if isinstance(raw_response, dict):
        novelty_check = NoveltyCheck.model_validate(raw_response)
    else:
        # assume it's already the right type; help type checker
        novelty_check = cast(NoveltyCheck, raw_response)

    novel = novelty_check.is_novel_enough
    partial_update = {"result": novelty_check, "is_novel_enough": novel}
    if novel:
        return partial_update
    elif not novelty_check.too_similar_to:
        raise RuntimeError("Image was determined NOT novel enough, yet no competitor image was given")
    else:
        competitor_idx = novelty_check.too_similar_to - 1
        competitor_img_path = state["archive_img_paths"][competitor_idx]
        paths = (state["new_img_path"], competitor_img_path)
        return partial_update | {"competing_img_paths": paths}
    



def compile_graph():
    builder = StateGraph(NoveltyCheckState)

    builder.add_node("check_novelty", check_novelty)

    builder.add_edge(START, "check_novelty")
    builder.add_edge("check_novelty", END)

    return builder.compile()



async def run():
    graph = compile_graph()

    new_img_file_name = "special2.png"
    archive_img_file_names = [
        "front2.png",
        "diff.png",
        "special1.png",
        "front1.png"
    ]

    input = NoveltyCheckState(
        design_task="an architectural style that's never been seen before",
        domain_description="image gen via diffusion models, july 2025",
        new_img_path=new_img_file_name,
        archive_img_paths=archive_img_file_names,
    )

    raw_response = await graph.ainvoke(input)
    print(raw_response)




if __name__ == "__main__":
    asyncio.run(run())








