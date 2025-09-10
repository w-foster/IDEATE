from typing import TypedDict, Dict, List, Optional, cast, Literal, Tuple, NotRequired
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
import os
import asyncio
from rich import print 
import base64
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command

from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

from langgraphs.evaluation.pairwise_evaluation_graph import (
    compile_graph as compile_evaluation_graph,
    EvaluationState,
    Evaluation
)
from langgraphs.novelty_checking.novelty_check_graph import (
    compile_graph as compile_novelty_check_graph,
    NoveltyCheckState,
    NoveltyCheck
)
from langgraphs.competitor_identification.competitor_identification_graph import (
    compile_graph as compile_competitor_identification_graph,
    OverallCompetitorIdentificationState
)

from core.branch_context import BranchContext

load_dotenv(find_dotenv())

VLM_MODEL = os.getenv("VLM_MODEL") or "gpt-4o"

"""
1) Is archive full?

2a) No -> is the new image novel enough (should we give it a space)
-- No = not allowed in; Yes = allowed in

2b) Yes -> which archive image is the new image most similar to?
-- E.g. batch check, then final check from top K; 
-- Once found most similar, have them compete; winner gets to stay
"""


class ArchiveAdditionState(TypedDict):
    design_task: str
    domain_description: str

    archive_full: bool

    archive_img_paths: List[str]
    new_img_path: str
    max_comparisons_at_once: int

    novelty_check_result: NotRequired[NoveltyCheck]

    competing_img_paths: NotRequired[Tuple[str, str]]

    evaluation_result: NotRequired[Evaluation]

    img_to_add_path: NotRequired[str]
    img_to_remove_path: NotRequired[str]

    flip_order: bool
    branch_context: Optional[BranchContext]



async def archive_capacity_router(state: ArchiveAdditionState) -> Command[Literal["check_novelty", "find_most_similar"]]:
    if state["archive_full"]:
        return Command(goto="find_most_similar")
    else:
        return Command(goto="check_novelty")


async def check_novelty(state: ArchiveAdditionState) -> Command[Literal["evaluate_pair", "final_node"]]:
    novelty_graph = compile_novelty_check_graph()

    input = NoveltyCheckState(
        design_task=state["design_task"],
        domain_description=state["domain_description"],
        archive_img_paths=state["archive_img_paths"],
        new_img_path=state["new_img_path"],
        branch_context=state.get("branch_context")
    )
    output = await novelty_graph.ainvoke(input)
    output = cast(NoveltyCheckState, output)

    if "is_novel_enough" not in output:
        raise RuntimeError("Novelty result is None")
    if "result" not in output:
        raise RuntimeError("NoveltyCheck result is None")
    
    if output["is_novel_enough"]:
        return Command(
            update={
                "novelty_check_result": output["result"],
                "img_to_add_path": state["new_img_path"],
                "img_to_remove_path": None  
            },
            goto="final_node"
        )
    else:
        if "competing_img_paths" not in output:
            raise RuntimeError("is_novel_enough is False, yet no competing_img_paths is given")
        return Command(
            update={
                "novelty_check_result": output["result"], 
                "competing_img_paths": output["competing_img_paths"]
            },
            goto="evaluate_pair"
        )




async def find_most_similar(state: ArchiveAdditionState) -> Dict:
    find_competitor_graph = compile_competitor_identification_graph()
    input = OverallCompetitorIdentificationState(
        design_task=state["design_task"],
        domain_description=state["domain_description"],
        archive_img_paths=state["archive_img_paths"],
        new_img_path=state["new_img_path"],
        max_comparisons=state["max_comparisons_at_once"]
    )
    output = await find_competitor_graph.ainvoke(input)
    output = cast(OverallCompetitorIdentificationState, output)

    if "most_similar_img_path" not in output:
        raise RuntimeError("Path to most similar image is missing from final output")
    
    most_similar_img_path = output["most_similar_img_path"]
    return {"competing_img_paths": [most_similar_img_path, state["new_img_path"]]}





async def evaluate_pair(state: ArchiveAdditionState) -> Dict:
    if "competing_img_paths" not in state:
        raise RuntimeError("Expected competing_img_paths to be present for evaluation")
    
    evalation_graph = compile_evaluation_graph()
    input = EvaluationState(
        design_task=state["design_task"],
        domain_description=state["domain_description"],
        img_file_names=state["competing_img_paths"],
        flip_order=state["flip_order"],
        branch_context=state.get("branch_context")
    )
    output = await evalation_graph.ainvoke(input)
    full_result = output["result"]
    winner = output["winner_img_path"]
    loser = output["loser_img_path"]

    if not winner:
        raise RuntimeError("Winner path is None after evaluation")
    if not loser:
        raise RuntimeError("Loser path is None after evaluation")
    
    if winner == state["new_img_path"]:
        to_add = winner
        to_remove = loser
    else:
        to_add = None
        to_remove = None

    return {"img_to_add_path": to_add, "img_to_remove_path": to_remove, "evaluation_result": full_result}




async def final_node(state: ArchiveAdditionState) -> Dict:
    # any final stuff. i think none needed rn
    return {}




def compile_graph():
    builder = StateGraph(ArchiveAdditionState)

    builder.add_node("archive_capacity_router", archive_capacity_router)
    builder.add_node("check_novelty", check_novelty)
    builder.add_node("find_most_similar", find_most_similar)
    builder.add_node("evaluate_pair", evaluate_pair)
    builder.add_node("final_node", final_node)

    builder.add_edge(START, "archive_capacity_router")
    builder.add_edge("find_most_similar", "evaluate_pair")
    builder.add_edge("evaluate_pair", "final_node")
    builder.add_edge("final_node", END)


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

    archive_full_input = ArchiveAdditionState(
        design_task="an architectural style that's never been seen before",
        domain_description="image generation via advanced diffusion model, july 2025",
        new_img_path=new_img_file_name,
        archive_img_paths=archive_img_file_names,
        archive_full=True,
        max_comparisons_at_once=5,
        flip_order=False
    )
    archive_full_output = await graph.ainvoke(archive_full_input)
    archive_full_output = cast(ArchiveAdditionState, archive_full_output)
    print(archive_full_output)


if __name__ == "__main__":
    asyncio.run(run())
