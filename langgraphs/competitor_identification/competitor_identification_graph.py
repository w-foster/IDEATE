from typing import TypedDict, Dict, List, Optional, cast, Tuple, NotRequired, Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
import os
import asyncio
from rich import print 
import base64
from pathlib import Path
import math
from operator import add

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Send

from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

from langgraphs.competitor_identification import prompts


load_dotenv(find_dotenv())


VLM_MODEL = os.getenv("VLM_MODEL") or "o3"


class MostSimilar(BaseModel):
    most_similar: int = Field(
        description="The integer label/number of the archive image which the new image is most similar to."
    )


class IdentifyMostSimilarRequest(BaseModel):
    design_task: str
    domain_description: str

    selected_archive_img_paths: List[str]
    new_img_path: str

    most_similar_path: Optional[str] = None


class PartialCompetitorIdentificationState(TypedDict):
    design_task: str
    domain_description: str

    current_candidate_img_paths: List[str]
    new_img_path: str
    max_comparisons: int

    comparison_winner_img_paths: Annotated[List[str], add]


async def begin_comparisons(state: PartialCompetitorIdentificationState) -> Dict:
    return {}


async def split_archive(state: PartialCompetitorIdentificationState):
    if state["max_comparisons"] < 2:
        raise ValueError("max_comparisons cannot be less than 2")
    
    main_list = state["current_candidate_img_paths"]
    n = len(main_list)
    max_capacity = state["max_comparisons"]

    num_chunks = math.ceil(n / max_capacity)
    base, rem = divmod(n, num_chunks)  # some chunks get +1
    paths_subarray = []
    i = 0
    for _ in range(num_chunks):
        size = base + (1 if rem > 0 else 0)
        paths_subarray.append(main_list[i : i + size])
        i += size
        rem -= 1
    
    return [Send("find_most_similar_from_subset", IdentifyMostSimilarRequest(
        design_task=state["design_task"],
        domain_description=state["domain_description"],
        new_img_path=state["new_img_path"],
        selected_archive_img_paths=paths
    )) for paths in paths_subarray]


async def find_most_similar_from_subset(request: IdentifyMostSimilarRequest) -> Dict:
    similar_img_identifier = ChatOpenAI(model=VLM_MODEL).with_structured_output(MostSimilar)

    selected_archive_img_paths = request.selected_archive_img_paths
    system_msg = AIMessage(content=prompts.FIND_MOST_SIMILAR_IMAGE_PROMPT)
    human_msg = prompts.build_find_most_similar_img_human_message(
        new_img_path=request.new_img_path,
        selected_archive_img_paths=selected_archive_img_paths,
        design_task=request.design_task,
        domain_description=request.domain_description
    )

    raw_response = await similar_img_identifier.ainvoke([system_msg, human_msg])
    print(raw_response)

    if isinstance(raw_response, dict):
        most_similar = MostSimilar.model_validate(raw_response)
    else:
        # assume it's already the right type; help type checker
        most_similar = cast(MostSimilar, raw_response)

    most_similar_idx = most_similar.most_similar - 1
    if most_similar_idx < 0 or most_similar_idx >= len(selected_archive_img_paths):
        raise RuntimeError("most_similar_idx is out of bounds! This was calculated from an LLM output.")
    most_similar_path = selected_archive_img_paths[most_similar_idx]

    return {"comparison_winner_img_paths": [most_similar_path]}


async def reduce_comparisons(state: PartialCompetitorIdentificationState) -> Dict:
    return {}


def compile_subgraph():
    builder = StateGraph(PartialCompetitorIdentificationState)

    builder.add_node("begin_comparisons", begin_comparisons)
    builder.add_node("split_archive", split_archive)
    builder.add_node("find_most_similar_from_subset", find_most_similar_from_subset) # type: ignore (silly langgraph doesnt know its own rules)
    builder.add_node("reduce_comparisons", reduce_comparisons)

    builder.add_edge(START, "begin_comparisons")
    builder.add_conditional_edges("begin_comparisons", split_archive, ["find_most_similar_from_subset"])
    builder.add_edge("find_most_similar_from_subset", "reduce_comparisons")
    builder.add_edge("reduce_comparisons", END)

    return builder.compile()


class OverallCompetitorIdentificationState(TypedDict):  # TODO: rename (away from 'overall')
    design_task: str
    domain_description: str

    archive_img_paths: List[str]
    new_img_path: str
    max_comparisons: int

    most_similar_img_path: NotRequired[str]



async def initialise_candidates(state: OverallCompetitorIdentificationState) -> Dict:
    find_similar_images_graph = compile_subgraph()

    initial_input = PartialCompetitorIdentificationState(
        design_task=state["design_task"],
        domain_description=state["domain_description"],
        current_candidate_img_paths=state["archive_img_paths"],
        new_img_path=state["new_img_path"],
        max_comparisons=state["max_comparisons"],
        comparison_winner_img_paths=[]
    )

    output = await find_similar_images_graph.ainvoke(initial_input)
    while len(output["comparison_winner_img_paths"]) > 1:
        new_input = PartialCompetitorIdentificationState(
            design_task=state["design_task"],
            domain_description=state["domain_description"],
            current_candidate_img_paths=output["comparison_winner_img_paths"],
            new_img_path=state["new_img_path"],
            max_comparisons=state["max_comparisons"],
            comparison_winner_img_paths=[]
        )
        output = await find_similar_images_graph.ainvoke(new_input)
    
    if len(output["comparison_winner_img_paths"]) < 1:
        raise RuntimeError("Length of comparison_winner_imgs_paths is less than 1")
    return {"most_similar_img_path": output["comparison_winner_img_paths"][0]}


def compile_graph():
    builder = StateGraph(OverallCompetitorIdentificationState)
    builder.add_node("initialise_candidates", initialise_candidates)
    builder.add_edge(START, "initialise_candidates")
    builder.add_edge("initialise_candidates", END)
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

    input = OverallCompetitorIdentificationState(
        design_task="an architectural style that's never been seen before",
        domain_description="image generation via advanced diffusion model, july 2025",
        archive_img_paths=archive_img_file_names,
        new_img_path=new_img_file_name,
        max_comparisons=2
    )

    raw_response = await graph.ainvoke(input)
    print(raw_response)




if __name__ == "__main__":
    asyncio.run(run())