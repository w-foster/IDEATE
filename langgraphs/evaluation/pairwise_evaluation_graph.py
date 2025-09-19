from typing import TypedDict, Dict, List, Optional, cast, Tuple, NotRequired
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
import os
import asyncio
from pathlib import Path
from rich import print 

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent, InjectedState

from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

from langgraphs.evaluation import prompts
from core.branch_context import BranchContext
from langgraphs.types import LGModelSpec


load_dotenv(find_dotenv())



class Evaluation(BaseModel):
    reasoning: str = Field(description="Your reasoning for why you picked the solution")
    best_solution: int = Field(description="The number label of the best solution (1 or 2)")
    confidence_level: int = Field(description="Your confidence level in your decision, 0 not confident at all, 10 maximally confident")


class EvaluationState(TypedDict):
    model_spec: LGModelSpec
    design_task: str 
    domain_description: str

    img_file_names: Tuple[str, str]
    flip_order: bool

    branch_context: Optional[BranchContext]

    result: NotRequired[Evaluation]
    winner_img_path: NotRequired[str]
    loser_img_path: NotRequired[str]



async def evaluate_images(state: EvaluationState):
    evaluator = ChatOpenAI(model=state["model_spec"]["name"]).with_structured_output(Evaluation)

    system_msg = AIMessage(content=prompts.create_evaluation_system_prompt(
        is_convergence_branch=state.get("branch_context") is not None
    ))
    human_msg = prompts.build_evaluation_human_message(
        img_file_names=state["img_file_names"],
        design_task=state["design_task"],
        domain_description=state["domain_description"],
        flip_order=state["flip_order"],
        branch_context=state.get("branch_context")
    )

    raw_response = await evaluator.ainvoke([system_msg, human_msg])

    if isinstance(raw_response, dict):
        evaluation = Evaluation.model_validate(raw_response)
    else:
        # assume it's already the right type; help type checker
        evaluation = cast(Evaluation, raw_response)

    idx_llm_sees = evaluation.best_solution - 1
    if idx_llm_sees not in (0, 1):
        raise ValueError(f"Unexpected best_solution value: {evaluation.best_solution}")
    true_idx = (1 - idx_llm_sees) if state["flip_order"] else idx_llm_sees

    winner_img_path = state["img_file_names"][true_idx]
    loser_img_path = state["img_file_names"][1 - true_idx]

    return {
        "result": evaluation, 
        "winner_img_path": winner_img_path, 
        "loser_img_path": loser_img_path
    }



def compile_graph():
    builder = StateGraph(EvaluationState)

    builder.add_node("evaluate_images", evaluate_images)

    builder.add_edge(START, "evaluate_images")
    builder.add_edge("evaluate_images", END)

    return builder.compile()



# async def run():
#     graph = compile_graph()

#     img_file_names = ("special2.png", "front1.png")

#     input = EvaluationState(
#         design_task="an architectural style that's never been seen before",
#         domain_description=(
#             "image generation with advanced diffusion model "
#             "called FLUX.1 KONTEXT, July 2025 -- your standards "
#             "should be high in terms of image quality"
#         ),
#         img_file_names=img_file_names,
#         flip_order=False
#     )

#     response = await graph.ainvoke(input)
#     print(response)




# if __name__ == "__main__":
#     asyncio.run(run())