from typing import TypedDict, Dict, List, Optional, Union, cast
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
from langchain_openai import ChatOpenAI

from langgraphs.strategising import prompts
from core.solution import Solution
from langgraphs.types import LGModelSpec
from langgraphs.utils import agent_model_name
from new_core.models.image_solution import ImageSolution  # temp just add in and union
from core.branch_context import BranchContext


load_dotenv(find_dotenv())


class RefinedCreativeStrategy(BaseModel):
    refined_creative_strategy: str = Field(
        description="The entire refined creative strategy, in detail",
        default=""
    )

class ArchiveAnalysis(BaseModel):
    archive_analysis: str = Field(
        description="The analysis of the entire current archive",
        default=""
    )


class CreativeStrategyRefinementState(TypedDict):
    model_spec: LGModelSpec
    design_task: str
    domain_description: str

    current_strategy: str
    high_level_guardrails: str
    archive_solutions: Union[List[Solution], List[ImageSolution]]
    num_offspring: int

    archive_analysis: str
    refined_strategy: str
    branch_context: Optional[BranchContext]


# TODO: pull out into its own graph!!!
async def analyse_archive(state: CreativeStrategyRefinementState):
    archive_analyser = ChatOpenAI(model=state["model_spec"]["name"]).with_structured_output(ArchiveAnalysis)

    archive_img_paths = [sol.img_path for sol in state["archive_solutions"]]

    system_msg = AIMessage(content=prompts.create_archive_analysis_system_prompt(
        is_convergence_branch=state.get("branch_context") is not None
    ))
    human_msg = prompts.build_archive_analysis_human_message(
        domain_description=state["domain_description"],
        archive_img_paths=archive_img_paths,
        num_offspring=state["num_offspring"],
        design_task=state["design_task"],
        branch_context=state.get("branch_context")
    )

    print(f"\n\n==== ANALYSING ENTIRE ARCHIVE ====\n\n")
    print(system_msg)

    analysis = await archive_analyser.ainvoke([system_msg, human_msg])
    analysis = cast(ArchiveAnalysis, analysis)
    print(analysis)

    return {"archive_analysis": analysis.archive_analysis}



async def generate_refined_strategy(state: CreativeStrategyRefinementState) -> Dict:
    strategy_refiner = create_react_agent( 
        model=agent_model_name(state["model_spec"]),
        prompt=prompts.create_creative_strategy_refinement_system_prompt(
            is_convergence_branch=state["branch_context"] is not None
        ),
        #tools=[TavilySearch(max_results=10)],
        tools=[],  # for now no tools... so no Tavily during refinement only init creation. Is that best?
        response_format=RefinedCreativeStrategy
    )
    
    user_prompt = prompts.create_user_strategy_refinement_request_prompt(
        domain_description=state["domain_description"],
        design_task=state["design_task"],
        archive_analysis=state["archive_analysis"],
        current_strategy=state["current_strategy"],
        guardrails=state["high_level_guardrails"],
        branch_context=state.get("branch_context")
    )
    input = {"messages": {"role": "user", "content": user_prompt}}
    print(f"\n\n{input}\n\n")

    raw_response = await strategy_refiner.ainvoke(input)
    print(f"\n\n{raw_response}\n\n")

    output: RefinedCreativeStrategy = raw_response["structured_response"]
    return {"refined_strategy": output.refined_creative_strategy}




def compile_graph():
    builder = StateGraph(CreativeStrategyRefinementState)

    builder.add_node("analyse_archive", analyse_archive)
    builder.add_node("generate_refined_strategy", generate_refined_strategy)
    
    builder.add_edge(START, "analyse_archive")
    builder.add_edge("analyse_archive", "generate_refined_strategy")
    builder.add_edge("generate_refined_strategy", END)

    return builder.compile()




# async def run():
#     graph = compile_graph()

#     input = CreativeStrategyRefinementState(
#         design_task="an architectural style that's never been seen before",
#         domain_description="image generation via an advanced diffusion model",
#         current_strategy=""
#     )

#     output = await graph.ainvoke(input)


# if __name__ == "__main__":
#     asyncio.run(run())
