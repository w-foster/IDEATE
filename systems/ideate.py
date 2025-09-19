import asyncio
import os
from dotenv import load_dotenv, find_dotenv

from new_core.embedders.clip_image_embedder import CLIPImageEmbedder
from new_core.interfaces import image_embedder, prompt_engineer
from new_core.interfaces.archive_analyser import IArchiveAnalyser
from new_core.interfaces.archive_store import IArchiveStore
from new_core.interfaces.constraints_generator import IConstraintsGenerator
from new_core.interfaces.creative_strategist import ICreativeStrategist
from new_core.interfaces.high_level_stages.archive_addition_policy import IArchiveAdditionPolicy
from new_core.interfaces.high_level_stages.solution_generator import ISolutionGenerator
from new_core.interfaces.ideator import IIdeator
from new_core.interfaces.image_generator import IImageGenerator
from new_core.interfaces.parent_selector import IParentSelector
from new_core.interfaces.prompt_engineer import IPromptEngineer
from new_core.interfaces.run_repository import IRunRepository
from new_core.models.ai_model_spec import AIModelSpec
from new_core.models.run_config import RunConfig
from new_core.models.task_constraints import TaskConstraints
from new_core.models.task_context import TaskContext
from new_core.selectors.random_selector import RandomSelector
from new_core.session.creative_session import CreativeSession
from new_core.archives.in_memory_image_archive_store import InMemoryImageArchiveStore
from new_core.composite_phases.two_stage_solution_generation import TwoStageSolutionGeneration
from new_core.langgraph_components.agents.lg_ideator import LGIdeator
from new_core.langgraph_components.agents.lg_competitor_identifier import LGCompetitorIdentifier
from new_core.langgraph_components.agents.lg_constraints_generator import LGConstraintsGenerator
from new_core.langgraph_components.agents.lg_pairwise_evaluator import LGPairwiseEvaluator
from new_core.langgraph_components.agents.lg_prompt_engineer import LGPromptEngineer
from new_core.langgraph_components.agents.lg_research_creative_strategist import LGResearchCreativeStrategist
from new_core.langgraph_components.multi_agent.lg_mono_archive_addition_policy import LGMonoArchiveAdditionPolicy
from new_core.adapters.flux_adapter import FluxAdapter
from new_core.session.fs_run_repository import FSRunRepository
from new_core.mock.mock_archive_analyser import MockArchiveAnalyser
from new_core.interfaces.image_embedder import IImageEmbedder
from new_core.embedders.clip_image_embedder import CLIPImageEmbedder

load_dotenv(find_dotenv())



# TODO: think of way to specify this on a per component basis...
dummy_model_spec = AIModelSpec(
    name="o3",
    provider="openai"
)

dummy_flux_model = "flux-kontext-max"


class IDEATESession:
    def __init__(self, run_config: RunConfig, task_context: TaskContext):
        # Strategising
        strategist: ICreativeStrategist = LGResearchCreativeStrategist(
            dummy_model_spec, 
            run_config.strategy_refinement_interval
        )
        constraints_generator: IConstraintsGenerator = LGConstraintsGenerator(dummy_model_spec)
        mock_archive_analyser: IArchiveAnalyser = MockArchiveAnalyser()

        # Solution generation
        ideator: IIdeator = LGIdeator(dummy_model_spec)
        prompt_engineer: IPromptEngineer = LGPromptEngineer(dummy_model_spec)
        image_generator: IImageGenerator = FluxAdapter(model=dummy_flux_model, use_raw_mode=False)

        solution_generator: ISolutionGenerator = TwoStageSolutionGeneration(
            ideator=ideator,
            prompt_engineer=prompt_engineer,
            image_generator=image_generator
        )

        # Archive addition (temp; TODO: extract archive analyser then make generic composite phase class)
        archive_addition_policy: IArchiveAdditionPolicy = LGMonoArchiveAdditionPolicy(dummy_model_spec)

        # Archive storage and persistence
        archive: IArchiveStore = InMemoryImageArchiveStore(run_config.max_archive_capacity)
        parent_selector: IParentSelector = RandomSelector()
        repo: IRunRepository = FSRunRepository()

        # Image embedder
        image_embedder: IImageEmbedder = CLIPImageEmbedder()

        RESULTS_ROOT_ABSOLUTE = os.getenv("RESULTS_ROOT_ABSOLUTE")
        if RESULTS_ROOT_ABSOLUTE is None:
            raise RuntimeError("environment variable 'RESULTS_ROOT_ABSOLUTE' is not set")

        self._creative_session = CreativeSession(
            run_config=run_config,
            task_context=task_context,
            archive=archive,
            parent_selector=parent_selector,
            solution_generator=solution_generator,
            archive_addition_policy=archive_addition_policy,
            strategist=strategist,
            constraints_generator=constraints_generator,
            repo=repo,
            archive_analyser=mock_archive_analyser,
            results_root=RESULTS_ROOT_ABSOLUTE,
            image_embedder=image_embedder
        )
    

    async def run(self) -> IArchiveStore:
        return await self._creative_session.run()

    

# ===================================
async def test_ideate_session():
    cfg = RunConfig(
        generations=17,
        strategy_refinement_interval=10,
        max_archive_capacity=25,
        max_solution_comparisons_per_call=5,
        offspring_per_generation=1,
        parents_per_ideation=2,
        initial_ideation_count=5,
        randomise_order_for_llm=True,
        k_for_internal_novelty=3
    )

    ctx = TaskContext(
        design_task="a hyper-realistic photo of a cow in a Cotswolds field",
        domain_description="Text-to-image generation via advanced diffusion model (September 2025)"
    )

    IDEATE = IDEATESession(cfg, ctx)
    await IDEATE.run()


if __name__ == "__main__":
    asyncio.run(test_ideate_session())

    