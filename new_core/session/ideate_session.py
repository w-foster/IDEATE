from typing import List, Optional
from new_core.session import utils
from new_core.interfaces.archive_analyser import IArchiveAnalyser
from new_core.interfaces.archive_store import IArchiveStore
from new_core.interfaces.competitor_identifier import ICompetitorIdentifier
from new_core.interfaces.creative_strategist import ICreativeStrategist
from new_core.interfaces.ideator import IIdeator
from new_core.interfaces.image_generator import IImageGenerator
from new_core.interfaces.parent_selector import IParentSelector
from new_core.interfaces.pairwise_evaluator import IPairwiseEvaluator
from new_core.interfaces.prompt_engineer import IPromptEngineer
from new_core.interfaces.run_repository import IRunRepository
from new_core.models.image_solution import ImageSolution
from new_core.models.run_config import RunConfig
from new_core.models.run_state import RunState
from new_core.models.task_context import TaskContext


class IDEATESession:
    def __init__(
        self,
        run_config: RunConfig,
        task_context: TaskContext,
        archive: IArchiveStore,
        strategist: ICreativeStrategist,
        archive_analyser: IArchiveAnalyser,
        constraints_generator: 
        ideator: IIdeator,
        prompt_engineer: IPromptEngineer,
        image_generator: IImageGenerator,
        competitor_identifier: ICompetitorIdentifier,
        pairwise_evaluator: IPairwiseEvaluator,
        parent_selector: IParentSelector,
        repo: IRunRepository,
        results_root: str,
        run_name: Optional[str] = None,
    ):
        # injected dependencies
        self.archive = archive
        self.strategist = strategist
        self.ideator = ideator
        self.prompt_engineer = prompt_engineer
        self.image_generator = image_generator
        self.archive_analyser = archive_analyser
        self.competitor_identifier = competitor_identifier
        self.pairwise_evaluator = pairwise_evaluator
        self.parent_selector = parent_selector
        self.repo = repo

        # config / context / state
        self._cfg: RunConfig = run_config
        self._ctx: TaskContext = task_context
        self._state: RunState = RunState()  

        # repository initialisation (for persistence)
        self.run_dir: str = utils.make_run_dir(results_root, run_name)
        self.repo.init_layout(self.run_dir)
        self.repo.save_config(self.run_dir, self._cfg.model_dump())


    
    
    async def _ensure_strategy(self):
        if self._state.current_strategy_id is None:
            new_strategy = await self.strategist.generate_strategy_from_task(self._ctx)
            strat_id = self._state.push_strategy_and_activate(new_strategy)
            # persist
            self.repo.append_strategy_version(
                self.run_dir, strat_id, new_strategy.text, previous_archive_analysis=None
            )

    async def _maybe_refine_strategy(self):
        if self._state.solutions_generated > 0 and \
            self._state.solutions_generated % self._cfg.strategy_refinement_interval == 0:

            active_strategy = self._state.get_active_strategy()
            feedback = await self.archive_analyser.generate_feedback(self._cfg, self.archive, active_strategy)
            refined_strategy = await self.strategist.refine_existing_strategy(self._ctx, active_strategy, feedback)
            strat_id = self._state.push_strategy_and_activate(refined_strategy)
            # persist
            self.repo.append_strategy_version(
                self.run_dir, strat_id, refined_strategy.text, previous_archive_analysis=feedback.text
            )

    async def _run_once(self, specify_no_parents: bool = False):
        await self._ensure_strategy()
        await self._maybe_refine_strategy()

        # ==== SELECTION ====
        if specify_no_parents:
            parent_solutions = None
        else:
            parent_solutions = await self.parent_selector.select(
                archive=self.archive,
                num_parents=self._cfg.parents_per_ideation
            )
            

        # ==== IDEATION ====
        new_idea = await self.ideator.ideate(
            self._ctx, 
            parent_solutions, 
            self._state.get_active_strategy(), 
            self.archive
        )
        new_prompt = await self.prompt_engineer.idea_to_prompt(self._ctx, new_idea)

        sample_url, gen_metadata = await self.image_generator.generate(new_prompt)
        sol_num = self._state.solutions_generated + 1
        sol_id = self._build_solution_id(sol_num)
        # persist
        self.repo.save_prompt_text(self.run_dir, sol_id, new_prompt.text)
        img_path = self.repo.save_image_from_url(self.run_dir, sol_id, sample_url)

        new_solution = ImageSolution(
            id=sol_id,
            prompt=new_prompt,
            idea=new_idea,
            img_path=img_path,
            metadata=gen_metadata or {},
        )

        # ==== ARCHIVE ADDITION ====
        competitor: Optional[ImageSolution] = await self.competitor_identifier.identify_competitor_or_none(
            new_solution=new_solution,
            archive=self.archive,
        )

        admitted = False
        if competitor is None:
            self.archive.add(new_solution)
            admitted = True
        else:
            winner: ImageSolution = await self.pairwise_evaluator.choose_winner(self._ctx, new_solution, competitor)
            if winner != new_solution and winner != competitor:
                raise RuntimeError("winner is not one of the two competing solutions")
            if winner == new_solution:
                self.archive.remove(competitor.id)
                self.archive.add(new_solution)
                admitted = True
        

        # persist
        self.repo.write_solution_markdown(
            self.run_dir,
            new_solution,
            sol_num,
            design_task=self._ctx.design_task,
            domain_description=self._ctx.domain_description,
            model_used=new_solution.metadata.get("model", "UNKNOWN"),
            added_to_archive=admitted,
            addition_workflow_output=None, 
        )

        self.repo.write_solution_metadata_json(
            self.run_dir,
            new_solution,
            sol_num,
            design_task=self._ctx.design_task,
            domain_description=self._ctx.domain_description,
            model_used=new_solution.metadata.get("model", "UNKNOWN"),
            added_to_archive=admitted,
            strategy_version=self._state.current_strategy_id or "None",
            generation_params=new_solution.metadata.get("result"),
        )

        self._state.solutions_generated += 1


    async def _initial_ideation(self):
        for _ in range(self._cfg.initial_ideation_count):
            await self._run_once(specify_no_parents=True)


    async def run(self) -> IArchiveStore:
        await self._initial_ideation()

        for gen in range(self._cfg.generations):
            await self._run_once()
            self._state.current_generation += 1
        return self.archive

    

    def _build_solution_id(self, solution_number) -> str:
        return f"sol_{solution_number:03d}"


