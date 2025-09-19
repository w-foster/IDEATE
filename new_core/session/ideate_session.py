from threading import local
from typing import Dict, List, Optional
from new_core.interfaces.constraints_generator import IConstraintsGenerator
from new_core.interfaces.high_level import archive_addition_policy
from new_core.interfaces.high_level.archive_addition_policy import IArchiveAdditionPolicy
from new_core.interfaces.high_level.solution_generator import ISolutionGenerator
from new_core.models.archive_addition_decision import ArchiveAdditionDecision
from new_core.models.solution_generation_result import SolutionGenerationResult
from new_core.models.task_constraints import TaskConstraints
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
        # Core data/configs
        run_config: RunConfig,
        task_context: TaskContext,

        # Archive storage & selection
        archive: IArchiveStore,
        parent_selector: IParentSelector,
        
        # High-level phases
        solution_generator: ISolutionGenerator,
        archive_addition_policy: IArchiveAdditionPolicy,

        # Granular steps/agents
        strategist: ICreativeStrategist,
        archive_analyser: IArchiveAnalyser,
        constraints_generator: IConstraintsGenerator,
        competitor_identifier: ICompetitorIdentifier,
        pairwise_evaluator: IPairwiseEvaluator,

        # Persistence
        repo: IRunRepository,
        results_root: str,
        run_name: Optional[str] = None,
    ):
        # injected dependencies
        self._archive = archive
        self._parent_selector = parent_selector
        self._solution_generator = solution_generator
        self._archive_addition_policy = archive_addition_policy
        self._strategist = strategist
        self._archive_analyser = archive_analyser
        self._competitor_identifier = competitor_identifier
        self._pairwise_evaluator = pairwise_evaluator
        self._constraints_generator = constraints_generator
        
        # config / context / state
        self._cfg: RunConfig = run_config
        self._ctx: TaskContext = task_context
        self._state: RunState = RunState()  

        # repository initialisation & persistence
        self._repo = repo
        self._run_dir: str = utils.make_run_dir(results_root, run_name)
        self._repo.init_layout(self._run_dir)
        self._repo.save_config(self._run_dir, self._cfg.model_dump())
    
    
    async def _ensure_strategy(self):
        if self._state.get_active_strategy_id() is not None:
            return
        
        if self._state.get_task_constraints() is None:
            new_constraints = await self._constraints_generator.generate_constraints_for_task(self._ctx)
            self._state.set_task_constraints(new_constraints)
        
        new_strategy = await self._strategist.generate_strategy_from_task(
            task_context=self._ctx,
            task_constraints=new_constraints
        )
        strat_id = self._state.push_strategy_and_activate(new_strategy)

    async def _maybe_refine_strategy(self):
        num_sols_generated = self._state.get_num_solutions_generated()
        if num_sols_generated > 0 and \
            num_sols_generated % self._cfg.strategy_refinement_interval == 0:

            active_strategy = self._state.get_active_strategy_or_throw()

            new_feedback = await self._archive_analyser.generate_feedback(
                current_strategy=active_strategy,
                run_config=self._cfg,
                archive=self._archive
            )

            constraints = self._state.get_task_constraints()
            if constraints is None:
                raise RuntimeError("'task_constraints' in RunState is none at time of strategy refinement")

            refined_strategy = await self._strategist.refine_existing_strategy(
                task_context=self._ctx,
                task_constraints=constraints,
                strategy=active_strategy,
                feedback=new_feedback
            )
            strat_id = self._state.push_strategy_and_activate(refined_strategy)

    async def _run_once(self, specify_no_parents: bool = False):
        # TODO: consider adding an IStrategising phase; 
        # cld offer constraints gen, strategy gen (pass in constraints), strategy refine (pass in feedback)
        await self._ensure_strategy()
        await self._maybe_refine_strategy()

        # ==== SELECTION ====
        if specify_no_parents:
            parent_solutions = None
        else:
            parent_solutions = await self._parent_selector.select(
                archive=self._archive,
                num_parents=self._cfg.parents_per_ideation
            )
            
        # ==== IDEATION / SOLUTION GENERATION ====
        gen_result: SolutionGenerationResult = await self._solution_generator.generate(
            task_context=self._ctx,
            parent_solutions=parent_solutions,
            strategy=self._state.get_active_strategy_or_throw(),
            archive=self._archive
        )
        new_solution = self._package_new_solution(gen_result)

        # ==== ARCHIVE ADDITION POLICY & ARCHIVE UPDATE ====
        addition_decision: ArchiveAdditionDecision = await self._archive_addition_policy.decide(
            task_context=self._ctx,
            run_config=self._cfg,
            archive=self._archive,
            new_solution=new_solution,
            randomise_order_for_llm=self._cfg.randomise_order_for_llm
        )
        new_sol_added = self._archive.apply_decision(new_solution=new_solution, decision=addition_decision)

        # ==== PERSISTENCE & RUNSTATE UPDATES ====
        self._persist_new_solution(
            new_solution=new_solution,
            addition_decision=addition_decision,
            gen_metadata=gen_result.gen_metadata
        )

        self._state.increment_num_solutions_generated()
        self._state.increment_current_generation_num()

        self._persist_population()


    async def _initial_ideation(self):
        for _ in range(self._cfg.initial_ideation_count):
            await self._run_once(specify_no_parents=True)


    async def run(self) -> IArchiveStore:
        await self._initial_ideation()

        for gen in range(self._cfg.generations):
            await self._run_once()
        return self._archive

    
    def _persist_new_solution(
        self, 
        new_solution: ImageSolution, 
        addition_decision: ArchiveAdditionDecision,
        gen_metadata: Dict
    ) -> None:
        if new_solution.prompt and getattr(new_solution.prompt, "text", None):
            self._repo.save_prompt_text(self._run_dir, new_solution.id, new_solution.prompt.text)
        
        self._repo.write_solution_markdown(
            run_dir=self._run_dir,
            solution=new_solution,
            solution_count=new_solution.sol_num,  #TODO: remove now that its in the sol
            design_task=self._ctx.design_task,
            domain_description=self._ctx.domain_description,
            model_used=new_solution.metadata.get('model', 'UNKNOWN'), #TODO: check if this is right?
            addition_decision_reasoning=addition_decision.decision_reasoning,
            added_to_archive=addition_decision.add_new_solution
        )

        self._repo.write_solution_metadata_json(
            run_dir=self._run_dir,
            solution=new_solution,
            solution_count=new_solution.sol_num,   #TODO: remove now that its in the sol
            design_task=self._ctx.design_task,
            domain_description=self._ctx.domain_description,
            model_used=new_solution.metadata.get('model', 'UNKNOWN'), #TODO: check if this is right?
            addition_decision_reasoning=addition_decision.decision_reasoning,
            added_to_archive=addition_decision.add_new_solution,
            strategy_version=self._state.get_active_strategy_id() or "None",
            generation_params=gen_metadata
        )
    
    def _persist_population(self) -> None:
        self._repo.append_population(
            self._run_dir,
            self._state.get_current_generation_num(),
            [s.id for s in self._archive.all()],
        )


    def _package_new_solution(self, gen_result: SolutionGenerationResult) -> ImageSolution:
        sol_num = self._state.get_next_solution_num()
        sol_id = self._build_solution_id(sol_num)
        local_img_path = self._repo.save_image_from_url(
            run_dir=self._run_dir, 
            sol_id=sol_id, 
            sample_url=gen_result.sample_path
        )
        return ImageSolution(
            id=sol_id,
            sol_num=sol_num,
            idea=gen_result.idea,
            prompt=gen_result.prompt,
            img_path=local_img_path,
            metadata=gen_result.gen_metadata or {}
        )


    def _build_solution_id(self, solution_number) -> str:
        return f"sol_{solution_number:03d}"


