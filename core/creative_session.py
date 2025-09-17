from typing import List, Optional, Dict, Any, cast
import asyncio
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import json
import traceback
from dotenv import load_dotenv, find_dotenv
import os

from core.archive import Archive
from core.image_generator import (
    FluxAPIAdapter as FluxImageGenerator,
    FluxModel
)
from core.solution import Solution
from core.utils import create_solution_markdown, init_run_dirs, save_solution_metadata_json
from core.image_embedder import ImageEmbedder

from langgraphs.strategising.creative_strategy_graph import compile_graph as compile_creative_strategy_graph
from langgraphs.ideation.ideation_graph import compile_graph as compile_ideation_graph
from langgraphs.blueprint_engineering.blueprint_engineering_graph import compile_graph as compile_blueprint_engineering_graph
from langgraphs.strategising.refine_creative_strategy_graph import compile_graph as compile_refine_creative_strategy_graph

from langgraphs.ideation.ideation_graph import IdeationState, NewIdea
from langgraphs.strategising.creative_strategy_graph import CreativeStrategyState
from langgraphs.strategising.refine_creative_strategy_graph import CreativeStrategyRefinementState
from langgraphs.blueprint_engineering.blueprint_engineering_graph import BlueprintState
from langgraphs.archive_addition.archive_addition_graph import ArchiveAdditionState

from core.branch_context import BranchContext

load_dotenv(find_dotenv())


class CreativeSession:
    def __init__(
        self,
        design_task: str,
        domain_description: str,
        genotpye_guidance: str,
        bfl_model: FluxModel,
        use_raw_mode: bool = True,
        initial_seeds: Optional[List[str]] = None,
        using_fixed_strategy: bool = False,
        max_archive_capacity: int = 20,
        check_novelty_before_full: bool = True,
        max_comparisons_at_once: int = 5,
        creative_strategy_refinement_interval: int = 8,
        initial_ideation_count: int = 5,
        offspring_per_gen: int = 1,
        run_id: Optional[str] = None,
        results_dir: Optional[str] = None,
        branch_context: Optional[BranchContext] = None,
    ):
        self.using_fixed_strategy = using_fixed_strategy

        self.design_task = design_task
        self.domain_description = domain_description
        self.blueprint_guidance = genotpye_guidance
        self.creative_strategy: str = ""

        # for convergence vs divergence (set before Archive so it's available during construction)
        self.is_convergent_branch = branch_context is not None
        self.branch_context: Optional[BranchContext] = branch_context

        self.archive = Archive(
            design_task=self.design_task,
            domain_description=self.domain_description,
            max_capacity=max_archive_capacity,
            check_novelty_before_full=check_novelty_before_full,
            max_comparisons_at_once=max_comparisons_at_once,
            branch_context=self.branch_context
        )
        self._resumed = False
        self.initial_seeds = initial_seeds or [design_task] * 2
        self.completed_first_ideations = False

        self.bfl_model = bfl_model
        self.use_raw_mode = use_raw_mode
        self.generator = FluxImageGenerator(use_raw_mode)

        self.image_embedder = ImageEmbedder()

        # Human feedback not implemented yet
        # self.feedback = FeedbackHandler()

        # Use provided run_id/results_dir or fall back to timestamp
        if run_id is not None and results_dir is not None:
            self.run_id = run_id
            self.results_dir = results_dir
        else:
            self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.results_dir = f"results/{self.run_id}"

        # initialize folders
        self.solution_count = 0
        self.paths = init_run_dirs(self.results_dir)
        self.generator.local_save_dir = Path(self.results_dir) / "artifacts" / "images"
        self.generator.local_save_dir.mkdir(parents=True, exist_ok=True)

        self.creative_strategy_refinement_interval = creative_strategy_refinement_interval
        self.initial_ideation_count = initial_ideation_count
        self.offspring_per_gen = offspring_per_gen

        self.creative_strategy_graph = compile_creative_strategy_graph()
        self.ideation_graph  = compile_ideation_graph()
        self.blueprint_engineering_graph = compile_blueprint_engineering_graph()
        self.creative_strategy_refinement_graph = compile_refine_creative_strategy_graph()

        self.strategy_version_counter = 0
        self.strategy_versions: Dict[str, str] = {}
        self.current_strategy_version: Optional[str] = None

        if self.branch_context is not None:
            (Path(self.results_dir) / "branch.json").write_text(json.dumps({
                "parent_run_dir": self.branch_context.parent_run_dir,
                "reference_img_paths": self.branch_context.reference_img_paths,
                "new_design_task": self.branch_context.new_design_task,
                "branch_depth": self.branch_context.branch_depth,
                "prior_reference_img_paths": getattr(self.branch_context, "prior_reference_img_paths", []),
                "prior_branch_texts": getattr(self.branch_context, "prior_branch_texts", []),
                "timestamp": datetime.now().isoformat()
            }, indent=2))



    async def initialise(self):
        if self.using_fixed_strategy:
            from core.fixed_strategy import STRATEGY
            self.creative_strategy = STRATEGY
        else:
            input = CreativeStrategyState(
                design_task=self.design_task,
                domain_description=self.domain_description,
                generated_strategy=None,
                branch_context=self.branch_context,
                high_level_guardrails=None
            )
            raw_response = await self.creative_strategy_graph.ainvoke(input)
            self.creative_strategy = raw_response["generated_strategy"]
            self.high_level_guardrails = raw_response["high_level_guardrails"]

            self._save_guardrails(self.high_level_guardrails)

        self.previous_archive_analysis = "No strategy has been applied yet."
    

    def _update_strategy(self):
        self.strategy_version_counter += 1
        strategy_id = f"strategy_{self.strategy_version_counter:03d}"
        self.strategy_versions[strategy_id] = self.creative_strategy
        self.current_strategy_version = strategy_id

        entry = {
            "strategy_id": strategy_id,
            "strategy": self.creative_strategy,
            "timestamp": datetime.now().isoformat(),
            "previous_archive_analysis": self.previous_archive_analysis
        }
        with open(Path(self.results_dir) / "strategy_versions.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _save_guardrails(self, guardrails: str):
        (Path(self.results_dir) / "guardrails.json").write_text(json.dumps({
            "design_task": self.design_task,
            "domain_description": self.domain_description,
            "guardrails": self.high_level_guardrails,
        }, indent=2))

    async def bootstrap(self):
        """Generate or load initial images for the first generation."""
        print(f"\nINITIAL SEEDS: {self.initial_seeds}\n")
        for prompt in self.initial_seeds:
            result = self.generator.generate_image(prompt=prompt, model=self.bfl_model, aspect_ratio="16:9")


    async def divergent_step(self):
        if not hasattr(self, 'run_id'):
            raise Exception("Call run() before calling divergent_step()")

        num_to_generate = self.offspring_per_gen if self.completed_first_ideations else self.initial_ideation_count
        for _ in range(num_to_generate):
            if self.solution_count > 0 and \
               self.solution_count % self.creative_strategy_refinement_interval == 0:
                await self._refine_creative_strategy_step()

            self.solution_count += 1
            sol_id = f"{self.run_id}_sol_{self.solution_count:03d}"

            if self.branch_context is not None:
                seed_ideas = ["No seeds for this design task yet"]
            else:
                if not self.completed_first_ideations:
                    seed_ideas = [self.design_task] * 2
                else:
                    sols = self.archive.get_all_solutions()
                    seed_ideas = [sol.idea for sol in (random.sample(sols, 2) if len(sols)>=2 else sols)]

            archive_ideas_except_seeds = [
                sol.idea for sol in self.archive.get_all_solutions()
                if sol.idea != self.design_task
            ]

            ideation_input = IdeationState(
                design_task=self.design_task,
                domain_descripton=self.domain_description,
                creative_strategy=self.creative_strategy,
                seed_ideas=seed_ideas,
                new_idea=None,
                archive_ideas_except_seeds=archive_ideas_except_seeds,
                branch_context=self.branch_context
            )
            raw_response = await self.ideation_graph.ainvoke(ideation_input)
            new_idea = raw_response["new_idea"]

            blueprint_engineering_input = BlueprintState(
                design_task=self.design_task,
                domain_description=self.domain_description,
                guidance=self.blueprint_guidance,
                idea=new_idea,
                blueprint=None,
                branch_context=self.branch_context
            )
            raw_response = await self.blueprint_engineering_graph.ainvoke(blueprint_engineering_input)
            new_prompt = raw_response["blueprint"]

            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda: self.generator.generate_image(
                            filename=f"{sol_id}.png",
                            prompt=new_prompt,
                            model=self.bfl_model,
                            aspect_ratio="16:9",
                            poll_timeout=300
                        ),
                    ),
                    timeout=300
                )
            except Exception as e:
                print(f"Skipped prompt '{new_prompt}' due to {type(e).__name__}: {e}")
                continue

            img_path = result["local_path"] or result["image_url"]

            new_sol = Solution(
                id=sol_id,
                prompt=new_prompt,
                idea=new_idea,
                img_path=img_path,
                metadata={"creative_strategy_name": self.current_strategy_version}
            )
            flip_order = self.solution_count % 2 == 0
            added, addition_workflow_output = await self.archive.add(new_sol, flip_order)

            with open(self.paths["prompts"] / f"{new_sol.id}.txt", "w") as f:
                f.write(new_prompt)

            try:
                embedding_tensor = self.image_embedder.embed(img_path)
                embedding_np = embedding_tensor.cpu().numpy()
                np.save(self.paths["embeddings"] / f"{new_sol.id}.npy", embedding_np)
            except Exception as e:
                print(f"Warning: failed embedding for {new_sol.id}: {e}")

            await self._save_solution_info(
                solution=new_sol,
                image_result=result,
                added_to_archive=added,
                addition_workflow_output=addition_workflow_output
            )

            print(">>> GENERATED :)")

        self.completed_first_ideations = True


    async def _save_solution_info(
        self,
        solution: Solution,
        image_result: Dict[str, Any],
        added_to_archive: bool,
        addition_workflow_output: Optional[ArchiveAdditionState]
    ):
        create_solution_markdown(
            solution=solution,
            run_id=self.run_id,
            solution_count=self.solution_count,
            design_task=self.design_task,
            domain_description=self.domain_description,
            model_used=image_result.get("model", "UNKNOWN"),
            added_to_archive=added_to_archive,
            generation_params=image_result.get("metadata", {}),
            request_id=image_result.get("request_id", "UNKNOWN"),
            addition_workflow_output=addition_workflow_output
        )

        save_solution_metadata_json(
            solution=solution,
            run_id=self.run_id,
            solution_count=self.solution_count,
            design_task=self.design_task,
            domain_description=self.domain_description,
            model_used=image_result.get("model", "UNKNOWN"),
            added_to_archive=added_to_archive,
            generation_params=image_result.get("metadata", {}),
            request_id=image_result.get("request_id", "UNKNOWN"),
            strategy_version=self.current_strategy_version
        )


    async def _refine_creative_strategy_step(self):
        input = CreativeStrategyRefinementState(
            design_task=self.design_task,
            domain_description=self.domain_description,
            current_strategy=self.creative_strategy,
            archive_solutions=self.archive.get_all_solutions(),
            num_offspring=self.creative_strategy_refinement_interval,
            archive_analysis="",
            refined_strategy="",
            high_level_guardrails=self.high_level_guardrails,
            branch_context=self.branch_context
        )
        output = cast(CreativeStrategyRefinementState, await self.creative_strategy_refinement_graph.ainvoke(input))
        self.creative_strategy = output["refined_strategy"]
        self.previous_archive_analysis = output["archive_analysis"]
        self._update_strategy()


    async def human_feedback_step(self):
        pass


    async def run(self, generations: int = 50):
        """Orchestrate full creative session."""
        if self.offspring_per_gen != 1:
            raise ValueError("Currently, only offspring_per_gen of 1 is supported")

        # Determine where to start numbering generations:
        pop_file = Path(self.results_dir) / "population_data.jsonl"
        last_gen = 0
        if pop_file.exists():
            with pop_file.open() as f:
                for line in f:  
                    entry = json.loads(line)
                    last_gen = entry.get("generation", last_gen)
        start_gen = last_gen + 1
        end_gen = generations

        # Only write a new strategy entry if this is a fresh run, not a resume
        if getattr(self, "_resumed", False):
            # clear the flag so future refinements still write
            del self._resumed
        else:
            self._update_strategy()

        # Rewrite config.json to capture final settings before continuing
        self._save_run_config(
            generations=end_gen,
            offspring_per_gen=self.offspring_per_gen,
            initial_ideation_count=self.initial_ideation_count,
            creative_strategy_refinement_interval=self.creative_strategy_refinement_interval,
            max_archive_capacity=self.archive.max_capacity,
            max_comparisons_at_once=self.archive.max_comparisons_at_once
        )

        # Run from start_gen up to end_gen (inclusive)
        for gen in range(start_gen, end_gen + 1):
            await self.divergent_step()

            # Dump current population snapshot
            archive_sols = self.archive.get_all_solutions()
            genome_ids = [sol.id for sol in archive_sols]
            population_entry = {
                "generation": gen,
                "timestamp": datetime.now().isoformat(),
                "genome_ids": genome_ids,
                "count": len(genome_ids),
            }
            pop_path = Path(self.results_dir) / "population_data.jsonl"
            with pop_path.open("a") as f:
                f.write(json.dumps(population_entry) + "\n")

            # Save novelty metrics for generation 'gen'
            self._save_novelty_metrics(generation=gen, k_neighbors=3)

            await self.human_feedback_step()



    def _save_novelty_metrics(self, generation: int, k_neighbors: int = 3):
        import json
        import numpy as np
        from pathlib import Path
        from datetime import datetime

        # Gather current archive solutions and their embeddings
        archive_sols = self.archive.get_all_solutions()
        embeddings = []
        valid_sols = []
        for sol in archive_sols:
            emb_path = Path(self.results_dir) / "artifacts" / "embeddings" / f"{sol.id}.npy"
            if emb_path.exists():
                emb = np.load(emb_path)
                # flatten if necessary
                if emb.ndim > 1:
                    emb = emb.flatten()
                embeddings.append(emb)
                valid_sols.append(sol)
            else:
                print(f"Warning: missing embedding for {sol.id}, skipping")

        if not embeddings:
            print("No valid embeddings found; skipping novelty metrics save.")
            return

        # Stack and normalize to unit length
        E = np.stack(embeddings)  # shape (n, d)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        E_norm = E / norms

        # Compute cosine-distance matrix
        sim_matrix = E_norm @ E_norm.T
        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, np.inf)

        # Compute average distance to k nearest neighbors for each solution
        avg_distances = []
        for i in range(dist_matrix.shape[0]):
            sorted_dists = np.sort(dist_matrix[i])
            k = min(k_neighbors, len(sorted_dists))
            avg = float(np.mean(sorted_dists[:k]))
            avg_distances.append(avg)

        # Bucket distances by strategy version
        strategy_to_distances: Dict[str, list] = {}
        for sol, dist in zip(valid_sols, avg_distances):
            strat = sol.metadata.get("creative_strategy_name", "None")
            strategy_to_distances.setdefault(strat, []).append(dist)

        # Build metrics per strategy
        strategy_metrics: Dict[str, Dict[str, Any]] = {}
        for strat, dists in strategy_to_distances.items():
            arr = np.array(dists)
            strategy_metrics[strat] = {
                "count":       int(len(arr)),
                "avg_novelty": float(arr.mean()) if len(arr) > 0 else 0.0,
                "std_novelty": float(arr.std())  if len(arr) > 0 else 0.0,
            }

        # Overall metrics
        mean_novelty = float(np.mean(avg_distances))
        mean_genome_length = float(
            np.mean([len(sol.prompt or "") for sol in valid_sols])
        ) if valid_sols else 0.0

        # Assemble entry
        novelty_entry = {
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
            "avg_distance_to_neighbors": avg_distances,
            "mean_novelty": mean_novelty,
            "mean_genome_length": mean_genome_length,
            "strategy_metrics": strategy_metrics,
        }

        # Append to file
        nov_path = Path(self.results_dir) / "novelty_metrics.jsonl"
        with nov_path.open("a") as f:
            f.write(json.dumps(novelty_entry) + "\n")



    def _save_run_config(
        self,
        generations: int,
        offspring_per_gen: int,
        initial_ideation_count: int,
        creative_strategy_refinement_interval: int,
        max_archive_capacity: int,
        max_comparisons_at_once: int,
        check_novelty_before_full: bool = True,
        random_seed: Optional[int] = None,
    ):
        vlm_model = os.getenv("VLM_MODEL") or "UNKNOWN"
        llm_model = os.getenv("LLM_MODEL") or "UNKNOWN"
        creative_strategy_llm_model = os.getenv("CREATIVE_STRATEGY_LLM_MODEL") or "UNKNOWN"

        cfg = {
            "run_id":                          self.run_id,
            "timestamp_utc":                   datetime.utcnow().isoformat(),
            "design_task":                     self.design_task,
            "domain_description":              self.domain_description,
            "blueprint_guidance":               self.blueprint_guidance,
            "using_fixed_strategy":            self.using_fixed_strategy,
            "bfl_model":                       str(self.bfl_model),
            "use_raw_mode":                    self.use_raw_mode,
            "vlm_model":                       vlm_model,
            "llm_model":                       llm_model,
            "creative_strategy_llm_model":     creative_strategy_llm_model,
            "max_archive_capacity":            max_archive_capacity,
            "generations":                     generations,
            "offspring_per_gen":               offspring_per_gen,
            "initial_ideation_count":          initial_ideation_count,
            "creative_strategy_refinement_interval": creative_strategy_refinement_interval,
            "check_novelty_before_full":       check_novelty_before_full,
            "max_comparisons_at_once":         max_comparisons_at_once,
            "random_seed":                     random_seed,
        }

        with open(Path(self.results_dir) / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def load_from_run(
        cls,
        run_dir: str,
        resume_after_generation: int
    ) -> "CreativeSession":
        base = Path(run_dir)

        cfg = json.loads((base / "config.json").read_text())

        session = cls(
            design_task               = cfg["design_task"],
            domain_description        = cfg["domain_description"],
            genotpye_guidance         = cfg["blueprint_guidance"],
            bfl_model                 = FluxModel[cfg["bfl_model"].split(".")[-1]],
            use_raw_mode              = cfg["use_raw_mode"],
            using_fixed_strategy      = False,
            max_archive_capacity      = cfg["max_archive_capacity"],
            check_novelty_before_full = cfg["check_novelty_before_full"],
            max_comparisons_at_once   = cfg["max_comparisons_at_once"],
            creative_strategy_refinement_interval = cfg["creative_strategy_refinement_interval"],
            initial_ideation_count    = cfg["initial_ideation_count"],
            offspring_per_gen         = cfg["offspring_per_gen"],
            run_id                    = cfg["run_id"],
            results_dir               = run_dir,
        )

        all_ids = []
        with open(base / "population_data.jsonl") as f:
            for line in f:
                entry = json.loads(line)
                if entry["generation"] <= resume_after_generation:
                    all_ids = entry["genome_ids"]

        if not all_ids:
            raise ValueError(f"No solutions for gens ≤ {resume_after_generation}")

        old_run_id = all_ids[0].split("_sol_")[0]
        session.archive.solutions = []
        for full_id in all_ids:
            sol_suffix = full_id.split(f"{old_run_id}_", 1)[1]
            meta_path  = base / "artifacts" / "metadata" / f"{sol_suffix}.json"
            meta       = json.loads(meta_path.read_text())
            meta["creative_strategy_name"] = meta.get("strategy_version")


            img_name = Path(meta["image_path"]).name
            new_img  = str(Path(run_dir) / "artifacts" / "images" / img_name)

            sol = Solution(
                id       = full_id,
                prompt   = meta["prompt"],
                idea     = meta["idea"],
                img_path = new_img,
                metadata = meta,
            )
            session.archive.solutions.append(sol)

        meta_dir = base / "artifacts" / "metadata"
        highest = 0
        if meta_dir.exists():
            for p in meta_dir.glob("sol_*.json"):
                try:
                    n = int(p.stem.split("_")[-1])
                    if n > highest:
                        highest = n
                except ValueError:
                    pass
        session.solution_count = highest
        session.completed_first_ideations = True

        strategies = []
        with open(base / "strategy_versions.jsonl") as f:
            for line in f:
                strategies.append(json.loads(line))
        session.strategy_versions = {
            e["strategy_id"]: e["strategy"] for e in strategies
        }
        session.strategy_version_counter  = len(strategies)
        last = strategies[-1]
        session.current_strategy_version   = last["strategy_id"]
        session.previous_archive_analysis  = last.get("previous_archive_analysis", "")

        session.creative_strategy = session.strategy_versions[session.current_strategy_version] #type: ignore

        session._resumed = True

        return session


if __name__ == "__main__":
    futuristic_city = "A futuristic city skyline at sunset"
    astronaut_horse_mars = "a photo of an astronaut riding a horse on Mars"
    novel_architecture = "an architectural style that's never been seen before"
    mythical_creature = "a new mythical creature"
    cotswolds_cow = "a beautiful, photo-realistic scene of a cow in a Cotswolds field"
    cotswolds_cow_2 = "a hyper-realistic photo of a cow in a Cotswolds field"
    anime_fight = "a fight scene from an anime"
    waly_labs_logo = "a logo for WalyLabs, a tech startup who builds agentic AI systems"
    bagel_logo = "a logo for a fun bagel shop called HappyBagel"
    bagel_logo_no_text = "a fun logo for a bagel store, no text"
    pixel_art_char = "pixel-art game avatar (32 x 32) with three upgrade states"
    architecture_sketch = "sketch of a never seen before architectural style"
    quadruped_robot = "a quadruped robot"
    

    realistic_image_domain = "High-quality photorealistic image generation using advanced Diffusion model (July 2025)"
    ambiguous_image_domain = "High-quality image generation using advanced Diffusion model (July 2025)"

    # BFL_MODEL=FluxModel.FLUX_1_KONTEXT_MAX
    # BFL_MODEL=FluxModel.FLUX_1_DEV
    BFL_MODEL=FluxModel.FLUX_1_1_PRO
    #BFL_MODEL=FluxModel.FLUX_1_1_PRO_ULTRA

    USE_RAW_MODE = False

    from core.prompts import PROMPT_ENGINEERING_GUIDANCE


    # session = CreativeSession(
    #     design_task=quadruped_robot,
    #     domain_description=ambiguous_image_domain,
    #     genotpye_guidance=PROMPT_ENGINEERING_GUIDANCE,
    #     bfl_model=BFL_MODEL,
    #     use_raw_mode=USE_RAW_MODE,
    #     using_fixed_strategy=False,
    #     max_archive_capacity=25,
    #     creative_strategy_refinement_interval=999
    #     #genotpye_guidance="The model being used is FLUX1.1 KONTEXT. Prompts can be long, such as a paragraph (but probably 350 words strict maximum), and should be highly detailed -- rather than leaving any ambiguity up to the model, being explicit about details will generally yield better results. Note, there are no negative prompt tags like '--no xyz', or other tags like '[...]', but you can specify if you don't want something to happen in the prompt. The prompt, then, should be highly detailed and reflect the spirit/content/semantics of the IDEA that is given, just in a way that makes sense for FLUX1.1 KONTEXT."
    # )

    tapir_robot = "Stalactite Mason Tapir \u2013 a broad-snouted, cave-roving quadruped robot that harvests mineral-rich drip water and 3-D prints ribbed calcite buttresses along fragile limestone tunnels; porous ceramic hide, silicone suction-pad feet, and faint aqua bioluminescent capillaries illuminate spelunkers who follow its head-lamp halo while it extrudes speleothem lattice arcs to steady cracking vaults."
    cow_with_baby = "From a crouched, wide-angle, cow-eye vantage, a drizzle-flecked Gloucester cow pauses mid-nuzzle with her curious calf beside a vivid poppy-and-cornflower strip at the paddock’s edge. A bruised slate sky threatens thunder, yet one late-afternoon sunshaft breaks through, rim-lighting rain-beaded coats like pearls. Dry-stone walls snake toward a distant honey-stone hamlet and church spire, half-veiled by haze. The air tastes of wet limestone and ozone; a low rumble hangs in the hush—an intimate, electric stillness seconds before the downpour."
    cow_winter = """Title: “First Snow Silence”

Micro-narrative (20 words):
Dawn glaze tints fresh snow lavender; lone Hereford creaks through crust, breath pluming; church bell echoes over winter-blanketed ridge faintly.

Key parameters selected  
• Space & Perspective: chest-high viewpoint tucked inside a leaf-stripped hawthorn hedge, 135 mm short-telephoto, hedge branches forming natural dark vignette  
• Time & Atmosphere: mid-winter blue-hour dawn, first light kissing horizon before sunbreak  
• Weather & Light Quality: overnight powder snow, soft violet-pink alpenglow reflecting off cloud base, diffused rim light on cow’s snow-dusted back  
• Field Details: dry-stone wall half-buried by drift, ridge-and-furrow undulations ghosting beneath snow, ice-bearded wooden stile foreground, distant honey-stone church tower just visible through pastel haze  
• Cow Behaviour & Pose: stationary broadside, head turned slightly to lens, visible warm exhale pluming and catching pink light

Lighting & weather summary  
Pre-sun lavender sky bounces cool fill across field while low, unseen sun edges horizon, giving subtle warm-cool split; fresh snow acts as giant reflector, lifting shadow detail and emphasising every breath plume and hoof-print.

Unique authenticity note  
The unmistakable stepped silhouette of St Kenelm’s tower at Sapperton anchors location; ridge-and-furrow lines—characteristic of medieval Cotswolds agriculture—still read through the snow sheen, reinforcing documentary believability."
"""
    robot_mechanical = "Causeway Forge Table-Walker — A squat, scaffold-like quadruped that strides flooded deltas, pile-driving telescopic auger legs into silt while 3-D–printing limestone slabs from pumped mud, leaving a sun-cured raised footpath in its wake for villagers and wildlife."

    robot_dir = "results/robot_o3_refine_contained"
    tapir_robot_img = "results/robot_o3_refine_contained/artifacts/images/run_20250806_181002_sol_054.png"
    
    robot_mechanical_img = "results/robot_o3_refine_contained/artifacts/images/run_20250806_181002_sol_030.png"

    cow_dir = "results/cows_o3_norefine_contained"
    cow_with_baby_img = "results/cows_o3_norefine_contained/artifacts/images/run_20250806_003904_sol_002.png"
    cow_winter_img = "results/cows_o3_norefine_contained/artifacts/images/run_20250806_003904_sol_021.png"

    branch_context = BranchContext(
        parent_run_dir=robot_dir,
        reference_img_paths=[robot_mechanical_img],
        reference_ideas=[robot_mechanical],
        branch_depth=1,
        new_design_task="i like the mechanical vibes of this one, less looking like an animal"
    )
    session = CreativeSession(
        design_task=quadruped_robot,
        domain_description=ambiguous_image_domain,
        genotpye_guidance=PROMPT_ENGINEERING_GUIDANCE,
        bfl_model=BFL_MODEL,
        use_raw_mode=USE_RAW_MODE,
        using_fixed_strategy=False,
        max_archive_capacity=25,
        creative_strategy_refinement_interval=999,
        branch_context=branch_context
    )
    asyncio.run(session.initialise())
    asyncio.run(session.run())


# helper to resume an existing run:
async def run_resumed_session():
    from core.resume_run import resume_run
    session = resume_run(
        old_run_dir="results/robot_o3_refine_contained",
        new_run_dir="results/robot_o3_norefine_contained_abl",
        resume_after_generation=4,
        override_creative_strategy_refinement_interval=99999
    )
    await session.run(generations=50)

# if __name__ == "__main__":
#     asyncio.run(run_resumed_session())
