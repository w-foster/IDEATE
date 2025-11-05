from __future__ import annotations

import argparse
import asyncio
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from core.image_generator import FluxAPIAdapter as FluxImageGenerator, FluxModel
from core.image_embedder import ImageEmbedder
from core.utils import init_run_dirs
from langchain_openai import ChatOpenAI

LLM_MODEL = "o3"

class Prompts(BaseModel):
    prompts: List[str] = Field(description="List of diverse, high-quality prompts for image generation")


@dataclass
class SinglePromptBaselineConfig:
    design_task: str
    domain_description: str
    model: FluxModel
    use_raw_mode: bool
    aspect_ratio: Optional[str]
    safety_tolerance: int
    num_images: int
    results_dir: Path
    llm_model: str = LLM_MODEL


def _write_config(cfg: SinglePromptBaselineConfig) -> None:
    out = {
        "run_id": cfg.results_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "design_task": cfg.design_task,
        "domain_description": cfg.domain_description,
        "using_single_prompt_baseline": True,
        "bfl_model": str(cfg.model),
        "use_raw_mode": cfg.use_raw_mode,
        "num_images": cfg.num_images,
        "offspring_per_gen": 1,
        "initial_ideation_count": cfg.num_images,
        "creative_strategy_refinement_interval": 0,
        "check_novelty_before_full": False,
        "max_comparisons_at_once": 0,
        "random_seed": None,
        "aspect_ratio": cfg.aspect_ratio,
        "safety_tolerance": cfg.safety_tolerance,
        "llm_model": cfg.llm_model,
    }
    (cfg.results_dir / "config.json").write_text(json.dumps(out, indent=2))


async def generate_prompts_archive(
    design_task: str,
    domain_description: str,
    num_images: int,
    llm_model: str = "gpt-4o-mini"
) -> List[str]:
    """Generate a diverse archive of prompts using a single LLM call."""

    llm = ChatOpenAI(
        model=llm_model,
    )

    # Create structured output LLM
    structured_llm = llm.with_structured_output(Prompts)

    system_prompt = f"""We want to make a diverse set of images which align with the design task at hand. Your role is to produce a set of prompts (in one go) to achieve this.
The prompts will be directly fed into a SOTA image generation (diffusion) model.

You may spend time reasoning about your task, and when you are ready output the prompts into the structured output schema provided."""

    user_prompt = f"Design task: '{design_task}'"

    print(f"\nGenerating {num_images} prompts using {llm_model}...")

    response: Prompts = await structured_llm.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    print(f"Successfully generated {len(response.prompts)} prompts")
    return response.prompts


def run_single_prompt_baseline(
    design_task: str,
    domain_description: str,
    num_images: int,
    model: FluxModel,
    use_raw_mode: bool,
    aspect_ratio: Optional[str],
    safety_tolerance: int,
    results_root: Optional[str],
    concurrency: int,
    save_embeddings: bool,
    llm_model: str = LLM_MODEL,
) -> Path:
    """Run the single-prompt baseline that generates all prompts in one LLM call."""

    # Prepare results directory
    run_id = f"single_prompt_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = Path(results_root or "baselines/single_prompt_archive/results") / run_id
    paths = init_run_dirs(str(results_dir))

    # Generate all prompts in one LLM call
    prompts = asyncio.run(generate_prompts_archive(
        design_task=design_task,
        domain_description=domain_description,
        num_images=num_images,
        llm_model=llm_model
    ))

    # Save prompts to file for inspection
    prompts_data = {"prompts": prompts}
    (results_dir / "generated_prompts.json").write_text(json.dumps(prompts_data, indent=2))

    # Init generator and embedder
    generator = FluxImageGenerator(use_raw_mode=use_raw_mode)
    generator.local_save_dir = results_dir / "artifacts" / "images"
    embedder = ImageEmbedder() if save_embeddings else None

    cfg = SinglePromptBaselineConfig(
        design_task=design_task,
        domain_description=domain_description,
        model=model,
        use_raw_mode=use_raw_mode,
        aspect_ratio=aspect_ratio,
        safety_tolerance=safety_tolerance,
        num_images=num_images,
        results_dir=results_dir,
        llm_model=llm_model,
    )
    _write_config(cfg)

    async def _worker(idx_prompt: Tuple[int, str], sem: asyncio.Semaphore) -> None:
        idx, prompt = idx_prompt
        sol_id = f"{run_id}_sol_{idx:03d}"
        async with sem:
            try:
                result = await asyncio.to_thread(
                    generator.generate_image,
                    prompt,  # Use the generated prompt directly
                    model,
                    1024,
                    1024,
                    aspect_ratio,
                    safety_tolerance,
                    int(idx),  # Use index as seed for reproducibility
                    True,
                    False,
                    f"{sol_id}.png",
                    300,
                )
            except Exception as e:
                print(f"Warning: generation failed for {sol_id}: {e}")
                return
        img_path = result.get("local_path") or result.get("image_url")
        if save_embeddings and embedder is not None and img_path:
            try:
                emb = await asyncio.to_thread(embedder.embed, img_path)
                np.save(paths["embeddings"] / f"{sol_id}.npy", emb.cpu().numpy())
            except Exception as e:
                print(f"Warning: failed embedding for {sol_id}: {e}")

    async def _run_all() -> None:
        sem = asyncio.Semaphore(max(1, int(concurrency)))
        await asyncio.gather(*[
            _worker((i + 1, prompt), sem) for i, prompt in enumerate(prompts)
        ])

    asyncio.run(_run_all())

    print(f"Single-prompt baseline run complete: {results_dir}")
    return results_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-prompt archive baseline runner")
    p.add_argument("--design", required=True, help="Design task prompt")
    p.add_argument("--domain", default="Image generation with advanced diffusion model",
                   help="Domain description text")
    p.add_argument("--model", default="FLUX_1_1_PRO",
                   help="Flux model (enum name like FLUX_1_1_PRO or value like flux-pro-1.1 / flux-kontext-max)")
    p.add_argument("--raw", action="store_true", help="Use raw mode for generator")
    p.add_argument("--aspect", default="16:9", help="Aspect ratio, e.g., 16:9")
    p.add_argument("--safety", type=int, default=6, help="Safety tolerance (1-6)")
    p.add_argument("--num", type=int, default=24, help="Number of images to generate")
    p.add_argument("--results-root", type=str, default=None,
                   help="Root results dir (default: ./baselines/single_prompt_archive/results)")
    p.add_argument("--concurrency", type=int, default=1, help="Max concurrent generation requests")
    p.add_argument("--no-embeddings", action="store_true", help="Do not compute/save embeddings")
    p.add_argument("--llm-model", default="gpt-4o-mini",
                   help="LLM model to use for prompt generation (default: gpt-4o-mini)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # resolve model from enum name or value string
    def _resolve_model(s: str) -> FluxModel:
        # Try enum name
        try:
            return FluxModel[s]
        except KeyError:
            pass
        # Try matching a value (e.g., "flux-kontext-max")
        for m in FluxModel:
            if m.value == s:
                return m
        raise SystemExit(f" Unknown model: {s}. Use enum name (e.g., FLUX_1_1_PRO) or value (e.g., flux-pro-1.1)")

    model = _resolve_model(args.model)

    run_single_prompt_baseline(
        design_task=args.design,
        domain_description=args.domain,
        num_images=int(args.num),
        model=model,
        use_raw_mode=bool(args.raw),
        aspect_ratio=args.aspect,
        safety_tolerance=int(args.safety),
        results_root=args.results_root,
        concurrency=int(args.concurrency),
        save_embeddings=not bool(args.no_embeddings),
        llm_model=args.llm_model,
    )


if __name__ == "__main__":
    main()
