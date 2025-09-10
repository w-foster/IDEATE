import argparse
import asyncio
import csv
from pathlib import Path
from typing import Optional

from core.creative_session import CreativeSession
from core.image_generator import FluxModel
from core.prompts import PROMPT_ENGINEERING_GUIDANCE


def parse_flux_model(model_str: str) -> FluxModel:
    """
    Accept either an enum name (e.g., FLUX_1_1_PRO) or the underlying value (e.g., flux-pro-1.1).
    Defaults to FLUX_1_1_PRO if unrecognized.
    """
    # Try enum name first
    try:
        return FluxModel[model_str]
    except KeyError:
        pass

    # Try by value
    for m in FluxModel:
        if m.value == model_str:
            return m

    # Fallback
    return FluxModel.FLUX_1_KONTEXT_MAX


async def run_single(prompt: str,
                     model: FluxModel,
                     generations: int,
                     domain_description: str,
                     use_raw_mode: bool,
                     initial_ideation_count: int,
                     refinement_interval: int,
                     max_archive_capacity: int,
                     max_comparisons_at_once: int,
                     offspring_per_gen: int) -> None:
    session = CreativeSession(
        design_task=prompt,
        domain_description=domain_description,
        genotpye_guidance=PROMPT_ENGINEERING_GUIDANCE,
        bfl_model=model,
        use_raw_mode=use_raw_mode,
        using_fixed_strategy=False,
        max_archive_capacity=max_archive_capacity,
        check_novelty_before_full=True,
        max_comparisons_at_once=max_comparisons_at_once,
        creative_strategy_refinement_interval=refinement_interval,
        initial_ideation_count=initial_ideation_count,
        offspring_per_gen=offspring_per_gen,
    )

    await session.initialise()
    await session.run(generations=generations)


async def run_batch(csv_path: Path,
                    prompt_col: str,
                    model: FluxModel,
                    generations: int,
                    domain_description: str,
                    use_raw_mode: bool,
                    initial_ideation_count: int,
                    refinement_interval: int,
                    max_archive_capacity: int,
                    max_comparisons_at_once: int,
                    offspring_per_gen: int,
                    limit: Optional[int]) -> None:
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if limit is not None:
        rows = rows[:limit]

    for idx, row in enumerate(rows, start=1):
        prompt = row.get(prompt_col) or row.get('prompt') or row.get('Prompt')
        if not prompt:
            print(f"[skip {idx}] No '{prompt_col}' column present for row: {row}")
            continue

        print(f"\n=== [{idx}/{len(rows)}] Running prompt: {prompt[:120]} ===")
        try:
            await run_single(
                prompt=prompt,
                model=model,
                generations=generations,
                domain_description=domain_description,
                use_raw_mode=use_raw_mode,
                initial_ideation_count=initial_ideation_count,
                refinement_interval=refinement_interval,
                max_archive_capacity=max_archive_capacity,
                max_comparisons_at_once=max_comparisons_at_once,
                offspring_per_gen=offspring_per_gen,
            )
        except Exception as e:
            print(f"[error] Failed run for prompt: {e}")


def main():
    parser = argparse.ArgumentParser(description="Batch run CreativeSession for a list of prompts.")
    parser.add_argument("csv", type=str, help="Path to CSV with a 'prompt' column (or specify --prompt-col)")
    parser.add_argument("--prompt-col", type=str, default="prompt", help="Column name for prompts")
    parser.add_argument("--model", type=str, default="FLUX_1_1_PRO", help="FluxModel enum name or value")
    parser.add_argument("--generations", type=int, default=50, help="Generations per run")
    parser.add_argument("--domain-description", type=str,
                        default="image generation via an advanced diffusion model (FLUX)",
                        help="Domain description string")
    parser.add_argument("--use-raw-mode", action="store_true", help="Enable raw mode for FLUX API")
    parser.add_argument("--initial-ideation-count", type=int, default=5)
    parser.add_argument("--refinement-interval", type=int, default=8)
    parser.add_argument("--max-archive-capacity", type=int, default=20)
    parser.add_argument("--max-comparisons-at-once", type=int, default=5)
    parser.add_argument("--offspring-per-gen", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to run")

    args = parser.parse_args()

    model = parse_flux_model(args.model)
    csv_path = Path(args.csv)

    asyncio.run(run_batch(
        csv_path=csv_path,
        prompt_col=args.prompt_col,
        model=model,
        generations=args.generations,
        domain_description=args.domain_description,
        use_raw_mode=args.use_raw_mode,
        initial_ideation_count=args.initial_ideation_count,
        refinement_interval=args.refinement_interval,
        max_archive_capacity=args.max_archive_capacity,
        max_comparisons_at_once=args.max_comparisons_at_once,
        offspring_per_gen=args.offspring_per_gen,
        limit=args.limit,
    ))


if __name__ == "__main__":
    main()


