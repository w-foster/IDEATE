import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Optional

from core.branch_context import BranchContext
from core.creative_session import CreativeSession
from core.image_generator import FluxModel
from core.prompts import PROMPT_ENGINEERING_GUIDANCE


def load_parent_config(parent_run_dir: Path) -> dict:
    cfg_path = parent_run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in {parent_run_dir}")
    return json.loads(cfg_path.read_text())


def infer_flux_model(cfg: dict) -> FluxModel:
    name = str(cfg["bfl_model"]).split(".")[-1]
    return FluxModel[name]


def resolve_image_paths(parent_run_dir: Path, images: List[str]) -> List[Path]:
    imgs_dir = parent_run_dir / "artifacts" / "images"
    out: List[Path] = []
    for img in images:
        p = Path(img)
        if not p.is_absolute():
            p = imgs_dir / img
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        out.append(p)
    return out


def find_metadata_for_image(parent_run_dir: Path, image_path: Path) -> dict:
    meta_dir = parent_run_dir / "artifacts" / "metadata"
    if not meta_dir.exists():
        raise FileNotFoundError(f"Missing metadata dir: {meta_dir}")

    filename = image_path.name
    for meta_file in sorted(meta_dir.glob("sol_*.json")):
        data = json.loads(meta_file.read_text())
        if str(data.get("image_path", "")).endswith(filename):
            return data

    stem = image_path.stem
    if "_sol_" in stem:
        try:
            sol_num = stem.split("_sol_")[-1]
            meta_file = meta_dir / f"sol_{sol_num}.json"
            if meta_file.exists():
                return json.loads(meta_file.read_text())
        except Exception:
            pass

    raise RuntimeError(f"No metadata matched image '{image_path}'")


def collect_reference_ideas(parent_run_dir: Path, image_paths: List[Path]) -> List[str]:
    ideas: List[str] = []
    for p in image_paths:
        meta = find_metadata_for_image(parent_run_dir, p)
        idea = meta.get("idea")
        if not idea:
            raise RuntimeError(f"Missing 'idea' in metadata for {p}")
        ideas.append(idea)
    return ideas


async def run_branch(
    parent_run_dir: Path,
    images: List[str],
    new_design_task: str,
    generations: int,
    initial_ideation_count: int,
    refinement_interval: int,
    max_archive_capacity: int,
    offspring_per_gen: int,
    out_results_dir: Optional[Path],
) -> None:
    parent_cfg = load_parent_config(parent_run_dir)
    model = infer_flux_model(parent_cfg)

    abs_image_paths = resolve_image_paths(parent_run_dir, images)
    reference_ideas = collect_reference_ideas(parent_run_dir, abs_image_paths)

    # Aggregate prior branch context if available
    prior_texts: List[str] = []
    prior_img_paths: List[str] = []
    prior_ideas: List[str] = []
    branch_json = parent_run_dir / "branch.json"
    branch_depth = 1
    if branch_json.exists():
        try:
            prev = json.loads(branch_json.read_text())
            branch_depth = int(prev.get("branch_depth", 1)) + 1
            # Prior images from previous branch.json
            prev_img_paths = prev.get("reference_img_paths", []) or []
            prior_img_paths.extend([str(Path(p).resolve()) for p in prev_img_paths])
            # Prior text
            prev_text = prev.get("new_design_task")
            if prev_text:
                prior_texts.append(prev_text)
            # Attempt to derive prior ideas from metadata
            for p in prev_img_paths:
                try:
                    meta = find_metadata_for_image(parent_run_dir, Path(p))
                    idea = meta.get("idea")
                    if idea:
                        prior_ideas.append(idea)
                except Exception:
                    pass
        except Exception:
            pass

    branch_ctx = BranchContext(
        parent_run_dir=str(parent_run_dir.resolve()),
        reference_img_paths=[str(p) for p in abs_image_paths],
        reference_ideas=reference_ideas,
        new_design_task=new_design_task,
        branch_depth=branch_depth,
        prior_reference_img_paths=prior_img_paths,
        prior_reference_ideas=prior_ideas,
        prior_branch_texts=prior_texts,
    )

    session = CreativeSession(
        design_task=parent_cfg["design_task"],
        domain_description=parent_cfg["domain_description"],
        blueprint_guidance=PROMPT_ENGINEERING_GUIDANCE,
        bfl_model=model,
        use_raw_mode=bool(parent_cfg.get("use_raw_mode", False)),
        using_fixed_strategy=False,
        max_archive_capacity=max_archive_capacity,
        check_novelty_before_full=bool(parent_cfg.get("check_novelty_before_full", True)),
        max_comparisons_at_once=int(parent_cfg.get("max_comparisons_at_once", 5)),
        creative_strategy_refinement_interval=refinement_interval,
        initial_ideation_count=initial_ideation_count,
        offspring_per_gen=offspring_per_gen,
        results_dir=str(out_results_dir) if out_results_dir else None,
        run_id=None if out_results_dir else None,
        branch_context=branch_ctx,
    )

    await session.initialise()
    await session.run(generations=generations)
    print(f"Branch complete. New results at: {session.results_dir}")


def main():
    ap = argparse.ArgumentParser(description="Create a convergent branch from a completed run.")
    ap.add_argument("--parent-run-dir", required=True, help="Path to parent run dir (contains config.json)")
    ap.add_argument("--images", nargs="+", required=True, help="One or more image filenames or absolute paths")
    ap.add_argument("--new-text", required=True, help="New design task text for this branch")
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--initial-ideation-count", type=int, default=5)
    ap.add_argument("--refinement-interval", type=int, default=999)
    ap.add_argument("--max-archive-capacity", type=int, default=25)
    ap.add_argument("--offspring-per-gen", type=int, default=1)
    ap.add_argument("--results-dir", type=str, default=None, help="Optional explicit output dir for the new run")

    args = ap.parse_args()
    parent_run_dir = Path(args.parent_run_dir).resolve()
    out_results_dir = Path(args.results_dir).resolve() if args.results_dir else None

    asyncio.run(run_branch(
        parent_run_dir=parent_run_dir,
        images=args.images,
        new_design_task=args.new_text,
        generations=args.generations,
        initial_ideation_count=args.initial_ideation_count,
        refinement_interval=args.refinement_interval,
        max_archive_capacity=args.max_archive_capacity,
        offspring_per_gen=args.offspring_per_gen,
        out_results_dir=out_results_dir,
    ))


if __name__ == "__main__":
    main()


