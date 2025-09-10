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

from core.image_generator import FluxAPIAdapter as FluxImageGenerator, FluxModel
from core.image_embedder import ImageEmbedder
from core.utils import init_run_dirs


@dataclass
class BaselineConfig:
    design_task: str
    domain_description: str
    model: FluxModel
    use_raw_mode: bool
    aspect_ratio: Optional[str]
    safety_tolerance: int
    seeds: List[int]
    k_neighbors: int
    results_dir: Path


def _write_config(cfg: BaselineConfig) -> None:
    out = {
        "run_id": cfg.results_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "design_task": cfg.design_task,
        "domain_description": cfg.domain_description,
        "using_fixed_strategy": False,
        "bfl_model": str(cfg.model),
        "use_raw_mode": cfg.use_raw_mode,
        "generations": len(cfg.seeds),
        "seeds": cfg.seeds,
        "offspring_per_gen": 1,
        "initial_ideation_count": len(cfg.seeds),
        "creative_strategy_refinement_interval": 0,
        "check_novelty_before_full": False,
        "max_comparisons_at_once": 0,
        "random_seed": None,
        "k_neighbors": cfg.k_neighbors,
        "aspect_ratio": cfg.aspect_ratio,
        "safety_tolerance": cfg.safety_tolerance,
    }
    (cfg.results_dir / "config.json").write_text(json.dumps(out, indent=2))


def run_baseline(
    design_task: str,
    domain_description: str,
    seeds: List[int],
    model: FluxModel,
    use_raw_mode: bool,
    aspect_ratio: Optional[str],
    safety_tolerance: int,
    k_neighbors: int,
    results_root: Optional[str],
    concurrency: int,
    save_embeddings: bool,
) -> Path:
    # Prepare results directory
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = Path(results_root or "direct_prompting/results") / run_id
    paths = init_run_dirs(str(results_dir))

    # Init generator and embedder
    generator = FluxImageGenerator(use_raw_mode=use_raw_mode)
    generator.local_save_dir = results_dir / "artifacts" / "images"
    embedder = ImageEmbedder() if save_embeddings else None

    cfg = BaselineConfig(
        design_task=design_task,
        domain_description=domain_description,
        model=model,
        use_raw_mode=use_raw_mode,
        aspect_ratio=aspect_ratio,
        safety_tolerance=safety_tolerance,
        seeds=seeds,
        k_neighbors=k_neighbors,
        results_dir=results_dir,
    )
    _write_config(cfg)

    async def _worker(idx_seed: Tuple[int, int], sem: asyncio.Semaphore) -> None:
        idx, seed = idx_seed
        sol_id = f"{run_id}_sol_{idx:03d}"
        async with sem:
            try:
                result = await asyncio.to_thread(
                    generator.generate_image,
                    design_task,
                    model,
                    1024,
                    1024,
                    aspect_ratio,
                    safety_tolerance,
                    int(seed),
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
            _worker((i, s), sem) for i, s in enumerate(seeds, start=1)
        ])

    asyncio.run(_run_all())

    print(f" Baseline run complete: {results_dir}")
    return results_dir


def _load_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.json"
    return json.loads(cfg_path.read_text())


def _resolve_model_from_cfg(cfg: Dict[str, Any]) -> FluxModel:
    raw = str(cfg.get("bfl_model", ""))
    key = raw.split(".")[-1] if "." in raw else raw
    try:
        return FluxModel[key]
    except KeyError:
        for m in FluxModel:
            if m.value == raw:
                return m
    raise SystemExit(f" Cannot resolve model from config: {raw}")


def _derive_seeds_from_cfg(cfg: Dict[str, Any]) -> List[int]:
    if isinstance(cfg.get("seeds"), list):
        try:
            return [int(x) for x in cfg["seeds"]]
        except Exception:
            pass
    n = int(cfg.get("generations", 0))
    return [i for i in range(1, n + 1)]


def _list_missing_indices(run_dir: Path, run_id: str, total: int) -> List[int]:
    imgs = run_dir / "artifacts" / "images"
    missing: List[int] = []
    for i in range(1, total + 1):
        name = f"{run_id}_sol_{i:03d}.png"
        if not (imgs / name).exists():
            missing.append(i)
    return missing


def repair_run(
    run_dir: Path,
    concurrency: int,
    save_embeddings: bool,
    attempts: int = 1,
) -> None:
    cfg = _load_config(run_dir)
    run_id = str(cfg.get("run_id", run_dir.name))
    design_task = str(cfg.get("design_task", ""))
    if not design_task:
        raise SystemExit(" design_task missing in config.json")
    model = _resolve_model_from_cfg(cfg)
    use_raw = bool(cfg.get("use_raw_mode", False))
    aspect_ratio = cfg.get("aspect_ratio")
    safety = int(cfg.get("safety_tolerance", 6))
    all_seeds = _derive_seeds_from_cfg(cfg)
    total = int(cfg.get("generations", len(all_seeds)))

    generator = FluxImageGenerator(use_raw_mode=use_raw)
    generator.local_save_dir = run_dir / "artifacts" / "images"
    embedder = ImageEmbedder() if save_embeddings else None
    emb_dir = run_dir / "artifacts" / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    async def _worker(idx: int, seed: int, sem: asyncio.Semaphore) -> None:
        sol_id = f"{run_id}_sol_{idx:03d}"
        async with sem:
            try:
                result = await asyncio.to_thread(
                    generator.generate_image,
                    design_task,
                    model,
                    1024,
                    1024,
                    aspect_ratio,
                    safety,
                    int(seed),
                    True,
                    False,
                    f"{sol_id}.png",
                    300,
                )
            except Exception as e:
                print(f"Warning: repair generation failed for {sol_id}: {e}")
                return
        img_path = result.get("local_path") or result.get("image_url")
        if save_embeddings and embedder is not None and img_path:
            try:
                emb = await asyncio.to_thread(embedder.embed, img_path)
                np.save(emb_dir / f"{sol_id}.npy", emb.cpu().numpy())
            except Exception as e:
                print(f"Warning: failed embedding during repair for {sol_id}: {e}")

    for attempt in range(1, int(attempts) + 1):
        missing = _list_missing_indices(run_dir, run_id, total)
        if not missing:
            print(" No missing images to repair.")
            return
        print(f"Attempt {attempt}: repairing {len(missing)} missing images → {missing}")
        async def _run_all() -> None:
            sem = asyncio.Semaphore(max(1, int(concurrency)))
            await asyncio.gather(*[
                _worker(i, all_seeds[i - 1], sem) for i in missing
            ])
        asyncio.run(_run_all())
    still_missing = _list_missing_indices(run_dir, run_id, total)
    if still_missing:
        print(f" Still missing after repair attempts: {still_missing}")
    else:
        print(" Repair completed. All images present.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct-prompting baseline runner")
    p.add_argument("--design", required=False, help="Design task prompt")
    p.add_argument("--tasks-file", required=False, type=str,
                   help="Path to a text file with one design task per line (comments starting with #)")
    p.add_argument("--domain", default="Direct prompting baseline", help="Domain description text")
    p.add_argument("--model", default="FLUX_1_1_PRO",
                   help="Flux model (enum name like FLUX_1_1_PRO or value like flux-pro-1.1 / flux-kontext-max)")
    p.add_argument("--raw", action="store_true", help="Use raw mode for generator")
    p.add_argument("--aspect", default="16:9", help="Aspect ratio, e.g., 16:9")
    p.add_argument("--safety", type=int, default=6, help="Safety tolerance (1-6)")
    p.add_argument("--k", type=int, default=3, help="k for k-NN novelty")
    p.add_argument("--num", type=int, default=24, help="Number of images (seeds) to generate")
    p.add_argument("--seed0", type=int, default=None, help="Optional base seed to generate sequential seeds")
    p.add_argument("--results-root", type=str, default=None, help="Root results dir (default: ./direct_prompting/results)")
    p.add_argument("--concurrency", type=int, default=1, help="Max concurrent generation requests")
    p.add_argument("--no-embeddings", action="store_true", help="Do not compute/save embeddings")
    # Repair mode
    p.add_argument("--repair-run", type=str, default=None, help="Path to an existing run to backfill missing images")
    p.add_argument("--repair-attempts", type=int, default=1, help="Number of repair passes to attempt")
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

    # Build seeds list template
    if args.seed0 is not None:
        base_seeds = [int(args.seed0) + i for i in range(int(args.num))]
    else:
        base_seeds = [i for i in range(1, int(args.num) + 1)]

    # If repair mode is requested, do only repair and exit
    if args.repair_run:
        rd = Path(args.repair_run).expanduser().resolve()
        if not rd.exists():
            raise SystemExit(f" repair run dir not found: {rd}")
        repair_run(
            run_dir=rd,
            concurrency=int(args.concurrency),
            save_embeddings=not bool(args.no_embeddings),
            attempts=int(args.repair_attempts),
        )
        return

    # Gather tasks: either single --design or from --tasks-file
    tasks: List[str] = []
    if args.tasks_file:
        fp = Path(args.tasks_file).expanduser().resolve()
        if not fp.exists():
            raise SystemExit(f" tasks file not found: {fp}")
        for ln in fp.read_text().splitlines():
            s = ln.strip()
            if not s or s.startswith('#'):
                continue
            tasks.append(s)
    if args.design:
        tasks.append(args.design)
    if not tasks:
        raise SystemExit(" Provide --design or --tasks-file with at least one task.")

    for idx, task in enumerate(tasks, start=1):
        # For reproducibility across tasks, use the same sequential seeds per run
        seeds = list(base_seeds)
        print(f"\n=== Running baseline ({idx}/{len(tasks)}): {task} ===\n")
        run_baseline(
            design_task=task,
            domain_description=args.domain,
            seeds=seeds,
            model=model,
            use_raw_mode=bool(args.raw),
            aspect_ratio=args.aspect,
            safety_tolerance=int(args.safety),
            k_neighbors=int(args.k),
            results_root=args.results_root,
            concurrency=int(args.concurrency),
            save_embeddings=not bool(args.no_embeddings),
        )


if __name__ == "__main__":
    main()


