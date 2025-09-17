from typing import Optional
import shutil
import json
import random
from pathlib import Path

import numpy as np

from core.utils import init_run_dirs
from core.creative_session import CreativeSession

def resume_run(
    old_run_dir: str,
    new_run_dir: str,
    resume_after_generation: int,
    *,
    override_creative_strategy_refinement_interval: Optional[int] = None,
    override_max_archive_capacity: Optional[int] = None,
    override_random_seed: Optional[int] = None,
) -> CreativeSession:
    old = Path(old_run_dir)
    new = Path(new_run_dir)
    new.mkdir(parents=True, exist_ok=True)
    init_run_dirs(str(new))

    # 1) Copy & patch config.json
    cfg = json.loads((old / "config.json").read_text())
    cfg["run_id"] = new.name
    if override_creative_strategy_refinement_interval is not None:
        cfg["creative_strategy_refinement_interval"] = override_creative_strategy_refinement_interval
    if override_max_archive_capacity is not None:
        cfg["max_archive_capacity"] = override_max_archive_capacity
    if override_random_seed is not None:
        cfg["random_seed"] = override_random_seed
    (new / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # 2) Truncate population_data.jsonl & novelty_metrics.jsonl
    def truncate_jsonl(fname: str):
        inp = old / fname
        out = new / fname
        with inp.open() as rin, out.open("w") as wout:
            for line in rin:
                entry = json.loads(line)
                if entry.get("generation", 0) <= resume_after_generation:
                    wout.write(line)

    truncate_jsonl("population_data.jsonl")
    truncate_jsonl("novelty_metrics.jsonl")

    # 3) Copy only the first N artifact files
    blueprint_ids = []
    with open(new / "population_data.jsonl") as f:
        for line in f:
            blueprint_ids = json.loads(line)["blueprint_ids"]

    old_run_id = blueprint_ids[0].split("_sol_")[0]
    sol_suffixes = { full_id.split(f"{old_run_id}_",1)[1] for full_id in blueprint_ids }

    for subdir in ("images", "embeddings", "prompts", "metadata", "markdowns"):
        src_dir = old / "artifacts" / subdir
        dst_dir = new / "artifacts" / subdir
        for file in src_dir.iterdir():
            name = file.stem
            suffix = name if subdir in ("metadata","markdowns") else name.split(f"{old_run_id}_",1)[1]
            if suffix in sol_suffixes:
                shutil.copy(file, dst_dir / file.name)

    # 4) Truncate strategy_versions.jsonl based on YOUR override
    new_cfg = json.loads((new / "config.json").read_text())
    interval = new_cfg["creative_strategy_refinement_interval"]
    num_refines = resume_after_generation // interval
    keep = 1 + num_refines  # initial + each actual refinement

    with (old / "strategy_versions.jsonl").open() as fin, \
         (new / "strategy_versions.jsonl").open("w") as fout:
        for idx, line in enumerate(fin):
            if idx < keep:
                fout.write(line)
            else:
                break

    # 5) Re‐seed RNG if requested
    if override_random_seed is not None:
        random.seed(override_random_seed)
        np.random.seed(override_random_seed)

    # 6) Finally, rehydrate the session
    session = CreativeSession.load_from_run(
        run_dir=str(new),
        resume_after_generation=resume_after_generation,
    )

    return session
