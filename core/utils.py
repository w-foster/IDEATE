from typing import Optional
from pathlib import Path
import os
from datetime import datetime
import shutil
import shutil
from pathlib import Path
from core.solution import Solution
import json


from datetime import datetime 
from pathlib import Path
import os
from core.solution import Solution

from langgraphs.archive_addition.archive_addition_graph import ArchiveAdditionState, NoveltyCheck, Evaluation


def create_solution_markdown(
    solution: Solution,
    run_id: str,
    solution_count: int,
    design_task: str,
    domain_description: str,
    model_used: str,
    added_to_archive: bool,
    addition_workflow_output: Optional[ArchiveAdditionState],
    base_results_dir: str = "./results",
    **metadata,
) -> str:
    """
    Write a full markdown record for `solution` to
        results/<run_id>/artifacts/markdowns/sol_###.md

    • No image copying is performed here.
    • The markdown content is unchanged from the previous version.
    """

    # ------------------------------------------------------------------ #
    #  target paths
    # ------------------------------------------------------------------ #
    artifacts_dir  = Path(base_results_dir) / run_id / "artifacts"
    markdowns_dir  = artifacts_dir / "markdowns"
    images_dir     = artifacts_dir / "images"           # images are saved here elsewhere

    markdowns_dir.mkdir(parents=True, exist_ok=True)

    sol_id = f"sol_{solution_count:03d}"
    img_filename = f"{sol_id}.png"
    relative_img_path = f"../images/{img_filename}" if (images_dir / img_filename).exists() \
                        else (solution.img_path or "No image available")

    # ------------------------------------------------------------------ #
    #  markdown body (identical to the old one)
    # ------------------------------------------------------------------ #
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    #ArchiveAdditionState()
    # info for markdown
    if not addition_workflow_output:
        addition_workflow_info = "NO ARCHIVE ADDITION INFO AVAILABLE (likely because this was the first solution)"
    else:
        addition_workflow_info = format_archive_addition_info(
            workflow_output=addition_workflow_output, 
            added=added_to_archive
        )

    markdown_content = f"""# Solution {sol_id}

## Overview
| Parameter      | Value          |
|----------------|----------------|
| Solution ID    | {sol_id} 
| Timestamp      | {timestamp} 
| Model Used     | {model_used} 
| Run ID         | {run_id} 

{addition_workflow_info}


## Creative Details

### **Idea**
{solution.idea}

### **Prompt**
{solution.prompt}

### **Task Context**
**Design Task:** {design_task}

**Domain:** {domain_description}

## Generated Image

![{sol_id}]({relative_img_path})

**Image Path:** `{solution.img_path}`  
**Local Copy:** `{images_dir / img_filename if (images_dir / img_filename).exists() else "No local copy"}`


### Generation Parameters
```json
{metadata.get('generation_params', 'No parameters recorded')}
"""
    
    md_path = markdowns_dir / f"{sol_id}.md"
    md_path.write_text(markdown_content, encoding="utf-8")
    return str(md_path)

def format_archive_addition_info(workflow_output: ArchiveAdditionState, added: bool) -> str:
    if "novelty_check_result" not in workflow_output:
        novelty_check_info = "Status: N/A (no novelty check occurred; perhaps archive was already full)"
    elif workflow_output["novelty_check_result"].is_novel_enough:
        novelty_check_info = "Status: PASS - new img deemed novel enough"
    else:
        novelty_check_info = "Status: FAIL - new img deemed too similar to an archive img"


    if "evaluation_result" not in workflow_output:
        evaluation_info = "Status: N/A (no eval occurred; perhaps archive was not full and img passed novelty check)"
    else:
        if "competing_img_paths" not in workflow_output:
            raise RuntimeError("Evaluation occured, but competing_img_paths missing from final state")
        if "evaluation_result" not in workflow_output:
            raise RuntimeError("Evaluation occured, but evaluation_result missing from final state")
        
        if workflow_output["competing_img_paths"][0] != workflow_output["new_img_path"]:
            competitor_path = workflow_output["competing_img_paths"][0]
        else:
            competitor_path = workflow_output["competing_img_paths"][1]

        if not added:
            evaluation_info = "Status: FAIL"
        else:
            evaluation_info = "Status: PASS"

        result: Evaluation = workflow_output["evaluation_result"]
        evaluation_info += f"\nCompetitor: {competitor_path}" \
            + f"\nReasoning: {result.reasoning}" \
            + f"\nConfidence Level (/10): {result.confidence_level}"

    final_output = f"""
## Novelty Check
{novelty_check_info}

## Evaluation
{evaluation_info}
"""
    return final_output


def save_solution_metadata_json(
    solution: Solution,
    run_id: str,
    solution_count: int,
    design_task: str,
    domain_description: str,
    model_used: str,
    added_to_archive: bool,
    strategy_version: Optional[str],
    base_results_dir: str = "./results",
    **metadata,
) -> str:
    """
    Persist a machine-readable record for `solution` under
    results/<run_id>/artifacts/metadata/sol_###.json

    Returns the absolute path to the JSON file.
    """
    if not strategy_version:
        raise ValueError("Strategy version cannot be None")
    # ---------- target folders ----------
    artifacts_dir  = Path(base_results_dir) / run_id / "artifacts"
    images_dir     = artifacts_dir / "images"
    metadata_dir   = artifacts_dir / "metadata"      # ← assumed present by init_run_dirs
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # ---------- derive filenames ----------
    sol_id        = f"sol_{solution_count:03d}"
    img_filename  = f"{sol_id}.png"
    local_img     = images_dir / img_filename
    local_img_str = str(local_img) if local_img.exists() else None

    # ---------- build JSON structure ----------
    timestamp = datetime.now().isoformat()

    record: dict = {
        "solution_id":      sol_id,
        "timestamp":        timestamp,
        "run_id":           run_id,
        "strategy_version": strategy_version,
        "model_used":       model_used,
        "added_to_archive": added_to_archive,

        "design_task":      design_task,
        "domain_description": domain_description,

        "idea":   solution.idea,
        "prompt": solution.prompt,

        "image_path":       solution.img_path,
        "local_image_copy": local_img_str,

        "generation_params": metadata.get("generation_params"),
    }

    # include any other free-form metadata fields that were passed
    for k, v in metadata.items():
        if k != "generation_params":
            record[k] = v

    # ---------- write file ----------
    json_path = metadata_dir / f"{sol_id}.json"
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    return str(json_path)




def init_run_dirs(base_results_dir: str):
    base = Path(base_results_dir)
    emb = base / "artifacts" / "embeddings"
    imgs = base / "artifacts" / "images"
    prompts = base / "artifacts" / "prompts"
    markdowns = base / "artifacts" / "markdowns"
    metadata = base / "artifacts" / "metadata"
    for p in (emb, imgs, prompts, markdowns, metadata):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "embeddings": emb,
        "images": imgs,
        "prompts": prompts,
        "markdowns": markdowns,
        "metadata": metadata,
        "population": base / "population_data.jsonl",
        "novelty": base / "novelty_metrics.jsonl"
    }


def save_prompt_text(run_id: str, sol_id: str, prompt: str, base_results_dir: str = "./results"):
    """
    Persist the prompt/blueprint to artifacts/prompts/{sol_id}.txt for a given run.
    """
    prompts_dir = Path(base_results_dir) / run_id / "artifacts" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    with open(prompts_dir / f"{sol_id}.txt", "w", encoding="utf-8") as f:
        f.write(prompt)



