from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json
import base64, requests

from new_core.interfaces.run_repository import IRunRepository
from new_core.models.image_solution import ImageSolution

class FSRunRepository(IRunRepository):
    def __init__(self) -> None:
        pass

    def init_layout(self, run_dir: str) -> None:
        base = Path(run_dir)
        for p in (
            base / "artifacts" / "embeddings",
            base / "artifacts" / "images",
            base / "artifacts" / "prompts",
            base / "artifacts" / "markdowns",
            base / "artifacts" / "metadata",
        ):
            p.mkdir(parents=True, exist_ok=True)

    def save_config(self, run_dir: str, cfg: Dict[str, Any]) -> None:
        p = Path(run_dir) / "config.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def save_guardrails(self, run_dir: str, design_task: str, domain_description: str, guardrails: str) -> None:
        payload = {
            "design_task": design_task,
            "domain_description": domain_description,
            "guardrails": guardrails,
        }
        p = Path(run_dir) / "guardrails.json"
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def append_strategy_version(
        self,
        run_dir: str,
        strategy_id: str,
        strategy_text: str,
        previous_archive_analysis: Optional[str],
    ) -> None:
        entry = {
            "strategy_id": strategy_id,
            "strategy": strategy_text,
            "timestamp": datetime.now().isoformat(),
            "previous_archive_analysis": previous_archive_analysis,
        }
        self._append_jsonl(Path(run_dir) / "strategy_versions.jsonl", entry)

    def append_population(self, run_dir: str, generation: int, blueprint_ids: List[str]) -> None:
        entry = {
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
            "blueprint_ids": blueprint_ids,
            "count": len(blueprint_ids),
        }
        self._append_jsonl(Path(run_dir) / "population_data.jsonl", entry)

    def append_novelty_metrics(self, run_dir: str, entry: Dict[str, Any]) -> None:
        self._append_jsonl(Path(run_dir) / "novelty_metrics.jsonl", entry)

    def save_prompt_text(self, run_dir: str, sol_id: str, prompt_text: str) -> None:
        prompts_dir = Path(run_dir) / "artifacts" / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (prompts_dir / f"{sol_id}.txt").write_text(prompt_text, encoding="utf-8")

    def write_solution_markdown(
        self,
        run_dir: str,
        solution: ImageSolution,
        solution_count: int,
        design_task: str,
        domain_description: str,
        model_used: str,
        added_to_archive: bool,
        addition_decision_reasoning: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Mirrors core.utils.create_solution_markdown style exactly:
        - artifacts/markdowns/sol_###.md
        - Table, sections, same field names and layout.
        """
        artifacts_dir = Path(run_dir) / "artifacts"
        markdowns_dir = artifacts_dir / "markdowns"
        images_dir = artifacts_dir / "images"
        markdowns_dir.mkdir(parents=True, exist_ok=True)

        sol_id = f"sol_{solution_count:03d}"
        img_filename = f"{sol_id}.png"
        relative_img_path = f"../images/{img_filename}" if (images_dir / img_filename).exists() \
                            else (solution.img_path or "No image available")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        addition_info = self._format_archive_addition_info(addition_decision_reasoning, added_to_archive)

        md = f"""# Solution {sol_id}

## Overview
| Parameter      | Value          |
|----------------|----------------|
| Solution ID    | {sol_id} 
| Timestamp      | {timestamp} 
| Model Used     | {model_used} 
| Run ID         | {Path(run_dir).name} 

{addition_info}


## Creative Details

### **Idea**
{getattr(solution.idea, "text", str(solution.idea))}

### **Prompt**
{getattr(solution.prompt, "text", str(solution.prompt))}

### **Task Context**
**Design Task:** {design_task}

**Domain:** {domain_description}

## Generated Image

![{sol_id}]({relative_img_path})

**Image Path:** `{solution.img_path}`  
**Local Copy:** `{images_dir / img_filename if (images_dir / img_filename).exists() else "No local copy"}`


### Generation Parameters
```json
{json.dumps(solution.metadata.get("generation_params", "No parameters recorded")) if isinstance(solution.metadata, dict) else "No parameters recorded"}
"""
        out = markdowns_dir / f"{sol_id}.md"
        out.write_text(md, encoding="utf-8")
        return str(out)

    def write_solution_metadata_json(
        self,
        run_dir: str,
        solution: ImageSolution,
        solution_count: int,
        design_task: str,
        domain_description: str,
        model_used: str,
        added_to_archive: bool,
        strategy_version: str,
        addition_decision_reasoning: Optional[Dict[str, Any]] = None,
        generation_params: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Mirrors core.utils.save_solution_metadata_json exactly.
        """
        artifacts_dir = Path(run_dir) / "artifacts"
        images_dir = artifacts_dir / "images"
        metadata_dir = artifacts_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        sol_id = f"sol_{solution_count:03d}"
        img_filename = f"{sol_id}.png"
        local_img = images_dir / img_filename
        local_img_str = str(local_img) if local_img.exists() else None

        record: Dict[str, Any] = {
            "solution_id": sol_id,
            "timestamp": datetime.now().isoformat(),
            "run_id": Path(run_dir).name,
            "strategy_version": strategy_version,
            "model_used": model_used,
            "added_to_archive": added_to_archive,
            "design_task": design_task,
            "domain_description": domain_description,
            "idea": getattr(solution.idea, "text", str(solution.idea)),
            "prompt": getattr(solution.prompt, "text", str(solution.prompt)),
            "image_path": solution.img_path,
            "local_image_copy": local_img_str,
            "generation_params": generation_params,
        }
        if addition_decision_reasoning is not None:
            record["addition_decision_reasoning"] = addition_decision_reasoning
        if extra:
            for k, v in extra.items():
                if k != "generation_params":
                    record[k] = v

        json_path = metadata_dir / f"{sol_id}.json"
        json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return str(json_path)

    # INTERNAL HELPERS ------------

    @staticmethod
    def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    @staticmethod
    def _format_archive_addition_info(addition_workflow_output: Optional[Dict[str, Any]], added: bool) -> str:
        """
        Dump the archive addition details without assuming any schema.
        If details are absent, emit the legacy placeholder line.
        """
        if not addition_workflow_output:
            return "NO ARCHIVE ADDITION INFO AVAILABLE (likely because this was the first solution)"

        # Pretty-print the entire structure as JSON so downstream tools/humans can inspect it.
        try:
            payload = json.dumps(addition_workflow_output, indent=2)
        except Exception:
            payload = str(addition_workflow_output)

        return f"""
## Archive Addition Details
```json
{payload}
```
""".strip()

    def save_image_from_url(self, run_dir: str, sol_id: str, sample_url: str) -> str:
        images_dir = Path(run_dir) / "artifacts" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        out = images_dir / f"{sol_id}.png"
        if sample_url.startswith("data:"):
            _, b64 = sample_url.split(",", 1)
            out.write_bytes(base64.b64decode(b64))
        else:
            r = requests.get(sample_url, timeout=10)
            r.raise_for_status()
            out.write_bytes(r.content)
        return str(out)