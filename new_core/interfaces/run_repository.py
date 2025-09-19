# new_core/interfaces/run_repository.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from new_core.models.image_solution import ImageSolution
import numpy as np

class IRunRepository(ABC):
    @abstractmethod
    def init_layout(self, run_dir: str) -> None:
        ...

    @abstractmethod
    def save_config(self, run_dir: str, cfg: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def save_guardrails(self, run_dir: str, design_task: str, domain_description: str, guardrails: str) -> None:
        ...

    @abstractmethod
    def append_strategy_version(
        self,
        run_dir: str,
        strategy_id: str,
        strategy_text: str,
        previous_archive_analysis: Optional[str],
    ) -> None:
        ...

    @abstractmethod
    def append_population(self, run_dir: str, generation: int, blueprint_ids: List[str]) -> None:
        ...

    @abstractmethod
    def append_novelty_metrics(self, run_dir: str, entry: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def save_prompt_text(self, run_dir: str, sol_id: str, prompt_text: str) -> None:
        ...
    
    @abstractmethod
    def save_image_from_url(self, run_dir: str, sol_id: str, sample_url: str) -> str:
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...
    
    @abstractmethod
    def save_image_embedding(self, run_dir: str, sol_id: str, emb_np: np.ndarray) -> str:
        ...