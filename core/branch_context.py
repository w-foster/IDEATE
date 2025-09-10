from dataclasses import dataclass, field
from typing import List

@dataclass
class BranchContext:
    parent_run_dir: str                 # completed parent run
    reference_img_paths: List[str]      # absolute paths to selected images
    reference_ideas: List[str]
    new_design_task: str                # revised brief for this branch
    branch_depth: int = 1               # 0 = base run; 1+ = nested branches
    # For multi-branch history (prior branches aggregated explicitly)
    prior_reference_img_paths: List[str] = field(default_factory=list)
    prior_reference_ideas: List[str] = field(default_factory=list)
    prior_branch_texts: List[str] = field(default_factory=list)