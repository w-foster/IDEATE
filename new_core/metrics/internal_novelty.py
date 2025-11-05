from pathlib import Path
from typing import Sequence, Dict, Any, List, Optional
from datetime import datetime
import numpy as np

from new_core.models.image_solution import ImageSolution

def _load_embeddings(run_dir: str, solutions: Sequence[ImageSolution]) -> tuple[Optional[np.ndarray], list[ImageSolution]]:
    emb_dir = Path(run_dir) / "artifacts" / "embeddings"
    E_list: list[np.ndarray] = []
    sols_kept: list[ImageSolution] = []
    for sol in solutions:
        p = emb_dir / f"{sol.id}.npy"
        if not p.exists():
            continue
        arr = np.load(p)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        E_list.append(arr.astype(np.float32, copy=False))
        sols_kept.append(sol)
    if not E_list:
        return None, []
    return np.stack(E_list), sols_kept

def _avg_knn_distances(E: np.ndarray, k_neighbors: int) -> list[float]:
    # L2-normalize
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    En = E / norms
    # cosine distances
    sim = En @ En.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, np.inf)

    avg_dists: list[float] = []
    for i in range(dist.shape[0]):
        row = np.sort(dist[i])
        k = int(min(k_neighbors, row.shape[0]))
        avg_dists.append(float(np.mean(row[:k])))
    return avg_dists

def build_novelty_entry(
    run_dir: str,
    solutions: Sequence[ImageSolution],
    *,
    k_neighbors: int = 3,
) -> Optional[Dict[str, Any]]:
    E, sols = _load_embeddings(run_dir, solutions)
    if E is None or not sols:
        return None

    avg_dists = _avg_knn_distances(E, k_neighbors=k_neighbors)

    # bucket by strategy id
    strat_to_d: Dict[str, List[float]] = {}
    for sol, d in zip(sols, avg_dists):
        strat = (
            sol.metadata.get("strategy_version")
            or sol.metadata.get("creative_strategy_name")
            or "None"
        )
        strat_to_d.setdefault(strat, []).append(d)

    strategy_metrics: Dict[str, Dict[str, Any]] = {}
    for strat, ds in strat_to_d.items():
        arr = np.array(ds, dtype=np.float32)
        strategy_metrics[strat] = {
            "count": int(arr.size),
            "avg_novelty": float(arr.mean()) if arr.size else 0.0,
            "std_novelty": float(arr.std(ddof=0)) if arr.size else 0.0,
        }

    mean_novelty = float(np.mean(avg_dists))
    # proxy “genome length”: prompt text length
    mean_genome_length = float(
        np.mean([len(getattr(sol.prompt, "text", "") or "") for sol in sols])
    ) if sols else 0.0

    return {
        "timestamp": datetime.now().isoformat(),
        "avg_distance_to_neighbors": avg_dists,
        "mean_novelty": mean_novelty,
        "mean_genome_length": mean_genome_length,
        "strategy_metrics": strategy_metrics,
    }