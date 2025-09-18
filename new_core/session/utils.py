from datetime import datetime
from pathlib import Path


def make_run_dir(results_root: str, run_name: str | None) -> str:
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    base = run_name or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = root / base
    i = 1
    while run_dir.exists():
        run_dir = root / f"{base}_{i:02d}"
        i += 1
    return str(run_dir)