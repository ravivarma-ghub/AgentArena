import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional


class ArenaMemoryStore:
    """Manages persistence of tournament sessions, generations, and agent artifacts."""

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = Path(storage_dir or Path(__file__).resolve().parent.parent / "runs")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, run_id: str, run_data: Dict[str, Any]) -> str:
        """Save tournament run data to disk."""
        filepath = self.storage_dir / f"{run_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2, default=str)
        return str(filepath)

    def load_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load tournament run data from disk."""
        filepath = self.storage_dir / f"{run_id}.json"
        if not filepath.exists():
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_runs(self) -> List[Dict[str, Any]]:
        """List summary of all completed runs."""
        runs = []
        for file in sorted(self.storage_dir.glob("*.json"), reverse=True):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    runs.append({
                        "run_id": data.get("run_id", file.stem),
                        "task": data.get("task", "")[:60] + "...",
                        "timestamp": data.get("timestamp", 0),
                        "generations": len(data.get("generation_history", {})),
                        "winner": data.get("final_arbiter", {}).get("top_candidates", [{}])[0].get("agent_name", "N/A"),
                        "filepath": str(file)
                    })
            except Exception:
                pass
        return runs
