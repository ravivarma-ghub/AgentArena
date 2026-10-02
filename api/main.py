import sys
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import validate_config, DEFAULT_POPULATION_SIZE, DEFAULT_NUM_GENERATIONS
from arena.controller import ArenaController
from memory.store import ArenaMemoryStore

app = FastAPI(
    title="Agent Arena API",
    description="Competitive & Evolutionary Multi-Agent Problem-Solving System API",
    version="1.0.0"
)

memory = ArenaMemoryStore()
active_runs: Dict[str, Dict[str, Any]] = {}


class TournamentRequest(BaseModel):
    task: str = Field(..., example="Compare the latest open-source LLMs and recommend one for a startup.")
    num_agents: int = Field(default=DEFAULT_POPULATION_SIZE, ge=2, le=30)
    num_generations: int = Field(default=DEFAULT_NUM_GENERATIONS, ge=1, le=4)
    num_judges: int = Field(default=2, ge=1, le=5)
    enable_peer_evaluation: bool = True
    enable_dynamic_tool_assignment: bool = True


@app.get("/health")
def health_check():
    status = validate_config()
    return {
        "status": "healthy" if status["groq_configured"] else "unconfigured",
        "config": status
    }


def _run_tournament_task(run_id: str, req: TournamentRequest):
    controller = ArenaController(
        num_judges=req.num_judges,
        peer_sample_k=2 if req.enable_peer_evaluation else 0
    )
    active_runs[run_id] = {"status": "running", "progress": 0.0, "message": "Starting tournament..."}

    def progress_callback(msg: str, pct: float, data=None):
        active_runs[run_id]["progress"] = round(pct, 2)
        active_runs[run_id]["message"] = msg
        if data:
            active_runs[run_id]["result"] = data

    try:
        result = controller.run_tournament(
            task=req.task,
            num_agents=req.num_agents,
            num_generations=req.num_generations,
            enable_peer_evaluation=req.enable_peer_evaluation,
            enable_dynamic_tool_assignment=req.enable_dynamic_tool_assignment,
            progress_callback=progress_callback
        )
        active_runs[run_id]["status"] = "completed"
        active_runs[run_id]["result"] = result
    except Exception as e:
        active_runs[run_id]["status"] = "failed"
        active_runs[run_id]["error"] = str(e)


@app.post("/arena/run")
def start_tournament(req: TournamentRequest, background_tasks: BackgroundTasks):
    import uuid
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    active_runs[run_id] = {"status": "queued", "progress": 0.0, "message": "Queued"}
    background_tasks.add_task(_run_tournament_task, run_id, req)
    return {"run_id": run_id, "status": "queued", "message": "Tournament started in background."}


@app.get("/arena/status/{run_id}")
def get_run_status(run_id: str):
    if run_id in active_runs:
        return active_runs[run_id]
    loaded = memory.load_run(run_id)
    if loaded:
        return {"status": "completed", "result": loaded}
    raise HTTPException(status_code=404, detail="Run ID not found")


@app.get("/arena/runs")
def list_runs():
    return memory.list_runs()
