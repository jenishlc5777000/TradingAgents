"""FastAPI backend for the React UI.

Run with:  uvicorn api:app --reload --port 8000
Model/provider settings come from .env (TRADINGAGENTS_* variables).
"""

import datetime as dt
import threading
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402

ANALYSTS = ("market", "social", "news", "fundamentals")

app = FastAPI(title="TradingAgents API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, dict] = {}


class RunRequest(BaseModel):
    ticker: str = Field(min_length=1)
    date: str = Field(default_factory=lambda: dt.date.today().isoformat())
    analysts: list[str] = Field(default_factory=lambda: list(ANALYSTS))


def _run(job_id: str, req: RunRequest) -> None:
    job = _jobs[job_id]
    try:
        ta = TradingAgentsGraph(
            selected_analysts=tuple(req.analysts), debug=False, config=DEFAULT_CONFIG.copy()
        )
        state, decision = ta.propagate(req.ticker, req.date)
        job["decision"] = decision if isinstance(decision, str) else str(decision)
        job["reports"] = {
            k: v for k, v in (state or {}).items() if isinstance(v, str) and v.strip()
        }
        job["status"] = "done"
    except Exception as exc:  # surface API-key/config errors to the UI
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"


@app.get("/api/config")
def config():
    return {
        "provider": DEFAULT_CONFIG.get("llm_provider"),
        "deep_model": DEFAULT_CONFIG.get("deep_think_llm"),
        "quick_model": DEFAULT_CONFIG.get("quick_think_llm"),
        "analysts": list(ANALYSTS),
    }


@app.post("/api/runs")
def start_run(req: RunRequest):
    bad = [a for a in req.analysts if a not in ANALYSTS]
    if bad or not req.analysts:
        raise HTTPException(400, f"Invalid analysts: {bad or 'none selected'}")
    req.ticker = req.ticker.strip().upper()
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "running", "ticker": req.ticker, "date": req.date}
    threading.Thread(target=_run, args=(job_id, req), daemon=True).start()
    return {"id": job_id}


@app.get("/api/runs/{job_id}")
def get_run(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown run")
    return job
