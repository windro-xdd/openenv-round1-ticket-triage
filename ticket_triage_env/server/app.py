from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from ..models import GradeProposal, ResetRequest, StepResult, TriageAction
from .environment import TASKS, TicketTriageEnvironment


env = TicketTriageEnvironment()
app = FastAPI(title="ticket-triage-openenv", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset", response_model=StepResult)
def reset(payload: ResetRequest | None = None) -> StepResult:
    try:
        obs = env.reset(task_id=(payload.task_id if payload else None))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StepResult(
        observation=obs,
        reward=0.0,
        done=False,
        info={"task_id": env.state().get("task_id"), "current_score": 0.0},
    )


@app.post("/step", response_model=StepResult)
def step(payload: TriageAction) -> StepResult:
    try:
        obs, reward, done, info = env.step(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StepResult(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state() -> dict[str, Any]:
    return env.state()


@app.get("/tasks")
def tasks() -> list[dict[str, Any]]:
    return env.task_catalog()


@app.post("/grade/{task_id}")
def grade(task_id: str, proposal: GradeProposal) -> dict[str, float]:
    try:
        score = env.evaluate_task(task_id, proposal.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"score": score}


@app.get("/meta")
def meta() -> dict[str, Any]:
    return {
        "name": "ticket_triage_env",
        "task_count": len(TASKS),
        "api": ["/reset", "/step", "/state", "/tasks", "/grade/{task_id}", "/health"],
    }
