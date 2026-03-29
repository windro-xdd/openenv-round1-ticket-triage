from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TriageAction(BaseModel):
    """Action schema for ticket triage decisions."""

    action_type: Literal[
        "set_priority",
        "set_team",
        "set_eta",
        "add_tag",
        "clear_tags",
        "submit",
    ]
    value: str | None = None


class TriageObservation(BaseModel):
    """Observation returned after each action."""

    ticket_id: str
    customer_tier: str
    title: str
    description: str
    selected_priority: str | None = None
    selected_team: str | None = None
    selected_eta_hours: int | None = None
    selected_tags: list[str] = Field(default_factory=list)
    allowed_priorities: list[str] = Field(
        default_factory=lambda: ["low", "medium", "high", "critical"]
    )
    allowed_teams: list[str] = Field(
        default_factory=lambda: ["billing", "support", "platform", "security"]
    )
    message: str = ""


class TriageReward(BaseModel):
    """Reward model for each transition."""

    value: float


class TriageState(BaseModel):
    """State metadata for the running episode."""

    episode_id: str
    task_id: str
    task_name: str
    step_count: int
    done: bool
    cumulative_reward: float
    action_history: list[dict[str, Any]] = Field(default_factory=list)


class StepResult(BaseModel):
    observation: TriageObservation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task_id: str | None = None


class GradeProposal(BaseModel):
    priority: str | None = None
    team: str | None = None
    eta_hours: int | None = None
    tags: list[str] = Field(default_factory=list)
