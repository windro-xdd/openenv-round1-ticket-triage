from __future__ import annotations

import uuid
from typing import Any

from ..models import TriageAction, TriageObservation, TriageState

TASKS: list[dict[str, Any]] = [
    {
        "task_id": "easy_password_reset",
        "task_name": "Easy - Account Access Recovery",
        "difficulty": "easy",
        "ticket": {
            "ticket_id": "INC-1001",
            "customer_tier": "standard",
            "title": "Cannot reset password after lockout",
            "description": "Customer is locked out after multiple failed login attempts.",
        },
        "rubric": {
            "priority": "medium",
            "team": "support",
            "eta_hours": 24,
            "tags": ["account", "customer-impact"],
        },
    },
    {
        "task_id": "medium_duplicate_charge",
        "task_name": "Medium - Billing Escalation",
        "difficulty": "medium",
        "ticket": {
            "ticket_id": "INC-2207",
            "customer_tier": "premium",
            "title": "Duplicate invoice charged twice",
            "description": "Premium customer reports two successful charges for same invoice.",
        },
        "rubric": {
            "priority": "high",
            "team": "billing",
            "eta_hours": 4,
            "tags": ["payment", "premium", "customer-impact"],
        },
    },
    {
        "task_id": "hard_token_leak",
        "task_name": "Hard - Security Incident",
        "difficulty": "hard",
        "ticket": {
            "ticket_id": "SEC-9003",
            "customer_tier": "enterprise",
            "title": "Possible token leak from production admin account",
            "description": "Suspicious API calls from unknown IP used an admin token and accessed sensitive endpoints.",
        },
        "rubric": {
            "priority": "critical",
            "team": "security",
            "eta_hours": 1,
            "tags": ["security", "incident", "prod", "privacy"],
        },
    },
]


class TicketTriageEnvironment:
    def __init__(self) -> None:
        self._task_index = -1
        self._max_steps = 8
        self._state = TriageState(
            episode_id="",
            task_id="",
            task_name="",
            step_count=0,
            done=False,
            cumulative_reward=0.0,
            action_history=[],
        )
        self._current_task: dict[str, Any] | None = None
        self._selection: dict[str, Any] = {
            "priority": None,
            "team": None,
            "eta_hours": None,
            "tags": [],
        }
        self._last_score = 0.0

    def reset(self, task_id: str | None = None) -> TriageObservation:
        if task_id:
            task = next((t for t in TASKS if t["task_id"] == task_id), None)
            if task is None:
                raise ValueError(f"Unknown task_id: {task_id}")
            self._current_task = task
        else:
            self._task_index = (self._task_index + 1) % len(TASKS)
            self._current_task = TASKS[self._task_index]

        self._state = TriageState(
            episode_id=str(uuid.uuid4()),
            task_id=self._current_task["task_id"],
            task_name=self._current_task["task_name"],
            step_count=0,
            done=False,
            cumulative_reward=0.0,
            action_history=[],
        )
        self._selection = {
            "priority": None,
            "team": None,
            "eta_hours": None,
            "tags": [],
        }
        self._last_score = 0.0
        return self._make_observation("Episode reset. Start triaging the ticket.")

    def step(self, action: TriageAction) -> tuple[TriageObservation, float, bool, dict[str, Any]]:
        if self._current_task is None:
            raise RuntimeError("Call reset() before step().")

        if self._state.done:
            return self._make_observation("Episode already done."), 0.0, True, {
                "current_score": self._last_score,
                "task_id": self._state.task_id,
            }

        self._state.step_count += 1
        message = "Action applied."
        reward = 0.0

        if action.action_type == "set_priority":
            self._selection["priority"] = action.value
        elif action.action_type == "set_team":
            self._selection["team"] = action.value
        elif action.action_type == "set_eta":
            try:
                eta = int(action.value or "")
                if eta > 0:
                    self._selection["eta_hours"] = eta
                else:
                    message = "ETA must be > 0."
                    reward = -0.05
            except ValueError:
                message = "ETA must be an integer hour value."
                reward = -0.05
        elif action.action_type == "add_tag":
            tag = (action.value or "").strip().lower()
            if tag and tag not in self._selection["tags"]:
                self._selection["tags"].append(tag)
            else:
                message = "Tag empty or already present."
                reward = -0.02
        elif action.action_type == "clear_tags":
            self._selection["tags"] = []
        elif action.action_type == "submit":
            final = self._score_current_selection()
            reward += max(-1.0, min(1.0, final - self._last_score))
            self._last_score = final
            self._state.done = True
            message = f"Submitted triage. Final grader score: {final:.3f}"

        current_score = self._score_current_selection()
        if action.action_type != "submit":
            shaped_delta = current_score - self._last_score
            reward += shaped_delta
            self._last_score = current_score

        if self._state.step_count >= self._max_steps and not self._state.done:
            self._state.done = True
            message = "Max steps reached. Episode finished."

        self._state.cumulative_reward += reward
        self._state.action_history.append(
            {"action_type": action.action_type, "value": action.value}
        )

        info = {
            "current_score": self._last_score,
            "task_id": self._state.task_id,
            "difficulty": self._current_task["difficulty"],
            "rubric_dimensions": ["priority", "team", "eta_hours", "tags"],
        }
        return self._make_observation(message), reward, self._state.done, info

    def _make_observation(self, message: str) -> TriageObservation:
        assert self._current_task is not None
        ticket = self._current_task["ticket"]
        return TriageObservation(
            ticket_id=ticket["ticket_id"],
            customer_tier=ticket["customer_tier"],
            title=ticket["title"],
            description=ticket["description"],
            selected_priority=self._selection["priority"],
            selected_team=self._selection["team"],
            selected_eta_hours=self._selection["eta_hours"],
            selected_tags=list(self._selection["tags"]),
            message=message,
        )

    def _score_current_selection(self) -> float:
        assert self._current_task is not None
        rubric = self._current_task["rubric"]

        priority_score = 1.0 if self._selection["priority"] == rubric["priority"] else 0.0
        team_score = 1.0 if self._selection["team"] == rubric["team"] else 0.0

        eta_value = self._selection["eta_hours"]
        if eta_value is None:
            eta_score = 0.0
        else:
            diff = abs(eta_value - rubric["eta_hours"])
            eta_score = max(0.0, 1.0 - (diff / max(rubric["eta_hours"], 1)))

        expected_tags = set(rubric["tags"])
        chosen_tags = set(self._selection["tags"])
        if not expected_tags and not chosen_tags:
            tags_score = 1.0
        else:
            intersection = len(expected_tags & chosen_tags)
            union = len(expected_tags | chosen_tags)
            tags_score = intersection / union if union else 0.0

        total = (
            0.35 * priority_score
            + 0.25 * team_score
            + 0.15 * eta_score
            + 0.25 * tags_score
        )
        return max(0.0, min(1.0, total))

    def state(self) -> dict[str, Any]:
        payload = self._state.model_dump()
        payload["current_score"] = self._last_score
        payload["task_difficulty"] = (
            self._current_task.get("difficulty") if self._current_task else "unknown"
        )
        return payload

    def task_catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "task_id": t["task_id"],
                "task_name": t["task_name"],
                "difficulty": t["difficulty"],
                "expected_score_range": [0.0, 1.0],
            }
            for t in TASKS
        ]

    def evaluate_task(self, task_id: str, proposal: dict[str, Any]) -> float:
        task = next((t for t in TASKS if t["task_id"] == task_id), None)
        if task is None:
            raise ValueError(f"Unknown task_id: {task_id}")

        rubric = task["rubric"]
        priority_score = 1.0 if proposal.get("priority") == rubric["priority"] else 0.0
        team_score = 1.0 if proposal.get("team") == rubric["team"] else 0.0

        eta_value = proposal.get("eta_hours")
        if eta_value is None:
            eta_score = 0.0
        else:
            diff = abs(int(eta_value) - rubric["eta_hours"])
            eta_score = max(0.0, 1.0 - (diff / max(rubric["eta_hours"], 1)))

        expected_tags = set(rubric["tags"])
        chosen_tags = set(proposal.get("tags", []))
        intersection = len(expected_tags & chosen_tags)
        union = len(expected_tags | chosen_tags)
        tags_score = intersection / union if union else 0.0

        score = (
            0.35 * priority_score
            + 0.25 * team_score
            + 0.15 * eta_score
            + 0.25 * tags_score
        )
        return max(0.0, min(1.0, score))
