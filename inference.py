from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")


@dataclass
class BaselineResult:
    task_id: str
    score: float
    reward_trace: list[float]


def _healthcheck(base_url: str) -> bool:
    try:
        with httpx.Client(base_url=base_url, timeout=2.0) as client:
            resp = client.get("/health")
            return resp.status_code == 200
    except Exception:
        return False


def _ensure_env_server(base_url: str) -> subprocess.Popen[str] | None:
    if _healthcheck(base_url):
        return None

    local_hosts = (
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    )
    if base_url not in local_hosts:
        raise RuntimeError(
            f"Environment unreachable at {base_url}. Set ENV_BASE_URL to a live environment."
        )

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "ticket_triage_env.server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    for _ in range(30):
        if _healthcheck(base_url):
            return proc
        time.sleep(0.25)

    proc.terminate()
    raise RuntimeError("Failed to start local environment server for inference.")


def llm_json_plan(ticket: dict[str, Any]) -> dict[str, Any]:
    if HF_TOKEN and API_BASE_URL:
        client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
        prompt = (
            "You are an expert support ticket classifier. Analyze this ticket carefully.\n\n"
            f"Ticket ID: {ticket.get('ticket_id', 'unknown')}\n"
            f"Title: {ticket.get('title', '')}\n"
            f"Description: {ticket.get('description', '')}\n"
            f"Customer Tier: {ticket.get('customer_tier', 'free')}\n\n"
            "Classification rules:\n"
            "- CRITICAL priority: security breaches, data leaks, system down, prod outages (ETA: 1-2h)\n"
            "- HIGH priority: payment failures, billing issues, premium customer problems (ETA: 2-8h)\n"
            "- MEDIUM priority: feature requests, account issues, non-urgent bugs (ETA: 24-48h)\n"
            "- LOW priority: questions, documentation, low-impact issues (ETA: 72h+)\n\n"
            "Team assignment:\n"
            "- security: auth, tokens, suspicious activity, data breaches, privacy\n"
            "- billing: payments, invoices, subscriptions, refunds, charges\n"
            "- infra: server errors, API issues, performance, deployments\n"
            "- support: general questions, account help, features, onboarding\n\n"
            "Tags should be specific (e.g., 'payment-failed', 'prod', 'privacy', 'api', 'customer-impact').\n"
            "Premium/enterprise customers get higher priority and shorter ETA.\n\n"
            "Return ONLY valid JSON with keys: priority, team, eta_hours, tags (array of strings)."
        )
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        data = json.loads(content)
        return {
            "priority": data.get("priority", "medium"),
            "team": data.get("team", "support"),
            "eta_hours": int(data.get("eta_hours", 24)),
            "tags": list(data.get("tags", [])),
        }

    title = ticket.get("title", "").lower()
    desc = ticket.get("description", "").lower()
    tier = ticket.get("customer_tier", "free").lower()
    text = f"{title} {desc}"

    # Premium/enterprise customers get priority boost
    is_premium = tier in ("premium", "enterprise")

    if (
        "token" in text
        or "security" in text
        or "suspicious" in text
        or "breach" in text
    ):
        return {
            "priority": "critical",
            "team": "security",
            "eta_hours": 1,
            "tags": ["security", "incident", "prod", "privacy"],
        }
    if "charge" in text or "invoice" in text or "payment" in text or "billing" in text:
        return {
            "priority": "high" if is_premium else "medium",
            "team": "billing",
            "eta_hours": 2 if is_premium else 4,
            "tags": ["payment", "customer-impact"]
            + (["premium"] if is_premium else []),
        }
    if "api" in text or "500" in text or "error" in text or "down" in text:
        return {
            "priority": "high" if is_premium else "medium",
            "team": "infra",
            "eta_hours": 4 if is_premium else 12,
            "tags": ["api", "prod", "customer-impact"]
            + (["premium"] if is_premium else []),
        }
    return {
        "priority": "medium" if is_premium else "low",
        "team": "support",
        "eta_hours": 12 if is_premium else 24,
        "tags": ["account", "customer-impact"] + (["premium"] if is_premium else []),
    }


def run_one_task(client: httpx.Client, task_id: str) -> BaselineResult:
    reset = client.post("/reset", json={"task_id": task_id}, timeout=30.0)
    reset.raise_for_status()
    obs = reset.json()["observation"]

    plan = llm_json_plan(
        {
            "ticket_id": obs["ticket_id"],
            "title": obs["title"],
            "description": obs["description"],
            "customer_tier": obs["customer_tier"],
        }
    )

    reward_trace: list[float] = []
    actions = (
        [
            {"action_type": "set_priority", "value": plan["priority"]},
            {"action_type": "set_team", "value": plan["team"]},
            {"action_type": "set_eta", "value": str(plan["eta_hours"])},
        ]
        + [{"action_type": "add_tag", "value": tag} for tag in plan["tags"]]
        + [{"action_type": "submit", "value": None}]
    )

    done = False
    for action in actions:
        if done:
            break
        step = client.post("/step", json=action, timeout=30.0)
        step.raise_for_status()
        payload = step.json()
        reward_trace.append(float(payload.get("reward", 0.0)))
        done = bool(payload.get("done", False))

    grade = client.post(
        f"/grade/{task_id}",
        json={
            "priority": plan["priority"],
            "team": plan["team"],
            "eta_hours": plan["eta_hours"],
            "tags": plan["tags"],
        },
        timeout=30.0,
    )
    grade.raise_for_status()
    score = float(grade.json()["score"])
    return BaselineResult(task_id=task_id, score=score, reward_trace=reward_trace)


def main() -> None:
    server_proc = _ensure_env_server(ENV_BASE_URL)
    try:
        with httpx.Client(base_url=ENV_BASE_URL) as client:
            health = client.get("/health", timeout=10.0)
            health.raise_for_status()

            tasks_resp = client.get("/tasks", timeout=10.0)
            tasks_resp.raise_for_status()
            tasks = tasks_resp.json()

            results: list[BaselineResult] = []
            for task in tasks:
                results.append(run_one_task(client, task["task_id"]))

        avg_score = sum(r.score for r in results) / max(len(results), 1)
        output = {
            "api_base_url": API_BASE_URL,
            "env_base_url": ENV_BASE_URL,
            "model_name": MODEL_NAME,
            "task_results": [
                {"task_id": r.task_id, "score": r.score, "reward_trace": r.reward_trace}
                for r in results
            ],
            "average_score": avg_score,
        }
        print(json.dumps(output, indent=2))
    finally:
        if server_proc is not None:
            server_proc.terminate()


if __name__ == "__main__":
    main()
