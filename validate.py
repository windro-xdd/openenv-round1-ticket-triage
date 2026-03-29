from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

import httpx
import yaml

ROOT = pathlib.Path(__file__).resolve().parent
MANIFEST = ROOT / "openenv.yaml"
BASE_URL = "http://127.0.0.1:8000"


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
    raise SystemExit("Failed to start local environment server for validation.")


def main() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text())
    required = ["spec_version", "name", "type", "runtime", "app", "port"]
    missing = [k for k in required if k not in manifest]
    if missing:
        raise SystemExit(f"Missing fields in openenv.yaml: {missing}")

    server_proc = _ensure_env_server(BASE_URL)
    try:
        with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
            health = client.get("/health")
            health.raise_for_status()

            reset = client.post("/reset", json={"task_id": "easy_password_reset"})
            reset.raise_for_status()
            reset_payload = reset.json()
            for key in ["observation", "reward", "done", "info"]:
                if key not in reset_payload:
                    raise SystemExit(f"/reset missing key: {key}")

            state = client.get("/state")
            state.raise_for_status()

            tasks = client.get("/tasks")
            tasks.raise_for_status()
            task_list = tasks.json()
            if len(task_list) < 3:
                raise SystemExit("Expected at least 3 tasks")

            for task in task_list:
                grade = client.post(
                    f"/grade/{task['task_id']}",
                    json={
                        "priority": "medium",
                        "team": "support",
                        "eta_hours": 24,
                        "tags": ["account"],
                    },
                )
                grade.raise_for_status()
                score = grade.json()["score"]
                if not (0.0 <= score <= 1.0):
                    raise SystemExit(
                        f"Invalid grader score for {task['task_id']}: {score}"
                    )

        print(json.dumps({"status": "ok", "manifest": manifest["name"]}, indent=2))
    finally:
        if server_proc is not None:
            server_proc.terminate()


if __name__ == "__main__":
    main()
