# Ticket Triage OpenEnv (Round 1)

A real-world OpenEnv environment where an agent triages customer support incidents by choosing:
- `priority` (`low|medium|high|critical`)
- `team` (`billing|support|platform|security`)
- `eta_hours` (positive integer)
- `tags` (string list)

This environment implements the required `step(action) -> observation, reward, done, info`, `reset()`, and `state()` interaction pattern over HTTP and includes 3 graded tasks (easy, medium, hard) with scores in `[0.0, 1.0]`.

## Why this is real-world

Ticket triage is a production workflow in SaaS/support/security operations. The agent must make structured decisions under uncertainty with partial feedback and strict escalation rules.

## Project structure

- `openenv.yaml` - OpenEnv manifest
- `ticket_triage_env/models.py` - typed Pydantic models for action/observation/reward/state
- `ticket_triage_env/server/environment.py` - environment logic + reward shaping + grader
- `ticket_triage_env/server/app.py` - FastAPI endpoints
- `inference.py` - baseline inference script (OpenAI client + required env vars)
- `Dockerfile` - Hugging Face Spaces compatible container build
- `validate.py` - local validation helper
- `scripts/validate-submission.sh` - pre-submission validator script

## OpenEnv API

- `POST /reset` -> initial `observation`, `reward`, `done`, `info`
- `POST /step` -> `observation`, `reward`, `done`, `info`
- `GET /state` -> current environment state
- `GET /tasks` -> task catalog
- `POST /grade/{task_id}` -> grader score in `[0.0, 1.0]`
- `GET /health` -> health check

## Action space

```json
{ "action_type": "set_priority", "value": "high" }
{ "action_type": "set_team", "value": "billing" }
{ "action_type": "set_eta", "value": "4" }
{ "action_type": "add_tag", "value": "payment" }
{ "action_type": "clear_tags", "value": null }
{ "action_type": "submit", "value": null }
```

## Observation space

Each step returns:

```json
{
  "observation": {
    "ticket_id": "INC-2207",
    "customer_tier": "premium",
    "title": "Duplicate invoice charged twice",
    "description": "...",
    "selected_priority": "high",
    "selected_team": "billing",
    "selected_eta_hours": 4,
    "selected_tags": ["payment", "premium", "customer-impact"],
    "allowed_priorities": ["low", "medium", "high", "critical"],
    "allowed_teams": ["billing", "support", "platform", "security"],
    "message": "Action applied."
  },
  "reward": 0.24,
  "done": false,
  "info": {
    "current_score": 0.76,
    "task_id": "medium_duplicate_charge",
    "difficulty": "medium"
  }
}
```

## Reward function

Reward is dense and shaped as score improvement over previous step:

- Priority match weight: `0.35`
- Team match weight: `0.25`
- ETA closeness weight: `0.15`
- Tag Jaccard similarity weight: `0.25`

Final score is clamped to `[0.0, 1.0]`.

## Tasks + graders

Task set:
1. `easy_password_reset` (easy)
2. `medium_duplicate_charge` (medium)
3. `hard_token_leak` (hard)

Each task has a deterministic rubric and agent grader via `POST /grade/{task_id}`.

## Local run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

## Baseline inference

`inference.py` is at repository root and uses OpenAI client with required environment variables:

- `API_BASE_URL` - LLM API endpoint (OpenAI-compatible)
- `MODEL_NAME` - model identifier
- `HF_TOKEN` - Hugging Face/API key

Environment server variable used by this project:
- `ENV_BASE_URL` - environment endpoint for `/reset`, `/step`, `/state` (defaults to `http://127.0.0.1:8000`)

Run:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token_here"
export ENV_BASE_URL="http://127.0.0.1:8000"
.venv/bin/python inference.py
```

If `HF_TOKEN` or `API_BASE_URL` is absent, script falls back to deterministic heuristics for reproducible local validation.

## Validation

```bash
.venv/bin/python validate.py
.venv/bin/openenv validate
./scripts/validate-submission.sh https://your-space.hf.space .
```

## Docker

```bash
docker build -t ticket-triage-openenv .
docker run --rm -p 8000:8000 ticket-triage-openenv
```

## Hugging Face Spaces deploy notes

- Use Docker Space
- Ensure app serves on `0.0.0.0:8000`
- Health endpoint: `/health`
- Reset endpoint: `/reset`
- Root-level `openenv.yaml` included

The service returns HTTP `200` on health and supports reset/step/state operations.
