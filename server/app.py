from __future__ import annotations

import uvicorn

from ticket_triage_env.server.app import app as triage_app

app = triage_app


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run("server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
