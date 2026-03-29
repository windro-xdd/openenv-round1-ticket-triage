from __future__ import annotations

import uvicorn


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run("ticket_triage_env.server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
