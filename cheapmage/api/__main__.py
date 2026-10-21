"""``python -m cheapmage.api`` - run the service with uvicorn."""
from __future__ import annotations

import logging

import uvicorn

from ..config import get_config_store


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    cfg = get_config_store().config
    host = cfg.api.host
    port = int(cfg.api.port)
    workers = 1
    print(f"[cheapmage] serving on http://{host}:{port} (docs: /docs)")
    uvicorn.run(
        "cheapmage.api.app:app",
        host=host,
        port=port,
        log_level="info",
        workers=workers,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
