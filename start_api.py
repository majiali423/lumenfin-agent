from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.api.app import create_app
from lumenfin.config import AppConfig
from lumenfin.stdio import configure_stdio_utf8


def main() -> None:
    configure_stdio_utf8()
    config = AppConfig.from_env()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
