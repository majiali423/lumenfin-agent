"""Path bootstrap for MCP server scripts (keeps `mcp` pip package importable)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

MCP_LAYER = Path(__file__).resolve().parent
REPO_ROOT = MCP_LAYER.parent
SRC = REPO_ROOT / "src"

for path in (str(SRC), str(MCP_LAYER)):
    if path not in sys.path:
        sys.path.insert(0, path)

logging.basicConfig(level=logging.ERROR, stream=sys.stderr, force=True)
for logger_name in ("mcp", "mcp.server", "mcp.server.fastmcp", "fastmcp"):
    logging.getLogger(logger_name).setLevel(logging.ERROR)
