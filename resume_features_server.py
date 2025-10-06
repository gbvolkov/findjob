from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import uvicorn


DEFAULT_HOST = os.environ.get("RESUME_API_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("RESUME_API_PORT", "8000"))
DEFAULT_LOG_LEVEL = os.environ.get("RESUME_API_LOG_LEVEL", "info")


def _parse_bool(value: Optional[str], fallback: bool = False) -> bool:
    if value is None:
        return fallback
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_RELOAD = _parse_bool(os.environ.get("RESUME_API_RELOAD"), fallback=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the Resume Feature Extraction FastAPI service.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind socket host (default: {DEFAULT_HOST!r}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Bind socket port (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Set Uvicorn log verbosity.",
    )
    parser.add_argument(
        "--reload",
        dest="reload",
        action="store_true",
        help="Enable auto-reload (overrides RESUME_API_RELOAD).",
    )
    parser.add_argument(
        "--no-reload",
        dest="reload",
        action="store_false",
        help="Disable auto-reload (overrides RESUME_API_RELOAD).",
    )
    parser.set_defaults(reload=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    python_log_level = args.log_level.upper()
    logging_level = getattr(logging, python_log_level, logging.INFO)
    if args.log_level == "trace":
        logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level)

    reload_enabled = DEFAULT_RELOAD if args.reload is None else args.reload

    logging.getLogger(__name__).info(
        "Starting resume features API on %s:%s (reload=%s)",
        args.host,
        args.port,
        reload_enabled,
    )

    uvicorn.run(
        "job_agent.resume_features_api:app",
        host=args.host,
        port=args.port,
        reload=reload_enabled,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()

