"""
logger.py
---------
Centralized logging configuration for the Intelligent Media Processing
Pipeline.

Responsibilities:
    - Configure a single, application-wide logging setup (console + file).
    - Provide a `get_logger(name)` factory so every module logs under its
      own name (e.g. "app.worker", "app.routes") while sharing the same
      handlers/formatting.
    - Ensure logs remain readable and correctly attributed even when
      emitted from FastAPI's BackgroundTasks, which run on a separate
      worker thread from the request thread.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# ---------------------------------------------------------------------------
# Log file location
# ---------------------------------------------------------------------------
# Logs are written to a `logs/` directory at the project root (sibling to
# `app/`, `uploads/`, etc.), mirroring how `database.py` locates the SQLite
# file relative to the project root rather than the current working
# directory. This makes logging behave consistently regardless of where
# the process is launched from (locally vs inside Docker).
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR: str = os.path.join(BASE_DIR, "logs")
LOG_FILE: str = os.path.join(LOG_DIR, "app.log")

os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Log format
# ---------------------------------------------------------------------------
# Format explicitly includes:
#   - asctime    -> timestamp, essential for reconstructing the order of
#                   events across concurrent request/background threads.
#   - levelname  -> log severity (INFO/WARNING/ERROR/etc.)
#   - name       -> the logger's name, set to the module's `__name__`
#                   wherever `get_logger()` is called, so every line is
#                   traceable to the exact module that emitted it
#                   (e.g. "app.worker" vs "app.routes").
#   - threadName -> BackgroundTasks execute on a separate thread from the
#                   request thread. Including threadName distinguishes
#                   "this log came from the HTTP request handler" vs
#                   "this log came from the background processing worker"
#                   when reading interleaved log output.
#   - message    -> the actual log message.
LOG_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(threadName)s | %(message)s"
)
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def _build_root_logger() -> logging.Logger:
    """
    Configure and return the application's root logger ("app"), attaching
    both a console handler and a rotating file handler exactly once.

    A rotating file handler (rather than a plain FileHandler) is used
    deliberately: without rotation, a long-running service would grow
    app.log unboundedly, eventually exhausting disk space. Capping at
    5MB with 3 backups keeps a useful amount of history (~20MB total)
    without unbounded growth.

    Returns:
        logging.Logger: The configured "app" logger, which all module-level
        loggers (via get_logger) become children of.
    """
    root_logger = logging.getLogger("app")

    # Guard against duplicate handlers. Without this check, if this module
    # were imported multiple times (e.g. once by main.py, once by a test
    # module, once by uvicorn's reloader) each import would re-attach
    # handlers, causing every log line to be printed multiple times.
    if root_logger.handlers:
        return root_logger

    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # -- Console handler --------------------------------------------------
    # Writes to stdout rather than stderr so that in Docker, `docker logs`
    # shows application output in the expected stream, and so log level
    # doesn't get visually flagged as an error by tooling that treats
    # stderr specially.
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # -- Rotating file handler ---------------------------------------------
    # maxBytes=5MB, backupCount=3 -> app.log, app.log.1, app.log.2, app.log.3
    # A reasonable default for a take-home/small-service context; in a
    # larger production deployment this would typically be replaced by
    # shipping logs to a centralized system (e.g. ELK, CloudWatch,
    # Datadog) instead of relying on local rotation.
    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Prevent log records from propagating to Python's root logger (the
    # unnamed logger at the top of the hierarchy). Without this, if any
    # third-party library (e.g. uvicorn) also attaches a handler to the
    # root logger, our messages could be printed twice.
    root_logger.propagate = False

    return root_logger


# Build the root "app" logger once, at module import time.
_build_root_logger()


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-scoped logger that shares the application's configured
    handlers (console + rotating file).

    Every module in the project should call this at the top of the file:

        from app.logger import get_logger
        logger = get_logger(__name__)

    Passing `__name__` (e.g. "app.worker", "app.routes") means log output
    is automatically attributed to the correct module, and namespaces
    cleanly under the shared "app" logger configured above (since
    `__name__` inside the `app` package resolves to "app.<module>", which
    Python's logging hierarchy treats as a child of the "app" logger and
    therefore inherits its handlers/formatting).

    Args:
        name: Typically `__name__` of the calling module.

    Returns:
        logging.Logger: A logger instance ready for use.
    """
    return logging.getLogger(name)


# A default, ready-to-use logger for this module and for simple scripts
# (e.g. quick debugging) that don't want to call get_logger(__name__)
# themselves.
logger = get_logger(__name__)