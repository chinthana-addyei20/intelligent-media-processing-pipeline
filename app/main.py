"""
main.py
-------
FastAPI application entry point for the Intelligent Media Processing
Pipeline.

Responsibilities:
    - Instantiate the FastAPI app with title/description/version metadata.
    - On startup: ensure uploads/ and logs/ directories exist, and create
      database tables.
    - Register all API routes from app.routes.
    - Provide a `uvicorn.run(...)` entry point for local execution.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

# `models` must be imported before `init_db()` is called so that all
# ORM classes are registered against `Base.metadata` — otherwise
# `Base.metadata.create_all()` would not know about the `image_records`
# table and would silently create no tables at all.
from app import models  # noqa: F401 - imported for side effect (model registration)
from app.database import init_db
from app.logger import get_logger
from app.routes import router

logger = get_logger(__name__)

# Resolve project-root-relative paths the same way database.py and
# image_processor.py do, so directories land in the same place regardless
# of the working directory the process was launched from.
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT: str = os.path.dirname(BASE_DIR)
UPLOAD_DIR: Path = Path(PROJECT_ROOT) / "uploads"
LOG_DIR: Path = Path(PROJECT_ROOT) / "logs"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Startup/shutdown lifecycle handler (modern replacement for the
    deprecated @app.on_event("startup") decorator).

    On startup: guarantees uploads/ and logs/ exist (defensive — Docker
    also creates them at build time) and creates DB tables if missing.
    """
    logger.info("Starting Intelligent Media Processing Pipeline...")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Verified directories: uploads='{UPLOAD_DIR}', logs='{LOG_DIR}'")

    init_db()
    logger.info("Database tables initialized.")

    yield  # application runs here

    logger.info("Shutting down Intelligent Media Processing Pipeline.")


app = FastAPI(
    title="Intelligent Media Processing Pipeline",
    description=(
        "An asynchronous image analysis service: blur/brightness scoring, "
        "OCR, Indian vehicle number validation, duplicate detection, and "
        "screenshot/tampering heuristics, processed via FastAPI BackgroundTasks."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    """Serve the frontend landing page."""
    return FileResponse(os.path.join(PROJECT_ROOT, "index.html"))


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)