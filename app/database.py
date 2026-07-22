"""
database.py
------------
Central place for all SQLAlchemy database configuration.

Responsibilities:
    - Create the SQLAlchemy engine bound to a local SQLite file.
    - Provide a `SessionLocal` factory for creating per-request DB sessions.
    - Provide the declarative `Base` class that all ORM models inherit from.
    - Provide a FastAPI-compatible dependency (`get_db`) for request-scoped sessions.
    - Provide an `init_db()` helper to create tables at application startup.

Design decisions:
    - SQLite is used instead of PostgreSQL/MySQL for this take-home assignment
      because it requires zero external infrastructure (no separate DB
      container, no network config) while still exercising the full
      SQLAlchemy ORM/ORM-session pattern used in production systems.
      See README "Trade-Offs" section for the full justification.
    - `check_same_thread=False` is required because SQLite by default only
      allows the connection to be used from the thread that created it.
      FastAPI's `BackgroundTasks` run on a separate worker thread from the
      request thread, so without this flag SQLite would raise
      `ProgrammingError: SQLite objects created in a thread can only be
      used in that same thread`.
    - A single shared `engine` is created at import time (module-level
      singleton), which is standard SQLAlchemy practice — engines are
      expensive to create and are meant to be reused for the lifetime of
      the application.
    - Sessions are NOT shared across requests/threads. Each request (and
      each background task) gets its own `Session` instance via
      `SessionLocal()`, which is the safe, idiomatic SQLAlchemy pattern.
"""

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ---------------------------------------------------------------------------
# Database file location
# ---------------------------------------------------------------------------
# The SQLite database file is stored at the project root (one level above
# the `app/` package) so it persists alongside `uploads/` and can be easily
# mounted as a Docker volume for data persistence across container restarts.
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_FILE: str = os.path.join(BASE_DIR, "media_pipeline.db")
SQLALCHEMY_DATABASE_URL: str = f"sqlite:///{DATABASE_FILE}"

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
# `connect_args={"check_same_thread": False}` is SQLite-specific and is
# required because FastAPI BackgroundTasks execute on a different thread
# than the one that handled the incoming HTTP request.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,  # set to True temporarily for verbose SQL query debugging
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
# `autocommit=False` and `autoflush=False` are the recommended defaults for
# web applications: we want explicit control over when data is flushed and
# committed (typically at the end of a request or background task), rather
# than SQLAlchemy silently flushing mid-operation.
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------
# All ORM models (see models.py) inherit from this Base class. It maintains
# the mapping between Python classes and database tables.
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session scoped to a single
    request, and guarantees it is closed afterwards (even if an exception
    is raised while handling the request).

    Usage:
        @app.get("/example")
        def example(db: Session = Depends(get_db)):
            ...

    Yields:
        Session: A SQLAlchemy session bound to the SQLite engine.
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Create all database tables based on the models registered against
    `Base.metadata`.

    This is intentionally simple (no migration framework like Alembic) as
    it fits the scope of a take-home assignment. It is called once at
    application startup (see main.py's startup event).

    Note:
        This function must be called AFTER all model modules have been
        imported, otherwise `Base.metadata` will not know about their
        table definitions. main.py enforces this ordering by importing
        `models` before calling `init_db()`.
    """
    Base.metadata.create_all(bind=engine)