"""
models.py
---------
SQLAlchemy ORM models for the Intelligent Media Processing Pipeline.

Currently contains a single table, `ImageRecord`, which stores:
    - Upload metadata (filename, hash, timestamps)
    - Pipeline state (status, failure_reason)
    - All image-analysis results (blur, brightness, OCR, vehicle number,
      duplicate/screenshot/tampering flags, confidence score)
"""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    Integer,
    String,
    Text,
)

from app.database import Base


class ProcessingStatus(str, enum.Enum):
    """
    Enumerates every valid state in the background processing lifecycle.

    Inheriting from `str` as well as `enum.Enum` means values serialize
    cleanly to plain strings in JSON responses (e.g. "pending" instead of
    "ProcessingStatus.PENDING"), while still giving us type-safe,
    autocompletable constants in Python code.

    State machine:
        PENDING -> PROCESSING -> COMPLETED
                              \\-> FAILED
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


def _utcnow() -> datetime:
    """
    Return the current UTC time.

    Used as the `default`/`onupdate` callable for timestamp columns.
    Wrapping `datetime.now(timezone.utc)` in a small function (rather than
    passing `datetime.utcnow` directly) gives us a timezone-AWARE
    timestamp instead of a naive one, avoiding ambiguity when the API
    serializes it to ISO-8601 in JSON responses.
    """
    return datetime.now(timezone.utc)


class ImageRecord(Base):
    """
    ORM model representing a single uploaded image and the full result of
    its analysis pipeline.

    One row is created per upload (at POST /upload time, status=PENDING)
    and is then updated in-place by the background worker as processing
    progresses through PROCESSING -> COMPLETED/FAILED.
    """

    __tablename__ = "image_records"

    # -- Primary key ---------------------------------------------------
    # A surrogate auto-incrementing integer primary key, kept separate
    # from `processing_id` (see below) since exposing an auto-incrementing
    # key via the API would leak information (e.g. total upload count)
    # and allow enumeration by clients.
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # -- Public-facing identifier ---------------------------------------
    # UUID4 generated at upload time and returned to the client as
    # "processing_id". Used to look up records via /status, /results,
    # and /failure. Indexed + unique since it is the primary lookup key
    # for nearly every read query in the app.
    processing_id = Column(
        String(36),
        unique=True,
        index=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
    )

    # -- Upload metadata --------------------------------------------------
    filename = Column(String(255), nullable=False)

    # Perceptual hash (via `imagehash`) of the image content, used for
    # duplicate detection. Indexed because duplicate-detection queries
    # look up existing rows by `image_hash` on every new upload.
    image_hash = Column(String(64), index=True, nullable=True)

    # -- Pipeline state -----------------------------------------------
    # Stored as a native SQL ENUM rather than a free-text string so the
    # database itself constrains the column to valid lifecycle values.
    # Indexed since status is frequently filtered on.
    status = Column(
        Enum(ProcessingStatus),
        default=ProcessingStatus.PENDING,
        nullable=False,
        index=True,
    )

    # -- Analysis results (all nullable: populated only once processing
    # reaches/completes the relevant stage; remain NULL if the pipeline
    # fails before reaching that stage) -----------------------------------

    # Variance of the Laplacian — lower values indicate a blurrier image.
    blur_score = Column(Float, nullable=True)

    # Mean pixel brightness (0-255 grayscale scale).
    brightness_score = Column(Float, nullable=True)

    # Raw text extracted via EasyOCR. Stored as `Text` (unbounded) rather
    # than `String` since OCR output length is unpredictable.
    extracted_text = Column(Text, nullable=True)

    # Best-candidate Indian vehicle registration number parsed out of
    # `extracted_text` via regex, e.g. "KA05MH1234".
    vehicle_number = Column(String(20), nullable=True)

    # Whether `vehicle_number` matches the official Indian numberplate
    # format. Nullable (not defaulted to False) so we can distinguish
    # "no vehicle number was found at all" (NULL) from "one was found but
    # is invalid" (False).
    vehicle_number_valid = Column(Boolean, nullable=True)

    # True if `image_hash` collides with a previously-uploaded image.
    is_duplicate = Column(Boolean, nullable=True, default=False)

    # Heuristic screenshot detection (aspect ratio, EXIF metadata,
    # filename hints — see image_processor.py).
    screenshot_detected = Column(Boolean, nullable=True, default=False)

    # Heuristic tampering/manipulation suspicion flag (JPEG ELA-based).
    tampered_suspected = Column(Boolean, nullable=True, default=False)

    # Aggregate confidence score in [0.0, 1.0] summarizing how trustworthy
    # the overall analysis result is (see image_processor.py for the
    # weighting formula).
    confidence_score = Column(Float, nullable=True)

    # -- Failure diagnostics --------------------------------------------
    # Human-readable explanation populated only when status == FAILED.
    failure_reason = Column(Text, nullable=True)

    # -- Retry bookkeeping ------------------------------------------------
    # Tracks how many times processing has been retried after a transient
    # failure (see worker.py's retry mechanism). Defaults to 0.
    retry_count = Column(Integer, nullable=False, default=0)

    # -- Timestamps -------------------------------------------------------
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:
        """Developer-friendly representation, useful in logs/debugging."""
        return (
            f"<ImageRecord(processing_id={self.processing_id!r}, "
            f"status={self.status!r}, filename={self.filename!r})>"
        )
