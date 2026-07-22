"""
worker.py
---------
Background processing pipeline invoked via FastAPI BackgroundTasks.

Owns the state machine: PENDING -> PROCESSING -> COMPLETED / FAILED.
Combines the pure functions in image_processor.py with database access
(duplicate lookup, status/result persistence) and retry handling, each
attempt running in its own SQLAlchemy session.
"""

from typing import Optional

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import ImageRecord, ProcessingStatus
from app.image_processor import (
    analyze_image,
    compute_confidence_score,
    get_upload_path,
    ImageProcessingError,
)
from app.logger import get_logger

logger = get_logger(__name__)

# Total attempts = MAX_RETRIES + 1 (i.e. 1 retry = 2 attempts total).
MAX_RETRIES: int = 1


def _check_duplicate(db: Session, image_hash: Optional[str], processing_id: str) -> bool:
    """
    Return True if another record already has this image_hash.

    Kept in worker.py (not image_processor.py) because it requires a
    database query, whereas image_processor.py is intentionally kept
    free of database dependencies.

    Args:
        db: Active SQLAlchemy session.
        image_hash: Perceptual hash computed for the current image.
        processing_id: The current record's processing_id, excluded from
            the lookup so a record never matches itself.

    Returns:
        bool: True if a different record shares this hash.
    """
    if not image_hash:
        return False
    existing = (
        db.query(ImageRecord)
        .filter(ImageRecord.image_hash == image_hash, ImageRecord.processing_id != processing_id)
        .first()
    )
    return existing is not None


def process_image_task(processing_id: str) -> None:
    """
    Entry point registered via `background_tasks.add_task()`.

    Runs the pipeline once, retrying up to MAX_RETRIES times on failure
    before giving up and marking the record FAILED. Each attempt is
    delegated to `_run_attempt`, which manages its own DB session.

    Args:
        processing_id: UUID of the ImageRecord to process.
    """
    for attempt in range(MAX_RETRIES + 1):
        completed = _run_attempt(processing_id, attempt)
        if completed:
            return
        if attempt < MAX_RETRIES:
            logger.warning(f"[{processing_id}] Retrying (attempt {attempt + 2}/{MAX_RETRIES + 1}).")
    logger.error(f"[{processing_id}] All attempts exhausted; record marked FAILED.")


def _run_attempt(processing_id: str, attempt: int) -> bool:
    """
    Execute a single processing attempt in its own DB session.

    Args:
        processing_id: UUID of the ImageRecord to process.
        attempt: Zero-based attempt index (0 = first try, 1 = first retry).

    Returns:
        bool: True on success (record set to COMPLETED). False on failure
        for this attempt (record set to PENDING if retries remain, else
        FAILED with failure_reason populated).
    """
    db: Session = SessionLocal()
    try:
        record = db.query(ImageRecord).filter(ImageRecord.processing_id == processing_id).first()
        if record is None:
            logger.error(f"[{processing_id}] Record not found; aborting.")
            return True  # nothing to retry

        # Mark as actively processing before doing any heavy work, so a
        # concurrent /status poll reflects reality.
        record.status = ProcessingStatus.PROCESSING
        record.retry_count = attempt
        db.commit()
        logger.info(f"[{processing_id}] Processing started (attempt {attempt + 1}).")

        image_path = get_upload_path(record.processing_id, record.filename)
        analysis = analyze_image(str(image_path), record.filename)

        is_duplicate = _check_duplicate(db, analysis["image_hash"], processing_id)
        confidence = compute_confidence_score(
            blur_score=analysis["blur_score"],
            brightness_score=analysis["brightness_score"],
            extracted_text=analysis["extracted_text"],
            vehicle_number_valid=analysis["vehicle_number_valid"],
            is_duplicate=is_duplicate,
            screenshot_detected=analysis["screenshot_detected"],
            tampered_suspected=analysis["tampered_suspected"],
        )

        # Persist all analysis results onto the record.
        record.image_hash = analysis["image_hash"]
        record.blur_score = analysis["blur_score"]
        record.brightness_score = analysis["brightness_score"]
        record.extracted_text = analysis["extracted_text"]
        record.vehicle_number = analysis["vehicle_number"]
        record.vehicle_number_valid = analysis["vehicle_number_valid"]
        record.is_duplicate = is_duplicate
        record.screenshot_detected = analysis["screenshot_detected"]
        record.tampered_suspected = analysis["tampered_suspected"]
        record.confidence_score = confidence
        record.failure_reason = None
        record.status = ProcessingStatus.COMPLETED
        db.commit()
        logger.info(f"[{processing_id}] Completed. confidence={confidence}")
        return True

    except Exception as exc:  # noqa: BLE001 - broad by design, this is the pipeline's failure boundary
        db.rollback()
        logger.exception(f"[{processing_id}] Attempt {attempt + 1} failed: {exc}")
        record = db.query(ImageRecord).filter(ImageRecord.processing_id == processing_id).first()
        if record is not None:
            if attempt >= MAX_RETRIES:
                record.status = ProcessingStatus.FAILED
                record.failure_reason = f"{type(exc).__name__}: {exc}"
            else:
                record.status = ProcessingStatus.PENDING  # eligible for next attempt
            record.retry_count = attempt
            db.commit()
        return False

    finally:
        db.close()