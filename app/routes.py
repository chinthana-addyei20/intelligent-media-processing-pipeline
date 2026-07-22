"""
routes.py
---------
HTTP layer for the Intelligent Media Processing Pipeline.

Defines all five required endpoints on a single APIRouter, using
dependency-injected SQLAlchemy sessions (via `get_db`) and delegating
actual analysis work to `worker.process_image_task` through
FastAPI's BackgroundTasks.
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from app.database import get_db
from app.image_processor import get_upload_path
from app.logger import get_logger
from app.models import ImageRecord, ProcessingStatus
from app.schemas import (
    AnalysisResult,
    ErrorResponse,
    FailureResponse,
    HealthResponse,
    ResultsResponse,
    StatusResponse,
    UploadResponse,
)
from app.worker import process_image_task

logger = get_logger(__name__)
router = APIRouter()

# Upload validation constants.
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def _get_record_or_404(db: Session, processing_id: str) -> ImageRecord:
    """
    Fetch an ImageRecord by processing_id or raise a 404 HTTPException.

    Args:
        db: Active SQLAlchemy session.
        processing_id: UUID identifying the upload.

    Returns:
        ImageRecord: The matching database row.

    Raises:
        HTTPException: 404 if no record exists for the given processing_id.
    """
    record = db.query(ImageRecord).filter(ImageRecord.processing_id == processing_id).first()
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No record found for processing_id '{processing_id}'.",
        )
    return record


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Image file to analyze."),
    db: Session = Depends(get_db),
) -> UploadResponse:
    """
    Accept an image upload, persist it to disk + DB as PENDING, and
    schedule background analysis.

    Validates content-type, non-empty payload, and max file size before
    writing anything to disk. Returns immediately without waiting for
    processing to complete; the background task updates the record's
    status as it progresses.

    Args:
        background_tasks: FastAPI's background task scheduler.
        file: Uploaded image file.
        db: Injected SQLAlchemy session.

    Returns:
        UploadResponse: processing_id and initial 'pending' status.

    Raises:
        HTTPException: 400 for invalid uploads, 500 for storage/DB failures.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(f"Rejected upload: unsupported content type '{file.content_type}'.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds max allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB.",
        )

    processing_id = str(uuid.uuid4())
    original_filename = file.filename or "upload"
    destination_path = get_upload_path(processing_id, original_filename)

    try:
        with open(destination_path, "wb") as out_file:
            out_file.write(contents)
    except OSError as exc:
        logger.error(f"[{processing_id}] Failed to write file to disk: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to store uploaded file."
        ) from exc

    record = ImageRecord(
        processing_id=processing_id,
        filename=original_filename,
        status=ProcessingStatus.PENDING,
    )
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception as exc:
        db.rollback()
        logger.error(f"[{processing_id}] Failed to create DB record: {exc}")
        destination_path.unlink(missing_ok=True)  # clean up orphaned file
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create processing record."
        ) from exc

    # Schedule analysis to run after the response is sent.
    background_tasks.add_task(process_image_task, processing_id)
    logger.info(f"[{processing_id}] Upload accepted ('{original_filename}'); queued for processing.")

    return UploadResponse(processing_id=record.processing_id, status=record.status)


@router.get(
    "/status/{processing_id}",
    response_model=StatusResponse,
    responses={404: {"model": ErrorResponse}},
)
def get_status(processing_id: str, db: Session = Depends(get_db)) -> StatusResponse:
    """
    Return the current lifecycle status of a processing job.

    Args:
        processing_id: UUID identifying the upload.
        db: Injected SQLAlchemy session.

    Returns:
        StatusResponse: Current status, retry count, and timestamps.

    Raises:
        HTTPException: 404 if no such record exists.
    """
    record = _get_record_or_404(db, processing_id)
    return StatusResponse.model_validate(record)


@router.get(
    "/results/{processing_id}",
    response_model=ResultsResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def get_results(processing_id: str, db: Session = Depends(get_db)) -> ResultsResponse:
    """
    Return full analysis results, only once status is COMPLETED.

    Args:
        processing_id: UUID identifying the upload.
        db: Injected SQLAlchemy session.

    Returns:
        ResultsResponse: Full analysis payload.

    Raises:
        HTTPException: 404 if no such record exists; 409 if processing
        has not yet completed.
    """
    record = _get_record_or_404(db, processing_id)
    if record.status != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Results not available. Current status: '{record.status.value}'.",
        )
    analysis = AnalysisResult.model_validate(record)
    return ResultsResponse(
        processing_id=record.processing_id,
        filename=record.filename,
        status=record.status,
        analysis=analysis,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.get(
    "/failure/{processing_id}",
    response_model=FailureResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def get_failure(processing_id: str, db: Session = Depends(get_db)) -> FailureResponse:
    """
    Return failure diagnostics, only once status is FAILED.

    Args:
        processing_id: UUID identifying the upload.
        db: Injected SQLAlchemy session.

    Returns:
        FailureResponse: Failure reason, retry count, and timestamps.

    Raises:
        HTTPException: 404 if no such record exists; 409 if the job did
        not fail (still in progress or completed successfully).
    """
    record = _get_record_or_404(db, processing_id)
    if record.status != ProcessingStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"No failure recorded. Current status: '{record.status.value}'.",
        )
    return FailureResponse(
        processing_id=record.processing_id,
        filename=record.filename,
        status=record.status,
        failure_reason=record.failure_reason or "Unknown failure.",
        retry_count=record.retry_count,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    """
    Report service liveness and database connectivity.

    Executes a trivial `SELECT 1` query to verify the database connection
    is actually usable, rather than just confirming the process is alive.

    Args:
        db: Injected SQLAlchemy session.

    Returns:
        HealthResponse: Overall status, DB connectivity flag, and timestamp.
    """
    db_connected = True
    try:
        db.execute(sa_text("SELECT 1"))
    except Exception as exc:
        logger.error(f"Health check DB connectivity failed: {exc}")
        db_connected = False

    return HealthResponse(
        status="ok" if db_connected else "degraded",
        database_connected=db_connected,
        timestamp=datetime.now(timezone.utc),
    )