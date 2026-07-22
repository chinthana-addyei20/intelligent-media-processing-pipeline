"""
schemas.py
----------
Pydantic v2 schemas defining the shape of every API request and response.

These schemas form the "public contract" of the API and are deliberately
kept separate from the SQLAlchemy models in models.py:
    - `models.py`  -> how data is stored (database concern)
    - `schemas.py` -> how data is exposed over HTTP (API concern)

Keeping them separate means internal-only columns (e.g. `id`, `retry_count`
bookkeeping details) never leak unintentionally into API responses, and
the ORM layer is free to evolve independently of the public API shape.

All schemas use Pydantic v2 syntax (`ConfigDict`, `field_validator`, etc.).
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models import ProcessingStatus


# ---------------------------------------------------------------------------
# Nested analysis result model
# ---------------------------------------------------------------------------
class AnalysisResult(BaseModel):
    """
    Structured container for all image-analysis outputs.

    Nested inside `ResultsResponse` rather than flattened, since it mirrors
    the natural grouping of the data: "metadata about the job" (id, status,
    timestamps) vs. "what the analysis found" (blur, OCR, duplicate flags,
    etc.), and keeps the JSON response self-documenting.
    """

    model_config = ConfigDict(from_attributes=True)

    blur_score: Optional[float] = Field(
        default=None,
        description="Variance of the Laplacian. Lower values indicate a blurrier image.",
    )
    brightness_score: Optional[float] = Field(
        default=None,
        description="Mean pixel brightness on a 0-255 grayscale scale.",
    )
    extracted_text: Optional[str] = Field(
        default=None,
        description="Raw text extracted from the image via EasyOCR.",
    )
    vehicle_number: Optional[str] = Field(
        default=None,
        description="Best-candidate Indian vehicle registration number parsed from extracted_text.",
    )
    vehicle_number_valid: Optional[bool] = Field(
        default=None,
        description=(
            "Whether vehicle_number matches the Indian numberplate format. "
            "Null if no candidate vehicle number was found at all."
        ),
    )
    is_duplicate: Optional[bool] = Field(
        default=None,
        description="True if this image's perceptual hash matches a previously uploaded image.",
    )
    screenshot_detected: Optional[bool] = Field(
        default=None,
        description="True if heuristics suggest the image is a screenshot rather than a camera photo.",
    )
    tampered_suspected: Optional[bool] = Field(
        default=None,
        description="True if heuristics suggest possible tampering/manipulation.",
    )
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Aggregate confidence score in [0.0, 1.0] summarizing overall analysis reliability.",
    )


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------
class UploadResponse(BaseModel):
    """
    Response returned immediately by POST /upload, before background
    processing begins.
    """

    model_config = ConfigDict(from_attributes=True)

    processing_id: str = Field(
        ..., description="UUID identifying this upload; used to poll status/results."
    )
    status: ProcessingStatus = Field(
        ..., description="Initial pipeline status, always 'pending' at upload time."
    )


# ---------------------------------------------------------------------------
# GET /status/{id}
# ---------------------------------------------------------------------------
class StatusResponse(BaseModel):
    """
    Lightweight response for GET /status/{id}, used for polling.

    Intentionally excludes analysis fields even if they already exist
    mid-pipeline. Full results are only meaningful once processing has
    reached a terminal state, and are served separately by /results/{id}.
    """

    model_config = ConfigDict(from_attributes=True)

    processing_id: str = Field(..., description="UUID identifying this upload.")
    status: ProcessingStatus = Field(..., description="Current pipeline status.")
    retry_count: int = Field(
        default=0, description="Number of times processing has been retried after a transient failure."
    )
    created_at: datetime = Field(..., description="Timestamp when the upload was received.")
    updated_at: datetime = Field(..., description="Timestamp of the most recent status change.")


# ---------------------------------------------------------------------------
# GET /results/{id}
# ---------------------------------------------------------------------------
class ResultsResponse(BaseModel):
    """
    Full response for GET /results/{id}, returned only once status is
    COMPLETED. The route layer is responsible for returning a 409-style
    error if results are requested before processing has finished.
    """

    model_config = ConfigDict(from_attributes=True)

    processing_id: str = Field(..., description="UUID identifying this upload.")
    filename: str = Field(..., description="Original filename of the uploaded image.")
    status: ProcessingStatus = Field(..., description="Pipeline status (expected: 'completed').")
    analysis: AnalysisResult = Field(..., description="Structured analysis results.")
    created_at: datetime = Field(..., description="Timestamp when the upload was received.")
    updated_at: datetime = Field(..., description="Timestamp when processing completed.")

    @field_validator("status")
    @classmethod
    def _status_should_be_completed(cls, value: ProcessingStatus) -> ProcessingStatus:
        """
        Defensive validation: ResultsResponse is only ever meant to be
        constructed for completed records. The route layer already
        enforces this before construction; this validator is a second
        line of defense so a programming mistake surfaces immediately as
        a clear Pydantic ValidationError instead of misleading data.
        """
        if value != ProcessingStatus.COMPLETED:
            raise ValueError(
                f"ResultsResponse requires status='completed', got '{value.value}'."
            )
        return value


# ---------------------------------------------------------------------------
# GET /failure/{id}
# ---------------------------------------------------------------------------
class FailureResponse(BaseModel):
    """
    Response for GET /failure/{id}, returned only once status is FAILED.

    Kept as its own schema (rather than reusing ResultsResponse with
    optional fields) since a failure and a success represent genuinely
    different response shapes.
    """

    model_config = ConfigDict(from_attributes=True)

    processing_id: str = Field(..., description="UUID identifying this upload.")
    filename: str = Field(..., description="Original filename of the uploaded image.")
    status: ProcessingStatus = Field(..., description="Pipeline status (expected: 'failed').")
    failure_reason: str = Field(..., description="Human-readable explanation of why processing failed.")
    retry_count: int = Field(..., description="Number of retry attempts made before giving up.")
    created_at: datetime = Field(..., description="Timestamp when the upload was received.")
    updated_at: datetime = Field(..., description="Timestamp when the failure was recorded.")

    @field_validator("status")
    @classmethod
    def _status_should_be_failed(cls, value: ProcessingStatus) -> ProcessingStatus:
        """Defensive validation mirroring ResultsResponse's approach above."""
        if value != ProcessingStatus.FAILED:
            raise ValueError(
                f"FailureResponse requires status='failed', got '{value.value}'."
            )
        return value


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """
    Response for GET /health, used by orchestrators (Docker healthcheck,
    load balancers, uptime monitors) to verify the service is alive and
    can reach its database.
    """

    model_config = ConfigDict(from_attributes=True)

    status: str = Field(..., description="Overall service health, e.g. 'ok' or 'degraded'.")
    database_connected: bool = Field(..., description="Whether a live database connection was verified.")
    timestamp: datetime = Field(..., description="Server time at which the health check was performed.")


# ---------------------------------------------------------------------------
# Generic error response (used across routes for consistent error shape)
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    """
    Standard error envelope returned by all routes on failure (404s, 400s,
    409s, 500s). A single consistent error shape across the whole API
    means client code can write one error-handling path instead of
    special-casing each endpoint's error format.
    """

    detail: str = Field(..., description="Human-readable error message.")