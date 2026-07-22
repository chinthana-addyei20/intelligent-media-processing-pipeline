"""
image_processor.py
-------------------
Pure image-analysis logic for the Intelligent Media Processing Pipeline.

This module intentionally has NO knowledge of the database or HTTP layer.
It exposes small, single-purpose functions that each perform one analysis
step, plus a top-level `analyze_image()` orchestrator that runs all of them
against a single image file and returns a plain dict of results.

Kept database-free on purpose (clean architecture): `worker.py` is the only
module that combines these results with SQLAlchemy sessions. The one
exception is duplicate detection — `compute_image_hash()` lives here (it's
a pure image operation), but the actual "does this hash already exist?"
lookup is a database query and therefore lives in worker.py, not here.

Implemented analysis steps:
    - Blur detection            (OpenCV Laplacian variance)
    - Brightness analysis       (mean grayscale pixel intensity)
    - OCR                       (EasyOCR)
    - Indian vehicle number     (regex extraction + format validation)
    - Perceptual image hashing  (imagehash, for duplicate detection)
    - Screenshot heuristics     (aspect ratio, EXIF presence, filename)
    - Tampering heuristics      (JPEG Error Level Analysis)
    - Confidence score          (weighted aggregate, 0.0-1.0)
"""

import io
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import easyocr
import imagehash
import numpy as np
from PIL import Image, ImageChops

from app.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class ImageProcessingError(Exception):
    """
    Raised for non-recoverable failures in the analysis pipeline — e.g. the
    file cannot be opened as an image at all.

    Distinguishing this from a generic Exception lets worker.py log a
    clearer failure_reason and lets us reason explicitly about which
    failures are "fatal" (file unreadable) vs. "degrade gracefully"
    (a single heuristic, like OCR or tamper detection, failing while the
    rest of the pipeline still produces useful output).
    """


# ---------------------------------------------------------------------------
# Upload path helper (shared by routes.py and worker.py)
# ---------------------------------------------------------------------------
# Computed the same way as database.py's BASE_DIR: two directories above
# this file (app/image_processor.py -> app/ -> project root), so the
# uploads directory resolves consistently regardless of the working
# directory the process was launched from (local vs Docker).
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR: Path = Path(BASE_DIR) / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_upload_path(processing_id: str, filename: str) -> Path:
    """
    Deterministically compute the on-disk path for an uploaded image.

    Prefixing the stored filename with `processing_id` guarantees no two
    uploads collide on disk, even if two clients upload files with the
    identical original name (e.g. "photo.jpg"). Because the path is
    derived purely from `processing_id` + `filename` (both of which are
    persisted in the database), any module can reconstruct the file's
    location later without storing a separate "file path" column.

    Args:
        processing_id: The UUID assigned to this upload.
        filename: The original filename provided by the client.

    Returns:
        Path: Absolute path where the file is/should be stored.
    """
    # os.path.basename strips any directory components the client might
    # have included in the filename, preventing path traversal
    # (e.g. "../../etc/passwd") from escaping the uploads directory.
    safe_name = os.path.basename(filename)
    return UPLOAD_DIR / f"{processing_id}_{safe_name}"


# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------
# Heuristic constants, not learned/calibrated values — documented here as
# a single source of truth so they're easy to tune without hunting through
# function bodies. See README "Trade-Offs" for why heuristics were chosen
# over a trained model for this assignment.
BLUR_THRESHOLD: float = 100.0          # Laplacian variance below this = blurry
BRIGHTNESS_LOW: float = 50.0           # mean pixel value below this = too dark
BRIGHTNESS_HIGH: float = 200.0         # mean pixel value above this = overexposed
ELA_TAMPER_THRESHOLD: float = 15.0     # mean ELA intensity above this = suspicious
JPEG_ELA_QUALITY: int = 90             # re-compression quality used for ELA


# ---------------------------------------------------------------------------
# Indian vehicle number patterns
# ---------------------------------------------------------------------------
# Standard Indian registration format: SS DD LLL NNNN
#   SS   = 2-letter state code (e.g. KA, MH, DL)
#   DD   = 1-2 digit RTO district code
#   LLL  = 0-3 letter series (often 1-2 letters; occasionally omitted)
#   NNNN = 4-digit unique number
#
# STRICT pattern validates a normalized (no spaces/hyphens) candidate.
STRICT_VEHICLE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")

# SEARCH pattern scans raw OCR text (which often contains stray spaces or
# hyphens between groups, e.g. "KA 05 MH 1234") to find a plate-shaped
# substring before normalizing and strictly validating it.
SEARCH_VEHICLE_PATTERN = re.compile(
    r"[A-Z]{2}[\s-]?[0-9]{1,2}[\s-]?[A-Z]{1,3}[\s-]?[0-9]{4}"
)

# LOOSE fallback: if no properly-shaped candidate is found, look for any
# 8-11 character alphanumeric token that mixes letters and digits. This
# lets us report "we found something that might be a plate, but it doesn't
# validate" (vehicle_number_valid=False) instead of silently reporting
# nothing at all.
LOOSE_CANDIDATE_PATTERN = re.compile(r"\b[A-Z0-9]{8,11}\b")


# ---------------------------------------------------------------------------
# OCR reader (lazy singleton)
# ---------------------------------------------------------------------------
# EasyOCR's Reader() constructor loads neural network weights into memory
# and is expensive (multi-second) to instantiate. Creating it once per
# process (module-level singleton, lazily built on first use) rather than
# once per request/image is essential for reasonable throughput — recreating
# it for every uploaded image would make each request far slower than the
# other analysis steps combined.
_ocr_reader: Optional["easyocr.Reader"] = None


def _get_ocr_reader() -> "easyocr.Reader":
    """
    Return the process-wide EasyOCR reader, creating it on first use.

    gpu=False is used because this project targets a plain Docker
    container without guaranteed GPU access — CPU inference is slower but
    portable, which matters more for a take-home reviewer running
    `docker-compose up --build` on an arbitrary machine.
    """
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("Initializing EasyOCR reader (English, CPU) — first call only.")
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


# ---------------------------------------------------------------------------
# Individual analysis functions
# ---------------------------------------------------------------------------
def compute_blur_score(gray_image: np.ndarray) -> float:
    """
    Compute a blur score using the variance of the Laplacian.

    The Laplacian operator highlights regions of rapid intensity change
    (edges). A sharp, in-focus image has many strong edges and therefore
    high variance; a blurry image has few/weak edges and low variance.
    This is a well-established, cheap blur-detection heuristic that avoids
    needing a trained model.

    Args:
        gray_image: Single-channel (grayscale) image array.

    Returns:
        float: Laplacian variance. Higher = sharper. See BLUR_THRESHOLD.

    Raises:
        ImageProcessingError: If the Laplacian computation fails.
    """
    try:
        variance = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        return float(variance)
    except Exception as exc:  # pragma: no cover - defensive
        raise ImageProcessingError(f"Blur detection failed: {exc}") from exc


def compute_brightness(gray_image: np.ndarray) -> float:
    """
    Compute mean pixel brightness on a 0-255 grayscale scale.

    A simple mean is used rather than a perceptual luminance formula
    (e.g. weighted RGB) since the image is already converted to grayscale
    upstream by `analyze_image()`, keeping this function trivial and fast.

    Args:
        gray_image: Single-channel (grayscale) image array.

    Returns:
        float: Mean pixel intensity in [0, 255].
    """
    return float(np.mean(gray_image))


def extract_text(image_path: str) -> str:
    """
    Run OCR on the given image and return all detected text, concatenated.

    OCR is treated as a "best effort" step: if it fails (corrupted image,
    EasyOCR internal error, unsupported format edge case) we log a warning
    and return an empty string rather than raising, so a single flaky OCR
    call doesn't fail the entire pipeline for an otherwise valid image.

    Args:
        image_path: Path to the image file on disk.

    Returns:
        str: All recognized text, space-joined. Empty string if none found
        or OCR failed.
    """
    try:
        reader = _get_ocr_reader()
        # detail=0 returns a plain list of recognized text strings instead
        # of (bounding_box, text, confidence) tuples, since we don't need
        # per-word positions/confidences for this pipeline.
        results = reader.readtext(image_path, detail=0)
        return " ".join(results).strip()
    except Exception as exc:
        logger.warning(f"OCR extraction failed for '{image_path}': {exc}")
        return ""


def validate_vehicle_number(text: str) -> Tuple[Optional[str], Optional[bool]]:
    """
    Search OCR text for an Indian vehicle registration number and validate
    its format.

    Three possible outcomes:
        1. A well-formed candidate is found -> (number, True)
        2. A plate-shaped candidate is found but fails strict validation,
           or only a loose alphanumeric candidate is found -> (number, False)
        3. Nothing resembling a plate is found -> (None, None)

    Distinguishing (None, None) from (number, False) matters: the former
    means "we don't even have evidence of a vehicle number," while the
    latter means "we found something but it doesn't validate" — useful
    for confidence scoring and for a human reviewer to know which case
    they're looking at.

    Args:
        text: Raw OCR-extracted text (any case/spacing).

    Returns:
        Tuple[Optional[str], Optional[bool]]: (vehicle_number, is_valid).
    """
    if not text:
        return None, None

    normalized_text = text.upper()

    match = SEARCH_VEHICLE_PATTERN.search(normalized_text)
    if match:
        candidate = re.sub(r"[\s-]", "", match.group(0))
        is_valid = bool(STRICT_VEHICLE_PATTERN.match(candidate))
        return candidate, is_valid

    # No plate-shaped match — fall back to a loose alphanumeric candidate
    # so we can still flag "something plate-like was seen, but invalid"
    # rather than reporting nothing.
    compact_text = re.sub(r"[\s-]", "", normalized_text)
    loose_match = LOOSE_CANDIDATE_PATTERN.search(compact_text)
    if loose_match:
        return loose_match.group(0), False

    return None, None


def compute_image_hash(image_path: str) -> Optional[str]:
    """
    Compute a perceptual hash (pHash) of the image for duplicate detection.

    Perceptual hashing (via the `imagehash` library) is used instead of a
    cryptographic hash (e.g. MD5/SHA256) because we want to detect
    near-duplicate images (same photo re-saved, slightly re-compressed, or
    minor edits) as duplicates too — a cryptographic hash would only catch
    byte-for-byte identical files.

    Args:
        image_path: Path to the image file on disk.

    Returns:
        Optional[str]: Hex string representation of the perceptual hash,
        or None if hashing failed (e.g. unreadable file).
    """
    try:
        with Image.open(image_path) as img:
            hash_value = imagehash.phash(img)
        return str(hash_value)
    except Exception as exc:
        logger.warning(f"Image hashing failed for '{image_path}': {exc}")
        return None


def detect_screenshot(image_path: str, filename: str) -> bool:
    """
    Heuristically determine whether an image is likely a screenshot rather
    than a camera photo.

    Combines three independent, individually weak signals into a simple
    point-based score, requiring at least 2 points before flagging as a
    screenshot (reduces false positives from any single weak signal):
        - Filename hints at "screenshot" (+2, strong signal)
        - Aspect ratio matches a common device/monitor screen ratio (+1)
        - No EXIF metadata present (+1) — camera photos almost always
          carry EXIF data (camera model, exposure, etc.); most screenshot
          tools don't write any.

    This is explicitly a heuristic, not a classifier — see README
    "Trade-Offs" for why ML-based classification was out of scope.

    Args:
        image_path: Path to the image file on disk.
        filename: Original filename provided at upload time.

    Returns:
        bool: True if heuristics suggest this is a screenshot.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            # Pillow only exposes _getexif() on JPEG-derived plugins; use
            # getattr defensively so PNGs/other formats don't raise.
            exif_getter = getattr(img, "_getexif", None)
            has_exif = bool(exif_getter()) if exif_getter else False

        aspect_ratio = round(width / height, 2) if height else 0.0
        # Common landscape screen/monitor ratios: 16:9, 4:3, ~19.5:9, 16:10, 5:4
        common_ratios = [1.78, 1.33, 2.17, 1.6, 1.25]
        ratio_hint = any(
            abs(aspect_ratio - ratio) < 0.05 or abs(aspect_ratio - (1 / ratio)) < 0.05
            for ratio in common_ratios
        )

        filename_hint = bool(
            re.search(r"screen\s*shot|scrnshot|screencapture", filename, re.IGNORECASE)
        )

        score = 0
        if filename_hint:
            score += 2
        if ratio_hint:
            score += 1
        if not has_exif:
            score += 1

        return score >= 2
    except Exception as exc:
        logger.warning(f"Screenshot heuristic failed for '{image_path}': {exc}")
        return False


def detect_tampering(image_path: str) -> bool:
    """
    Heuristically flag possible image tampering using JPEG Error Level
    Analysis (ELA).

    ELA works by re-saving the image at a known JPEG quality and computing
    the pixel-wise difference against the original. Regions that have been
    digitally edited (pasted, cloned, retouched) were typically
    compressed at a different quality/generation than the rest of the
    image, so they "light up" with a different error level under
    re-compression than untouched regions. A high *mean* error level
    across the whole image is used here as a simple, coarse proxy for
    "this image shows compression-inconsistency signs" — genuine forensic
    ELA tooling additionally inspects the *spatial distribution* of error
    levels (looking for localized hot spots), which is out of scope for
    this heuristic implementation.

    Args:
        image_path: Path to the image file on disk.

    Returns:
        bool: True if the mean ELA intensity exceeds ELA_TAMPER_THRESHOLD.
    """
    try:
        original = Image.open(image_path).convert("RGB")

        # Re-compress in-memory at a fixed JPEG quality.
        buffer = io.BytesIO()
        original.save(buffer, "JPEG", quality=JPEG_ELA_QUALITY)
        buffer.seek(0)
        recompressed = Image.open(buffer)

        ela_image = ImageChops.difference(original, recompressed)

        # Normalize the difference image so its brightest pixel maps to
        # 255, making the mean intensity comparable across images of
        # varying original quality/content.
        extrema = ela_image.getextrema()
        max_diff = max(channel_extrema[1] for channel_extrema in extrema) or 1
        scale = 255.0 / max_diff

        ela_array = np.array(ela_image, dtype=np.float32) * scale
        mean_ela_intensity = float(np.mean(ela_array))

        return mean_ela_intensity > ELA_TAMPER_THRESHOLD
    except Exception as exc:
        logger.warning(f"Tampering heuristic failed for '{image_path}': {exc}")
        return False


def compute_confidence_score(
    blur_score: Optional[float],
    brightness_score: Optional[float],
    extracted_text: Optional[str],
    vehicle_number_valid: Optional[bool],
    is_duplicate: Optional[bool],
    screenshot_detected: Optional[bool],
    tampered_suspected: Optional[bool],
) -> float:
    """
    Compute an aggregate confidence score in [0.0, 1.0] summarizing how
    trustworthy/usable the overall analysis result is.

    Starts at a perfect score of 1.0 and subtracts weighted penalties for
    each undesirable signal detected. Weights are heuristic and roughly
    ordered by how strongly each signal undermines trust in the image:
    tampering is the most severe (0.30), followed by duplication (0.20),
    blur (0.25), invalid vehicle number / poor lighting / missing text /
    screenshot flag (0.10-0.15 each). The final score is clamped to
    [0.0, 1.0].

    Args:
        blur_score: Laplacian variance from compute_blur_score, or None.
        brightness_score: Mean brightness from compute_brightness, or None.
        extracted_text: OCR output (empty string counts as "no text").
        vehicle_number_valid: True/False/None from validate_vehicle_number.
        is_duplicate: Whether this image matches a previously seen hash.
        screenshot_detected: Screenshot heuristic result.
        tampered_suspected: Tampering heuristic result.

    Returns:
        float: Confidence score rounded to 2 decimal places, in [0.0, 1.0].
    """
    score = 1.0
    deductions: List[Tuple[str, float]] = []

    if blur_score is not None and blur_score < BLUR_THRESHOLD:
        deductions.append(("blurry image", 0.25))

    if brightness_score is not None and (
        brightness_score < BRIGHTNESS_LOW or brightness_score > BRIGHTNESS_HIGH
    ):
        deductions.append(("poor lighting (too dark/bright)", 0.15))

    if not extracted_text:
        deductions.append(("no text extracted via OCR", 0.10))

    if vehicle_number_valid is False:
        deductions.append(("vehicle number found but invalid format", 0.15))

    if is_duplicate:
        deductions.append(("duplicate of a previously seen image", 0.20))

    if screenshot_detected:
        deductions.append(("screenshot detected", 0.10))

    if tampered_suspected:
        deductions.append(("possible tampering detected", 0.30))

    total_deduction = sum(weight for _, weight in deductions)
    final_score = max(0.0, min(1.0, score - total_deduction))

    if deductions:
        logger.debug(f"Confidence deductions applied: {deductions} -> score={final_score:.2f}")

    return round(final_score, 2)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def analyze_image(image_path: str, filename: str) -> Dict[str, Any]:
    """
    Run the full analysis pipeline (minus duplicate detection and
    confidence scoring, which require database access / cross-step
    results and are therefore finalized by worker.py) against a single
    image.

    Failure semantics:
        - If the file is missing or cannot be decoded as an image at all,
          this raises ImageProcessingError — this is a fatal condition
          the worker should record as a failed job (and potentially
          retry, in case of a transient disk issue).
        - Individual heuristic steps (OCR, screenshot detection, tamper
          detection, hashing) catch their own exceptions internally and
          degrade gracefully (returning "", False, or None) rather than
          aborting the whole pipeline — a flaky heuristic on one image
          shouldn't fail an otherwise-successful analysis.

    Args:
        image_path: Path to the image file on disk.
        filename: Original filename (used for screenshot heuristics).

    Returns:
        Dict[str, Any]: Keys — blur_score, brightness_score,
        extracted_text, vehicle_number, vehicle_number_valid, image_hash,
        screenshot_detected, tampered_suspected.

    Raises:
        ImageProcessingError: If the image file cannot be read at all.
    """
    if not os.path.exists(image_path):
        raise ImageProcessingError(f"Image file not found at path: {image_path}")

    color_image = cv2.imread(image_path)
    if color_image is None:
        raise ImageProcessingError(
            f"Unable to read '{filename}' as an image — it may be corrupted "
            "or in an unsupported format."
        )

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    blur_score = compute_blur_score(gray_image)
    brightness_score = compute_brightness(gray_image)

    extracted_text = extract_text(image_path)
    vehicle_number, vehicle_number_valid = validate_vehicle_number(extracted_text)

    image_hash = compute_image_hash(image_path)
    screenshot_detected = detect_screenshot(image_path, filename)
    tampered_suspected = detect_tampering(image_path)

    logger.info(
        f"Analysis complete for '{filename}': "
        f"blur={blur_score:.2f}, brightness={brightness_score:.2f}, "
        f"text_len={len(extracted_text)}, vehicle_number={vehicle_number}, "
        f"screenshot={screenshot_detected}, tampered={tampered_suspected}"
    )

    return {
        "blur_score": round(blur_score, 2),
        "brightness_score": round(brightness_score, 2),
        "extracted_text": extracted_text,
        "vehicle_number": vehicle_number,
        "vehicle_number_valid": vehicle_number_valid,
        "image_hash": image_hash,
        "screenshot_detected": screenshot_detected,
        "tampered_suspected": tampered_suspected,
    }