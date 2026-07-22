# Intelligent Media Processing Pipeline

An asynchronous FastAPI service that accepts image uploads and runs them
through a multi-stage analysis pipeline — blur/brightness scoring, OCR,
Indian vehicle number validation, duplicate detection, and screenshot /
tampering heuristics — using `BackgroundTasks` and SQLite for state.

---

## 1. Project Overview

A client uploads an image via `POST /upload` and immediately receives a
`processing_id`. Analysis runs asynchronously in the background; the
client polls `GET /status/{id}` and retrieves `GET /results/{id}` (or
`GET /failure/{id}`) once the job reaches a terminal state. This
request/accept-then-poll pattern keeps the upload endpoint fast and
avoids blocking HTTP workers on CPU-heavy OCR/CV work.

**Stack:** FastAPI · Python 3.11 · SQLAlchemy · SQLite · OpenCV · EasyOCR ·
imagehash · Pillow · Docker

---

## 2. Architecture Diagram (ASCII)
┌─────────────────────────────────────────┐
                    │                Client                   │
                    └───────────────────┬───────────────────┘
                                        │ HTTP
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │              FastAPI App                 │
                    │  ┌─────────────────────────────────┐    │
                    │  │  routes.py (APIRouter)           │    │
                    │  │  /upload /status /results         │    │
                    │  │  /failure /health                 │    │
                    │  └───────────┬───────────┬───────────┘    │
                    │              │           │                │
                    │   BackgroundTasks    Depends(get_db)      │
                    │              │           │                │
                    │              ▼           ▼                │
                    │  ┌───────────────┐  ┌──────────────┐     │
                    │  │  worker.py     │  │ database.py   │     │
                    │  │ (pipeline FSM) │  │ (SQLAlchemy)  │     │
                    │  └───────┬───────┘  └──────┬───────┘     │
                    │          │                  │             │
                    │          ▼                  ▼             │
                    │  ┌───────────────┐  ┌──────────────┐     │
                    │  │image_processor │  │  SQLite DB    │     │
                    │  │.py (CV/OCR)    │  │ (image_records│     │
                    │  └───────┬───────┘  │  table)        │     │
                    │          │           └──────────────┘     │
                    │          ▼                                │
                    │  ┌───────────────┐                        │
                    │  │  /uploads dir  │                        │
                    │  └───────────────┘                        │
                    └─────────────────────────────────────────┘
                    ---

## 3. Service Flow

1. Client sends `POST /upload` with an image file.
2. Route validates type/size, saves the file to `/uploads`, creates a
   `PENDING` DB row, and schedules `worker.process_image_task` via
   `BackgroundTasks`.
3. The response (`processing_id`, `status: pending`) is returned
   **immediately** — before analysis starts.
4. FastAPI runs the background task after the response is sent.
5. Client polls `GET /status/{id}` until status is `completed` or
   `failed`, then fetches `GET /results/{id}` or `GET /failure/{id}`.

---

## 4. Processing Flow
Inside `worker.py`, each attempt:
1. Sets status to `processing`.
2. Runs `image_processor.analyze_image()` — blur, brightness, OCR,
   vehicle number, hashing, screenshot/tamper heuristics.
3. Looks up the image hash in the DB to flag duplicates.
4. Computes an aggregate `confidence_score` (0.0–1.0).
5. Commits `completed` with all fields populated, **or** on exception,
   retries once (`pending` again) before finally marking `failed` with a
   `failure_reason`.

---

## 5. Queue Strategy

This project uses **FastAPI's built-in `BackgroundTasks`**, not a
dedicated task queue (Celery/RQ/Redis). Each accepted upload runs its
analysis as an `asyncio`-scheduled task on the same process, right after
the HTTP response is returned. There is no external broker, no separate
worker process, and no task persistence beyond the DB row itself.

This is intentionally the simplest option that satisfies the async
processing requirement — see **Trade-Offs** and **Future Improvements**
for why a real queue (Celery + Redis) would replace this in production.

---

## 6. Design Decisions

- **Separate schemas from models** (`schemas.py` vs `models.py`) so the
  DB layer can evolve independently of the public API contract.
- **UUID as the public identifier**, integer `id` kept internal — avoids
  leaking row counts / enabling enumeration via sequential IDs.
- **`image_processor.py` has no DB dependency** — pure functions only;
  `worker.py` is the sole place DB state and analysis results meet.
- **Graceful degradation for heuristics**: OCR, hashing, screenshot, and
  tamper detection each catch their own exceptions and default to a safe
  value (`""`, `None`, `False`) rather than failing the whole pipeline —
  only a genuinely unreadable image file is treated as fatal.
- **Lazy-loaded, process-wide EasyOCR singleton** — avoids re-loading
  OCR model weights (multi-second cost) on every single upload.
- **Lifespan-based startup** (not the deprecated `@app.on_event`) —
  ensures `uploads/`/`logs/` exist and DB tables are created before the
  app starts accepting traffic.

---

## 7. Database Schema

**Table: `image_records`**

| Column | Type | Notes |
|---|---|---|
| `id` | Integer PK | Internal only, not exposed via API |
| `processing_id` | String(36), unique, indexed | Public UUID identifier |
| `filename` | String(255) | Original uploaded filename |
| `image_hash` | String(64), indexed | Perceptual hash (pHash) |
| `status` | Enum | pending / processing / completed / failed |
| `blur_score` | Float | Laplacian variance |
| `brightness_score` | Float | Mean grayscale intensity |
| `extracted_text` | Text | Raw OCR output |
| `vehicle_number` | String(20) | Best-candidate plate number |
| `vehicle_number_valid` | Boolean | Null = none found; True/False = validated |
| `is_duplicate` | Boolean | Matches a prior image hash |
| `screenshot_detected` | Boolean | Screenshot heuristic result |
| `tampered_suspected` | Boolean | ELA tampering heuristic result |
| `confidence_score` | Float | Aggregate score, 0.0–1.0 |
| `failure_reason` | Text | Populated only when status = failed |
| `retry_count` | Integer | Attempts made (max 1 retry) |
| `created_at` / `updated_at` | DateTime (UTC) | Timestamps |

---

## 8. API Documentation

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload` | Upload an image; returns `processing_id` + `pending` status |
| `GET` | `/status/{processing_id}` | Poll current pipeline status |
| `GET` | `/results/{processing_id}` | Full analysis results (409 if not completed) |
| `GET` | `/failure/{processing_id}` | Failure reason (409 if not failed) |
| `GET` | `/health` | Service + DB connectivity check |

Interactive docs are auto-generated by FastAPI at `/docs` (Swagger UI)
and `/redoc`.

---

## 9. Running Locally

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The app creates `uploads/`, `logs/`, and `media_pipeline.db`
automatically on startup. Visit `http://localhost:8000/docs`.

---

## 10. Docker Instructions

```bash
# One-time: pre-create the SQLite file so Docker bind-mounts a file,
# not a directory, on first run.
touch media_pipeline.db

docker-compose up --build
```

The API is then available at `http://localhost:8000`. `uploads/`,
`logs/`, and `media_pipeline.db` are bind-mounted to the host, so data
persists across `docker-compose down` / `up` cycles.

> **Note:** EasyOCR downloads its model weights (~100MB) on first OCR
> call, requiring internet access the first time the container performs
> OCR.

---

## 11. Sample curl Requests/Responses

**Upload**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@sample_images/auto1.jpg"
```
```json
{
  "processing_id": "b3f1c2e4-9a7d-4e21-8b3a-6d2f1a9c0e12",
  "status": "pending"
}
```

**Status**
```bash
curl http://localhost:8000/status/b3f1c2e4-9a7d-4e21-8b3a-6d2f1a9c0e12
```
```json
{
  "processing_id": "b3f1c2e4-9a7d-4e21-8b3a-6d2f1a9c0e12",
  "status": "completed",
  "retry_count": 0,
  "created_at": "2026-07-21T06:00:00Z",
  "updated_at": "2026-07-21T06:00:03Z"
}
```

**Results**
```bash
curl http://localhost:8000/results/b3f1c2e4-9a7d-4e21-8b3a-6d2f1a9c0e12
```
```json
{
  "processing_id": "b3f1c2e4-9a7d-4e21-8b3a-6d2f1a9c0e12",
  "filename": "auto1.jpg",
  "status": "completed",
  "analysis": {
    "blur_score": 245.67,
    "brightness_score": 128.4,
    "extracted_text": "KA05MH1234",
    "vehicle_number": "KA05MH1234",
    "vehicle_number_valid": true,
    "is_duplicate": false,
    "screenshot_detected": false,
    "tampered_suspected": false,
    "confidence_score": 1.0
  },
  "created_at": "2026-07-21T06:00:00Z",
  "updated_at": "2026-07-21T06:00:03Z"
}
```

**Failure**
```bash
curl http://localhost:8000/failure/b3f1c2e4-9a7d-4e21-8b3a-6d2f1a9c0e12
```
```json
{
  "processing_id": "b3f1c2e4-9a7d-4e21-8b3a-6d2f1a9c0e12",
  "filename": "auto2.jpg",
  "status": "failed",
  "failure_reason": "ImageProcessingError: Unable to read 'auto2.jpg' as an image",
  "retry_count": 1,
  "created_at": "2026-07-21T06:05:00Z",
  "updated_at": "2026-07-21T06:05:02Z"
}
```

**Health**
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "database_connected": true,
  "timestamp": "2026-07-21T06:10:00Z"
}
```

---

## 12. Assumptions

The implementation is based on the following assumptions:

- Uploaded files are valid image formats (JPEG, JPG, or PNG) and are within the configured upload size limit.
- Vehicle number validation verifies only the format using regular expressions and does not check against any official government vehicle registration database.
- Duplicate detection uses perceptual image hashing (pHash), which identifies visually similar images but may not detect every possible duplicate.
- Screenshot and tampering detection are heuristic-based techniques intended to indicate potential issues rather than provide definitive forensic evidence.
- OCR accuracy depends on the quality, orientation, lighting conditions, and readability of the uploaded image.
- FastAPI BackgroundTasks are sufficient for the asynchronous processing requirements of this assignment. A dedicated task queue (such as Celery with Redis) would be more appropriate for large-scale production deployments.
- SQLite is assumed to be sufficient for a single-instance application. A production deployment would typically use PostgreSQL or another client-server database for improved concurrency and scalability.




---

## 13. Trade-Offs

| Choice | Instead of | Why |
|---|---|---|
| **SQLite** | PostgreSQL | Zero external infrastructure — no separate DB container/network config needed for a take-home; the ORM layer (`database.py`) is written to make swapping the connection string straightforward later. |
| **FastAPI `BackgroundTasks`** | Celery + broker | No message broker or separate worker process required to satisfy the async-processing requirement; sufficient for single-instance, moderate-volume workloads. |
| **Heuristic analysis** | Trained ML models | Blur/brightness/screenshot/tamper detection use classical CV heuristics (Laplacian variance, ELA, aspect-ratio/EXIF checks) rather than trained classifiers — faster to build, no training data/GPU needed, fully deterministic and explainable. |
| **Local disk storage** | AWS S3 | Uploaded files are written to a local `/uploads` directory (Docker volume) — no cloud credentials or bucket setup required to run the project. |

---

## 14. Scalability Considerations

- **Background tasks run in-process** — vertical scaling only; horizontal
  scaling (multiple app instances) would cause each instance's
  `BackgroundTasks` to only see uploads it personally received.
- **SQLite has limited write concurrency** — fine for a single instance;
  concurrent writers across multiple processes/containers would need a
  proper client-server database.
- **EasyOCR inference is CPU-bound and the slowest pipeline stage** —
  throughput is currently bounded by however many background tasks can
  run concurrently within one process.
- **Local disk storage doesn't scale across multiple app instances** —
  files uploaded to one container's disk aren't visible to others without
  shared/networked storage.

---

## 15. Failure Handling Strategy

- **Upload-time validation** (content-type, size, empty file) rejects bad
  input before any DB row or background task is created.
- **Fatal vs. non-fatal analysis errors**: an unreadable image file is
  fatal (`ImageProcessingError`) and triggers the retry/failure path;
  individual heuristics (OCR, hashing, screenshot/tamper detection) catch
  their own errors internally and degrade to safe defaults instead of
  failing the whole job.
- **Automatic retry (max 1)**: on a fatal error, the record returns to
  `pending`, `retry_count` increments, and the pipeline runs again in a
  fresh DB session before finally marking `failed` with a
  `failure_reason`.
- **Every DB write is wrapped in try/except with rollback** — a failure
  partway through never leaves the session in a corrupted state.
- **`/results` and `/failure` return `409 Conflict`**, not stale/partial
  data, if queried before the job reaches the relevant terminal state.

---

## 16. Future Improvements

- Replace `BackgroundTasks` with **Celery + Redis** for durable,
  horizontally-scalable task execution.
- Migrate from **SQLite to PostgreSQL** for concurrent-write safety and
  production-grade durability.
- Move uploaded files from local disk to **AWS S3** for durability and
  multi-instance access.
- Replace heuristic tamper detection with an **ML-based tampering /
  forgery detection model**.
- Support **distributed workers** consuming from a shared queue, decoupled
  from the API process.
- Add **metrics and monitoring** (Prometheus/Grafana, structured request
  tracing) beyond the current file/console logging.



  ##17 . Project Links

  ## Project Links

- GitHub Repository: https://github.com/chinthana-addyei20/intelligent-media-processing-pipeline
- Live Deployment (Render): https://intelligent-media-processing-pipeline-xhd8.onrender.com
- API Documentation: https://intelligent-media-processing-pipeline-xhd8.onrender.com/docs
