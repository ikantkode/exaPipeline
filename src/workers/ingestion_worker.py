# src/workers/ingestion_worker.py
import os
import json
import logging
from datetime import datetime
from pathlib import Path

from celery import Celery
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import settings & services (must be after Celery app to avoid circular imports)
from config.settings import settings
from src.services.ocr_service import OCRService

# ----------------------------------------------------------------------
# Celery App Configuration (Fixed: Use settings consistently + no override risk)
# ----------------------------------------------------------------------
app = Celery(
    'ingestion_tasks',
    broker=settings.redis_url,  # ← Always uses REDIS_URL from .env (e.g., redis://redis:6379/0)
    backend=settings.redis_url   # Optional: can be disabled if not using results
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
    # Ensure tasks go to correct queues
    task_routes={
        "process_pdf": {"queue": settings.ingestion_queue},
    },
)

# ----------------------------------------------------------------------
# Logging Configuration
# ----------------------------------------------------------------------
logger = logging.getLogger("ingestion-worker")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ----------------------------------------------------------------------
# Robust OCR Processing with Retry + Proper File Path Handling
# ----------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((FileNotFoundError, ConnectionError, TimeoutError)),
    reraise=True,
)
def _safe_ocr_process(ocr_service: OCRService, file_path: str):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"OCR input file not found: {file_path}")
    return ocr_service.process_pdf(file_path)


# ----------------------------------------------------------------------
# Main Celery Task — Fully Fixed & Hardened
# ----------------------------------------------------------------------
@app.task(bind=True, name="process_pdf", max_retries=3, default_retry_delay=60)
def process_pdf(self, doc_id: str, file_path: str, filename: str):
    """
    Ingest PDF → OCR → Save Markdown + Metadata → Trigger Classification
    """
    output_dir = Path(settings.data_dir) / "ingested" / doc_id
    md_path = output_dir / "document.md"
    metadata_path = output_dir / "metadata.json"

    try:
        logger.info(f"Starting ingestion for doc_id={doc_id}, file={filename}")

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Validate input file exists before OCR
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Uploaded file missing: {file_path}")

        # Process with OCR (with retry logic)
        ocr_service = OCRService()
        result = _safe_ocr_process(ocr_service, file_path)

        # Write markdown
        md_path.write_text(result["markdown"], encoding="utf-8")

        # Generate metadata
        metadata = {
            "doc_id": doc_id,
            "filename": filename,
            "original_path": file_path,
            "markdown_path": str(md_path),
            "page_count": result["markdown"].count("---") + 1,
            "ocr_engine": result.get("engine", "exaOCR"),
            "processing_timestamp": datetime.utcnow().isoformat() + "Z",
            "file_size_bytes": os.path.getsize(file_path),
            "file_hash": result.get("file_id", "")
        }

        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # Clean up uploaded file
        try:
            os.remove(file_path)
            logger.info(f"Removed temporary file: {file_path}")
        except OSError as e:
            logger.warning(f"Could not remove temp file {file_path}: {e}")

        # Trigger next stage: Classification
        from src.workers.classification_worker import classify_document

        classify_document.apply_async(
            args=(doc_id, str(md_path)),
            queue=settings.classification_queue,
            countdown=2  # Small delay to ensure file is fully written
        )

        logger.info(f"Successfully ingested {filename} → {doc_id} | Next: classification")

        return {"status": "success", "doc_id": doc_id, "markdown_path": str(md_path)}

    except FileNotFoundError as exc:
        logger.error(f"File not found during ingestion {doc_id}: {exc}")
        raise self.retry(exc=exc, countdown=60)

    except Exception as exc:
        logger.exception(f"Ingestion failed for {doc_id} ({filename}): {exc}")
        # Optional: move failed file to quarantine
        failed_dir = Path(settings.data_dir) / "failed" / doc_id
        failed_dir.mkdir(parents=True, exist_ok=True)
        try:
            if Path(file_path).exists():
                shutil.move(file_path, failed_dir / filename)
        except Exception:
            pass
        raise self.retry(exc=exc, countdown=120)