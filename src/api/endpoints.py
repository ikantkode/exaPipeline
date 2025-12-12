from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import uuid
import os
from pathlib import Path
from datetime import datetime

from src.services.queue_service import QueueService  # Unused now but kept for future
from config.settings import settings

router = APIRouter(prefix="/api/v1")

UPLOAD_DIR = Path(settings.data_dir) / "uploads"  # Shared persistent location

@router.post("/ingest")
async def ingest_pdfs(files: List[UploadFile] = File(...)):
    """Ingest PDFs and start the processing pipeline"""
    file_ids = []
    
    # Ensure base upload directory exists
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Empty filename")
            
        # Generate unique ID
        doc_id = str(uuid.uuid4())
        
        # Create dedicated directory per document
        doc_upload_dir = UPLOAD_DIR / doc_id
        doc_upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original PDF to shared volume
        original_path = doc_upload_dir / file.filename
        with open(original_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Queue ingestion task with persistent path
        from src.workers.ingestion_worker import process_pdf
        process_pdf.apply_async(
            args=(doc_id, str(original_path), file.filename),
            queue=settings.ingestion_queue  # Uses settings value for consistency
        )
        
        file_ids.append({"doc_id": doc_id, "filename": file.filename})
    
    return {
        "message": f"{len(files)} files queued for processing",
        "file_ids": file_ids,
        "upload_location": "shared volume /app/data/uploads"
    }

# Status and export endpoints unchanged (already correct)
@router.get("/status/{doc_id}")
async def get_status(doc_id: str):
    stages = [
        ("ingested", "OCR Processing"),
        ("classified", "Classification"),
        ("chunks", "Chunking"),
        ("annotated", "Annotation"),
        ("synthetic", "Synthesis"),
        ("validated", "Validation")
    ]
    
    status = {}
    for stage_dir, stage_name in stages:
        stage_path = os.path.join(settings.data_dir, stage_dir, doc_id)
        if os.path.exists(stage_path):
            status[stage_name] = "completed"
        else:
            status[stage_name] = "pending"
    
    # Add uploads check
    upload_path = os.path.join(settings.data_dir, "uploads", doc_id)
    status["Upload"] = "completed" if os.path.exists(upload_path) else "pending"
    
    return {"doc_id": doc_id, "status": status}

@router.get("/export/training")
async def export_training_data(format: str = "sft"):
    from src.core.packaging import PackageBuilder
    
    builder = PackageBuilder()
    
    if format == "sft":
        data = builder.build_sft_dataset()
    elif format == "rlaif":
        data = builder.build_rlaif_dataset()
    elif format == "rlhf":
        data = builder.build_pairwise_dataset()
    else:
        raise HTTPException(status_code=400, detail="Invalid format")
    
    return {
        "format": format,
        "count": len(data),
        "data": data[:10]
    }