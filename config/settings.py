# exaPipeline/config/settings.py
import os
from typing import Dict, List

class Settings:
    # External services
    exaocr_url: str = os.getenv("EXAOCR_URL", "http://192.168.1.151:45001")
    vllm_url: str = os.getenv("VLLM_URL", "http://192.168.1.151:45000")
    
    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://192.168.1.151:6379/0")
    
    # Data
    data_dir: str = os.getenv("DATA_DIR", "/app/data")
    
    # Queue names
    ingestion_queue: str = os.getenv("INGESTION_QUEUE", "ingestion_queue")
    classification_queue: str = os.getenv("CLASSIFICATION_QUEUE", "classification_queue")
    chunking_queue: str = os.getenv("CHUNKING_QUEUE", "chunking_queue")
    annotation_queue: str = os.getenv("ANNOTATION_QUEUE", "annotation_queue")
    synthesis_queue: str = os.getenv("SYNTHESIS_QUEUE", "synthesis_queue")
    validation_queue: str = os.getenv("VALIDATION_QUEUE", "validation_queue")
    
    # Temperatures
    classification_temperature: float = float(os.getenv("CLASSIFICATION_TEMPERATURE", "0.1"))
    annotation_temperature: float = float(os.getenv("ANNOTATION_TEMPERATURE", "0.2"))
    synthesis_temperature: float = float(os.getenv("SYNTHESIS_TEMPERATURE", "0.7"))
    
    # CHUNKING SETTINGS — THESE WERE MISSING
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "2000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    max_chunk_length: int = int(os.getenv("MAX_CHUNK_LENGTH", "3000"))
    
    # Document types
    DOCUMENT_TYPES: List[str] = [
        "certified_payroll", "submittal", "specification", "contract",
        "invoice", "receipt", "delay_report", "email", "check_copy"
    ]

settings = Settings()