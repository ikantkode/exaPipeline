from celery import Celery
import os
import json
import logging
from datetime import datetime

from config.settings import settings
from src.services.llm_service import LLMService
from src.services.task_service import TaskService  # Changed from QueueService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Celery app with proper configuration
app = Celery('tasks', broker=settings.redis_url)

# Configure the app
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    broker_connection_retry_on_startup=True,  # Important for Celery 5.3+
)

@app.task(name='classify_document')
def classify_document(doc_id: str, md_path: str):
    """Classify document type using LLM"""
    
    try:
        # Read markdown content
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Classify using LLM
        llm_service = LLMService()
        doc_type = llm_service.classify_document(content)
        
        # Create output directory
        output_dir = os.path.join(settings.data_dir, "classified", doc_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save classification
        with open(os.path.join(output_dir, "type.txt"), 'w') as f:
            f.write(doc_type)
        
        # Save metadata
        metadata = {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "classification_timestamp": datetime.now().isoformat(),
            "source_path": md_path
        }
        
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Queue for chunking using TaskService
        TaskService.enqueue_chunking(doc_id, md_path, doc_type)
        
        logger.info(f"Classified {doc_id} as {doc_type}")
        
    except Exception as e:
        logger.error(f"Failed to classify {doc_id}: {e}")
        raise