from celery import Celery
import os
import json
import logging
from datetime import datetime
from typing import Any

from config.settings import settings
from src.services.llm_service import LLMService
from src.services.task_service import TaskService  # Updated from QueueService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Celery app
app = Celery('tasks', broker=settings.redis_url)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    broker_connection_retry_on_startup=True,
)

@app.task(name='annotate_chunk')
def annotate_chunk(doc_id: str, chunk_path: str, doc_type: str, chunk_num: int) -> None:
    """Annotate a document chunk using LLM"""
    try:
        with open(chunk_path, 'r', encoding='utf-8') as f:
            content = f.read()

        llm_service = LLMService()
        annotations = llm_service.extract_annotations(content, doc_type)

        output_dir = os.path.join(settings.data_dir, "annotated", doc_id)
        os.makedirs(output_dir, exist_ok=True)

        annotation_data = {
            "doc_id": doc_id,
            "chunk_num": chunk_num,
            "doc_type": doc_type,
            "content": content,
            "annotations": annotations,
            "annotation_timestamp": datetime.now().isoformat()
        }

        annotation_file = f"chunk_{chunk_num:03d}_annotations.json"
        annotation_path = os.path.join(output_dir, annotation_file)

        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2)

        # Enqueue synthetic data generation using TaskService
        TaskService.enqueue_synthesis(doc_id, chunk_num, content, annotations)

        logger.info(f"Annotated {doc_id} chunk {chunk_num}")

    except Exception as e:
        logger.error(f"Failed to annotate {doc_id} chunk {chunk_num}: {e}")
        raise