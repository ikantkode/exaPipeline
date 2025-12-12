# src/workers/synthesis_worker.py
from celery import Celery
from typing import List, Dict, Any
import logging
import os
import json

from src.services.llm_service import LLMService
from src.services.task_service import TaskService

logger = logging.getLogger(__name__)

# Celery app
celery = Celery(__name__)
celery.conf.broker_url = os.getenv("REDIS_URL", "redis://192.168.1.151:6379/0")
celery.conf.result_backend = "rpc://"

llm_service = LLMService()

def ensure_directory(directory: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    os.makedirs(directory, exist_ok=True)

def save_json_fixed(path: str, data: Dict[str, Any]) -> None:
    """Save dictionary to JSON file with proper error handling."""
    ensure_directory(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

@celery.task
def generate_synthetic(doc_id: str, chunk_id: int, content: str, annotations: Dict[str, Any], num_variations: int = 3) -> None:
    logger.info(f"Generating {num_variations} synthetic variations for {doc_id} chunk {chunk_id}")
    
    try:
        variations: List[Dict[str, Any]] = llm_service.generate_synthetic_variations(
            content=content,
            annotations=annotations,
            num_variations=num_variations
        )
        
        synthetic_dir = f"/app/data/synthetic/{doc_id}"
        ensure_directory(synthetic_dir)
        
        # Save each variation
        for i, variation_dict in enumerate(variations):
            # Ensure variation_dict is a proper dictionary
            if not isinstance(variation_dict, dict):
                logger.warning(f"Variation {i} is not a dict: {type(variation_dict)}, converting...")
                variation_dict = {"content": str(variation_dict)}
            
            # Add metadata to the variation
            variation_dict.update({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "variation_index": i,
                "original_content": content,
                "is_synthetic": True
            })
            
            # Create file path
            file_path = os.path.join(synthetic_dir, f"chunk_{chunk_id:04d}_syn_{i+1:02d}.json")
            
            # Debug logging
            logger.info(f"Type of file_path: {type(file_path)}, value: {file_path}")
            logger.info(f"Type of variation_dict: {type(variation_dict)}")
            
            # Save with fixed function
            save_json_fixed(file_path, variation_dict)
            logger.info(f"Saved synthetic variation {i+1} → {file_path}")
            
            # Enqueue validation task for this synthetic variation
            TaskService.enqueue_validation(
                doc_id=doc_id,
                chunk_num=chunk_id,
                variation_num=i+1,
                is_synthetic=True
            )
            logger.info(f"Enqueued validation for synthetic variation {i+1}")
            
        logger.info(f"Successfully saved {len(variations)} synthetic variations for {doc_id} chunk {chunk_id}")
        
    except Exception as e:
        logger.error(f"Synthetic generation failed for {doc_id} chunk {chunk_id}: {e}", exc_info=True)
        raise