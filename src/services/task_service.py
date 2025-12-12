import logging
from celery import current_app
from config.settings import settings

logger = logging.getLogger(__name__)

class TaskService:
    @staticmethod
    def enqueue_ingestion(doc_id: str, file_path: str, filename: str):
        """Enqueue PDF processing task"""
        from src.workers.ingestion_worker import process_pdf
        process_pdf.apply_async(
            args=(doc_id, file_path, filename),
            queue=settings.ingestion_queue
        )
        logger.info(f"Enqueued ingestion task for {doc_id}")
    
    @staticmethod
    def enqueue_classification(doc_id: str, md_path: str):
        """Enqueue classification task"""
        from src.workers.classification_worker import classify_document
        classify_document.apply_async(
            args=(doc_id, md_path),
            queue=settings.classification_queue
        )
    
    @staticmethod
    def enqueue_chunking(doc_id: str, md_path: str, doc_type: str):
        """Enqueue chunking task"""
        from src.workers.chunking_worker import chunk_document
        chunk_document.apply_async(
            args=(doc_id, md_path, doc_type),
            queue=settings.chunking_queue
        )
    
    @staticmethod
    def enqueue_annotation(doc_id: str, chunk_path: str, doc_type: str, chunk_num: int):
        """Enqueue annotation task"""
        from src.workers.annotation_worker import annotate_chunk
        annotate_chunk.apply_async(
            args=(doc_id, chunk_path, doc_type, chunk_num),
            queue=settings.annotation_queue
        )
    
    @staticmethod
    def enqueue_synthesis(doc_id: str, chunk_num: int, content: str, annotations: dict):
        """Enqueue synthesis task"""
        from src.workers.synthesis_worker import generate_synthetic
        generate_synthetic.apply_async(
            args=(doc_id, chunk_num, content, annotations),
            queue=settings.synthesis_queue
        )
    
    @staticmethod
    def enqueue_validation(doc_id: str, chunk_num: int, variation_num: int, is_synthetic: bool):
        """Enqueue validation task"""
        from src.workers.validation_worker import validate_data
        validate_data.apply_async(
            args=(doc_id, chunk_num, variation_num, is_synthetic),
            queue=settings.validation_queue
        )