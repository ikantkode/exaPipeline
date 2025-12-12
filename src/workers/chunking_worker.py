from celery import Celery
import os
import json
import re
import logging
from datetime import datetime
from typing import List, Dict, Any

from config.settings import settings
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

def chunk_by_headings(content: str, max_length: int = 2000) -> List[Dict[str, Any]]:
    """Chunk markdown content by headings"""
    chunks = []
    lines = content.split('\n')
    
    current_chunk = []
    current_length = 0
    
    for line in lines:
        line_length = len(line)
        
        # Check if we should start a new chunk
        if (current_length + line_length > max_length and 
            line.strip().startswith('#') and 
            current_chunk):
            
            chunks.append({
                'content': '\n'.join(current_chunk),
                'chunk_type': 'heading_based'
            })
            current_chunk = [line]
            current_length = line_length
        else:
            current_chunk.append(line)
            current_length += line_length
    
    # Add the last chunk
    if current_chunk:
        chunks.append({
            'content': '\n'.join(current_chunk),
            'chunk_type': 'heading_based'
        })
    
    return chunks

@app.task(name='chunk_document')
def chunk_document(doc_id: str, md_path: str, doc_type: str):
    """Chunk document into smaller pieces"""
    
    try:
        # Read markdown content
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create chunks
        chunks = chunk_by_headings(content, settings.chunk_size)
        
        # Create output directory
        output_dir = os.path.join(settings.data_dir, "chunks", doc_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each chunk
        chunk_files = []
        for i, chunk in enumerate(chunks, 1):
            chunk_filename = f"chunk_{i:03d}.md"
            chunk_path = os.path.join(output_dir, chunk_filename)
            
            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write(chunk['content'])
            
            chunk_files.append({
                'filename': chunk_filename,
                'type': chunk['chunk_type'],
                'length': len(chunk['content'])
            })
        
        # Save metadata
        metadata = {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "total_chunks": len(chunks),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "chunks": chunk_files,
            "chunking_timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Queue each chunk for annotation using TaskService
        for i in range(1, len(chunks) + 1):
            chunk_path = os.path.join(output_dir, f"chunk_{i:03d}.md")
            TaskService.enqueue_annotation(doc_id, chunk_path, doc_type, i)
        
        logger.info(f"Chunked {doc_id} into {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Failed to chunk {doc_id}: {e}")
        raise