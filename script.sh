#!/bin/bash

# insert-code.sh
# Inserts code into existing pipeline files

set -e

echo "Inserting code into existing pipeline files..."

# 1. Docker Compose
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # Pipeline API Gateway
  pipeline-api:
    build: .
    container_name: pipeline-api
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./config:/app/config
      - ./src:/app/src
    environment:
      - REDIS_URL=redis://redis:6379/0
      - EXAOCR_URL=http://host.docker.internal:45001
      - VLLM_URL=http://host.docker.internal:45000
      - LOG_LEVEL=INFO
    depends_on:
      - redis
    command: uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
    networks:
      - pipeline-network

  # Redis for job queue
  redis:
    image: redis:7-alpine
    container_name: pipeline-redis
    ports:
      - "6379:6379"
    networks:
      - pipeline-network

  # Celery worker for ingestion
  ingestion-worker:
    build: .
    container_name: ingestion-worker
    volumes:
      - ./data:/app/data
      - ./config:/app/config
      - ./src:/app/src
    environment:
      - REDIS_URL=redis://redis:6379/0
      - EXAOCR_URL=http://host.docker.internal:45001
      - VLLM_URL=http://host.docker.internal:45000
      - WORKER_TYPE=ingestion
    depends_on:
      - redis
      - pipeline-api
    command: celery -A src.workers.ingestion_worker worker --loglevel=info --concurrency=2 -Q ingestion_queue
    networks:
      - pipeline-network

  # Celery worker for classification
  classification-worker:
    build: .
    container_name: classification-worker
    volumes:
      - ./data:/app/data
      - ./config:/app/config
      - ./src:/app/src
    environment:
      - REDIS_URL=redis://redis:6379/0
      - EXAOCR_URL=http://host.docker.internal:45001
      - VLLM_URL=http://host.docker.internal:45000
      - WORKER_TYPE=classification
    depends_on:
      - redis
    command: celery -A src.workers.classification_worker worker --loglevel=info --concurrency=4 -Q classification_queue
    networks:
      - pipeline-network

  # Celery worker for chunking
  chunking-worker:
    build: .
    container_name: chunking-worker
    volumes:
      - ./data:/app/data
      - ./config:/app/config
      - ./src:/app/src
    environment:
      - REDIS_URL=redis://redis:6379/0
      - WORKER_TYPE=chunking
    depends_on:
      - redis
    command: celery -A src.workers.chunking_worker worker --loglevel=info --concurrency=4 -Q chunking_queue
    networks:
      - pipeline-network

  # Celery worker for annotation
  annotation-worker:
    build: .
    container_name: annotation-worker
    volumes:
      - ./data:/app/data
      - ./config:/app/config
      - ./src:/app/src
    environment:
      - REDIS_URL=redis://redis:6379/0
      - VLLM_URL=http://host.docker.internal:45000
      - WORKER_TYPE=annotation
    depends_on:
      - redis
    command: celery -A src.workers.annotation_worker worker --loglevel=info --concurrency=4 -Q annotation_queue
    networks:
      - pipeline-network

  # Celery worker for synthesis
  synthesis-worker:
    build: .
    container_name: synthesis-worker
    volumes:
      - ./data:/app/data
      - ./config:/app/config
      - ./src:/app/src
    environment:
      - REDIS_URL=redis://redis:6379/0
      - VLLM_URL=http://host.docker.internal:45000
      - WORKER_TYPE=synthesis
    depends_on:
      - redis
    command: celery -A src.workers.synthesis_worker worker --loglevel=info --concurrency=2 -Q synthesis_queue
    networks:
      - pipeline-network

  # Celery worker for validation
  validation-worker:
    build: .
    container_name: validation-worker
    volumes:
      - ./data:/app/data
      - ./config:/app/config
      - ./src:/app/src
    environment:
      - REDIS_URL=redis://redis:6379/0
      - VLLM_URL=http://host.docker.internal:45000
      - WORKER_TYPE=validation
    depends_on:
      - redis
    command: celery -A src.workers.validation_worker worker --loglevel=info --concurrency=4 -Q validation_queue
    networks:
      - pipeline-network

  # Flower for monitoring
  flower:
    build: .
    container_name: pipeline-flower
    ports:
      - "5555:5555"
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    command: celery -A src.workers.ingestion_worker flower --port=5555
    networks:
      - pipeline-network

networks:
  pipeline-network:
    driver: bridge
EOF
echo "Created docker-compose.yml"

# 2. Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/data/ingested \
    /app/data/classified \
    /app/data/chunks \
    /app/data/annotated \
    /app/data/synthetic \
    /app/data/validated \
    /app/data/train

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
echo "Created Dockerfile"

# 3. Requirements
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
celery==5.3.4
redis==5.0.1
flower==2.0.1
requests==2.31.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-multipart==0.0.6
aiofiles==23.2.1
watchfiles==0.21.0
loguru==0.7.2
PyPDF2==3.0.1
markdown-it-py==3.0.0
python-json-logger==2.0.7
httpx==0.25.1
tenacity==8.2.3
pandas==2.1.3
numpy==1.24.4
EOF
echo "Created requirements.txt"

# 4. Config Settings
mkdir -p config
cat > config/settings.py << 'EOF'
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # External Services
    exaocr_url: str = os.getenv("EXAOCR_URL", "http://localhost:45001")
    vllm_url: str = os.getenv("VLLM_URL", "http://localhost:45000")
    
    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Data paths
    data_dir: str = os.getenv("DATA_DIR", "/app/data")
    
    # Processing settings
    chunk_size: int = 2000
    chunk_overlap: int = 200
    max_chunk_length: int = 3000
    
    # Model settings
    classification_temperature: float = 0.1
    annotation_temperature: float = 0.2
    synthesis_temperature: float = 0.7
    
    # Queue names
    ingestion_queue: str = "ingestion_queue"
    classification_queue: str = "classification_queue"
    chunking_queue: str = "chunking_queue"
    annotation_queue: str = "annotation_queue"
    synthesis_queue: str = "synthesis_queue"
    validation_queue: str = "validation_queue"
    
    class Config:
        env_file = ".env"

settings = Settings()
EOF
echo "Created config/settings.py"

# 5. Main application
mkdir -p src
cat > src/main.py << 'EOF'
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import uuid
import logging

from src.api.endpoints import router
from src.services.queue_service import QueueService
from src.utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Construction AI Pipeline API",
    description="Pipeline for training Qwen3:4B on construction documents",
    version="1.0.0"
)

# Include routers
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Construction AI Pipeline API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF
echo "Created src/main.py"

# 6. API Endpoints
mkdir -p src/api
cat > src/api/endpoints.py << 'EOF'
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from typing import List
import uuid
import os
from datetime import datetime

from src.services.queue_service import QueueService
from config.settings import settings

router = APIRouter(prefix="/api/v1")

@router.post("/ingest")
async def ingest_pdfs(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """Ingest PDFs and start the processing pipeline"""
    file_ids = []
    
    for file in files:
        # Generate unique ID for this document
        doc_id = str(uuid.uuid4())
        
        # Save file temporarily
        temp_path = f"/tmp/{doc_id}_{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Queue ingestion task
        queue = QueueService()
        queue.enqueue(
            queue_name=settings.ingestion_queue,
            task_name="process_pdf",
            args=(doc_id, temp_path, file.filename)
        )
        
        file_ids.append(doc_id)
    
    return {
        "message": f"{len(files)} files queued for processing",
        "file_ids": file_ids
    }

@router.get("/status/{doc_id}")
async def get_status(doc_id: str):
    """Get processing status for a document"""
    # Check each stage directory
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
    
    return {"doc_id": doc_id, "status": status}

@router.get("/export/training")
async def export_training_data(format: str = "sft"):
    """Export training data in specified format"""
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
        "data": data[:10]  # Return first 10 samples
    }
EOF
echo "Created src/api/endpoints.py"

# 7. OCR Service
mkdir -p src/services
cat > src/services/ocr_service.py << 'EOF'
import requests
import time
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self):
        self.base_url = settings.exaocr_url
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def process_pdf(self, file_path: str, force_ocr: bool = True) -> Dict[str, Any]:
        """Process PDF through exaOCR API"""
        
        # Upload file
        with open(file_path, 'rb') as f:
            files = {'files': (file_path, f, 'application/pdf')}
            params = {'force_ocr': str(force_ocr).lower()}
            
            response = requests.post(
                f"{self.base_url}/upload/",
                files=files,
                params=params
            )
        
        if response.status_code != 200:
            raise Exception(f"OCR upload failed: {response.text}")
        
        result = response.json()
        file_id = result.get('file_id')
        
        if not file_id:
            raise Exception("No file_id in response")
        
        # Wait for processing
        md_id = self._wait_for_processing(file_id)
        
        # Download markdown
        markdown = self._download_markdown(md_id)
        
        # Cleanup
        self._cleanup(file_id)
        
        return {
            'markdown': markdown,
            'file_id': file_id,
            'md_id': md_id
        }
    
    def _wait_for_processing(self, file_id: str, timeout: int = 300) -> str:
        """Wait for OCR processing to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = requests.get(f"{self.base_url}/progress/{file_id}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'completed':
                    return result.get('md_id')
            
            time.sleep(2)
        
        raise TimeoutError(f"OCR processing timeout for {file_id}")
    
    def _download_markdown(self, md_id: str) -> str:
        """Download processed markdown"""
        response = requests.get(f"{self.base_url}/download-markdown/{md_id}")
        
        if response.status_code != 200:
            raise Exception(f"Failed to download markdown: {response.text}")
        
        return response.text
    
    def _cleanup(self, file_id: str):
        """Cleanup temporary files"""
        try:
            requests.delete(f"{self.base_url}/cleanup/{file_id}")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
EOF
echo "Created src/services/ocr_service.py"

# 8. LLM Service
cat > src/services/llm_service.py << 'EOF'
import requests
import json
from typing import Dict, Any, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.base_url = settings.vllm_url
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, 
                 prompt: str, 
                 temperature: float = 0.7,
                 max_tokens: int = 2000,
                 system_prompt: Optional[str] = None) -> str:
        """Generate text using Qwen3:4B via vLLM"""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": "qwen3-4b-instruct",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise Exception(f"LLM call failed: {response.text}")
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def classify_document(self, content: str) -> str:
        """Classify document type"""
        system_prompt = "You are a document classifier. Return ONLY the document type."
        
        prompt = f"""Classify this document into one of these types:

["certified_payroll", "submittal", "specification", "contract", 
 "invoice", "receipt", "delay_report", "email", "check_copy"]

Document content:
{content[:2000]} # Limit content for classification

Return ONLY the type."""
        
        result = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=settings.classification_temperature,
            max_tokens=50
        )
        
        return result.strip().strip('"').strip("'")
    
    def extract_annotations(self, content: str, doc_type: str) -> Dict[str, Any]:
        """Extract structured annotations from document chunk"""
        system_prompt = "You are a structured-data extractor. Return JSON only."
        
        prompt = f"""Extract information from this {doc_type} document chunk.

Extract:
- key dates
- company names
- people involved
- monetary amounts
- compliance status
- action items
- any tables as JSON

Return as JSON with these fields:
{{
  "dates": [],
  "companies": [],
  "people": [],
  "amounts": [],
  "compliance_status": "",
  "action_items": [],
  "tables": []
}}

Document chunk:{content}
        
        result = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=settings.annotation_temperature,
            max_tokens=1000
        )
        
        # Try to parse JSON
        try:
            # Find JSON in response
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            json_str = result[json_start:json_end]
            
            return json.loads(json_str)
        except:
            logger.warning(f"Failed to parse JSON from: {result[:200]}")
            return {"error": "Failed to parse LLM response"}
    
    def generate_synthetic_variations(self, content: str, annotations: Dict[str, Any], n_variations: int = 3) -> List[Dict[str, Any]]:
        """Generate synthetic variations of document chunk"""
        system_prompt = "You are generating synthetic training samples. Return JSON only."
        
        prompt = f"""Generate {n_variations} realistic variations of this document chunk.

Original chunk:{content[:1000]}
Original annotations:{json.dumps(annotations, indent=2)}

For each variation:
1. Change company names, dates, amounts
2. Paraphrase wording
3. Maintain realistic construction document style
4. Keep the same structure

Return as JSON array:
[
  {{
    "content": "...",
    "annotations": {{...}}
  }},
  ...
]"""
        
        result = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=settings.synthesis_temperature,
            max_tokens=3000
        )
        
        try:
            json_start = result.find('[')
            json_end = result.rfind(']') + 1
            json_str = result[json_start:json_end]
            
            return json.loads(json_str)
        except:
            logger.warning(f"Failed to parse synthetic variations: {result[:200]}")
            return []
EOF
echo "Created src/services/llm_service.py"

# 9. Queue Service
cat > src/services/queue_service.py << 'EOF'
import redis
import json
from typing import Any, Dict, List, Optional
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

class QueueService:
    def __init__(self):
        self.redis_client = redis.Redis.from_url(settings.redis_url)
    
    def enqueue(self, queue_name: str, task_name: str, args: tuple, kwargs: Optional[Dict] = None) -> str:
        """Enqueue a task to Redis queue"""
        task = {
            'task': task_name,
            'args': args,
            'kwargs': kwargs or {}
        }
        
        # Push to Redis list (queue)
        result = self.redis_client.rpush(queue_name, json.dumps(task))
        
        logger.info(f"Enqueued task {task_name} to {queue_name}")
        return str(result)
    
    def dequeue(self, queue_name: str) -> Optional[Dict]:
        """Dequeue a task from Redis queue"""
        task_json = self.redis_client.lpop(queue_name)
        
        if task_json:
            return json.loads(task_json)
        return None
    
    def get_queue_length(self, queue_name: str) -> int:
        """Get length of queue"""
        return self.redis_client.llen(queue_name)
    
    def get_all_queue_names(self) -> List[str]:
        """Get all queue names (simplified - in production use Redis patterns)"""
        return [
            settings.ingestion_queue,
            settings.classification_queue,
            settings.chunking_queue,
            settings.annotation_queue,
            settings.synthesis_queue,
            settings.validation_queue
        ]
EOF
echo "Created src/services/queue_service.py"

# 10. Ingestion Worker
mkdir -p src/workers
cat > src/workers/ingestion_worker.py << 'EOF'
from celery import Celery
import os
import json
import shutil
import logging
from datetime import datetime

from config.settings import settings
from src.services.ocr_service import OCRService
from src.services.queue_service import QueueService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Celery app
app = Celery('ingestion_worker', broker=settings.redis_url)

@app.task(name='process_pdf')
def process_pdf(doc_id: str, file_path: str, filename: str):
    """Process PDF through OCR and save results"""
    
    try:
        # Create output directory
        output_dir = os.path.join(settings.data_dir, "ingested", doc_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process with OCR
        ocr_service = OCRService()
        result = ocr_service.process_pdf(file_path)
        
        # Save markdown
        md_path = os.path.join(output_dir, "document.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(result['markdown'])
        
        # Save metadata
        metadata = {
            "filename": filename,
            "doc_id": doc_id,
            "pages": result['markdown'].count('---'),  # Approximate page count
            "ocr_timestamp": datetime.now().isoformat(),
            "file_size": os.path.getsize(file_path),
            "hash": result.get('file_id', '')
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Cleanup temp file
        os.remove(file_path)
        
        # Queue for classification
        queue = QueueService()
        queue.enqueue(
            queue_name=settings.classification_queue,
            task_name="classify_document",
            args=(doc_id, md_path)
        )
        
        logger.info(f"Successfully processed {filename} as {doc_id}")
        
    except Exception as e:
        logger.error(f"Failed to process {filename}: {e}")
        # Cleanup on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
EOF
echo "Created src/workers/ingestion_worker.py"

# 11. Classification Worker
cat > src/workers/classification_worker.py << 'EOF'
from celery import Celery
import os
import json
import logging
from datetime import datetime

from config.settings import settings
from src.services.llm_service import LLMService
from src.services.queue_service import QueueService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Celery('classification_worker', broker=settings.redis_url)

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
        
        # Queue for chunking
        queue = QueueService()
        queue.enqueue(
            queue_name=settings.chunking_queue,
            task_name="chunk_document",
            args=(doc_id, md_path, doc_type)
        )
        
        logger.info(f"Classified {doc_id} as {doc_type}")
        
    except Exception as e:
        logger.error(f"Failed to classify {doc_id}: {e}")
        raise
EOF
echo "Created src/workers/classification_worker.py"

# 12. Chunking Worker
cat > src/workers/chunking_worker.py << 'EOF'
from celery import Celery
import os
import json
import re
import logging
from datetime import datetime
from typing import List, Dict, Any

from config.settings import settings
from src.services.queue_service import QueueService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Celery('chunking_worker', broker=settings.redis_url)

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
        
        # Queue each chunk for annotation
        queue = QueueService()
        for i in range(1, len(chunks) + 1):
            chunk_path = os.path.join(output_dir, f"chunk_{i:03d}.md")
            queue.enqueue(
                queue_name=settings.annotation_queue,
                task_name="annotate_chunk",
                args=(doc_id, chunk_path, doc_type, i)
            )
        
        logger.info(f"Chunked {doc_id} into {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Failed to chunk {doc_id}: {e}")
        raise
EOF
echo "Created src/workers/chunking_worker.py"

# 13. Annotation Worker
cat > src/workers/annotation_worker.py << 'EOF'
from celery import Celery
import os
import json
import logging
from datetime import datetime

from config.settings import settings
from src.services.llm_service import LLMService
from src.services.queue_service import QueueService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Celery('annotation_worker', broker=settings.redis_url)

@app.task(name='annotate_chunk')
def annotate_chunk(doc_id: str, chunk_path: str, doc_type: str, chunk_num: int):
    """Annotate a document chunk using LLM"""
    
    try:
        # Read chunk content
        with open(chunk_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract annotations using LLM
        llm_service = LLMService()
        annotations = llm_service.extract_annotations(content, doc_type)
        
        # Create output directory
        output_dir = os.path.join(settings.data_dir, "annotated", doc_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save annotations
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
        
        with open(annotation_path, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        
        # Queue for synthetic data generation
        queue = QueueService()
        queue.enqueue(
            queue_name=settings.synthesis_queue,
            task_name="generate_synthetic",
            args=(doc_id, chunk_num, content, annotations)
        )
        
        logger.info(f"Annotated {doc_id} chunk {chunk_num}")
        
    except Exception as e:
        logger.error(f"Failed to annotate {doc_id} chunk {chunk_num}: {e}")
        raise
EOF
echo "Created src/workers/annotation_worker.py"

# 14. Synthesis Worker
cat > src/workers/synthesis_worker.py << 'EOF'
from celery import Celery
import os
import json
import logging
from datetime import datetime

from config.settings import settings
from src.services.llm_service import LLMService
from src.services.queue_service import QueueService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Celery('synthesis_worker', broker=settings.redis_url)

@app.task(name='generate_synthetic')
def generate_synthetic(doc_id: str, chunk_num: int, content: str, annotations: dict):
    """Generate synthetic variations of annotated chunk"""
    
    try:
        # Generate synthetic variations using LLM
        llm_service = LLMService()
        variations = llm_service.generate_synthetic_variations(content, annotations, n_variations=3)
        
        # Create output directory
        output_dir = os.path.join(settings.data_dir, "synthetic", doc_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each variation
        for i, variation in enumerate(variations, 1):
            synthetic_data = {
                "doc_id": doc_id,
                "original_chunk": chunk_num,
                "variation_num": i,
                "is_synthetic": True,
                "content": variation.get('content', ''),
                "annotations": variation.get('annotations', {}),
                "generation_timestamp": datetime.now().isoformat()
            }
            
            synthetic_file = f"chunk_{chunk_num:03d}_syn_{i:03d}.json"
            synthetic_path = os.path.join(output_dir, synthetic_file)
            
            with open(synthetic_path, 'w') as f:
                json.dump(synthetic_data, f, indent=2)
        
        # Queue for validation
        queue = QueueService()
        for i in range(1, len(variations) + 1):
            synthetic_path = os.path.join(output_dir, f"chunk_{chunk_num:03d}_syn_{i:03d}.json")
            queue.enqueue(
                queue_name=settings.validation_queue,
                task_name="validate_data",
                args=(doc_id, chunk_num, i, True)  # True for synthetic
            )
        
        # Also queue the original for validation
        queue.enqueue(
            queue_name=settings.validation_queue,
            task_name="validate_data",
            args=(doc_id, chunk_num, 0, False)  # False for original
        )
        
        logger.info(f"Generated {len(variations)} synthetic variations for {doc_id} chunk {chunk_num}")
        
    except Exception as e:
        logger.error(f"Failed to generate synthetic for {doc_id} chunk {chunk_num}: {e}")
        raise
EOF
echo "Created src/workers/synthesis_worker.py"

# 15. Validation Worker
cat > src/workers/validation_worker.py << 'EOF'
from celery import Celery
import os
import json
import logging
from datetime import datetime

from config.settings import settings
from src.services.llm_service import LLMService
from src.services.queue_service import QueueService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Celery('validation_worker', broker=settings.redis_url)

def validate_with_llm(content: str, annotations: dict, is_synthetic: bool) -> dict:
    """Validate data consistency using LLM"""
    llm_service = LLMService()
    
    system_prompt = "You are a data validator. Return JSON only."
    
    prompt = f"""Validate this {'synthetic' if is_synthetic else 'original'} document data.

Check:
1. Are the annotations consistent with the content?
2. Are there any hallucinations (information not in content)?
3. Is the formatting correct?
4. Is the data realistic for construction documents?

Content:{content[:1000]}
Annotations:{json.dumps(annotations, indent=2)}

Return JSON with:
{{
  "valid": true/false,
  "score": 0-100,
  "errors": [],
  "warnings": []
}}"""
    
    try:
        result = llm_service.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=500
        )
        
        # Parse JSON response
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        json_str = result[json_start:json_end]
        
        return json.loads(json_str)
    except Exception as e:
        logger.warning(f"Validation LLM failed: {e}")
        return {"valid": False, "score": 0, "errors": ["Validation failed"], "warnings": []}

@app.task(name='validate_data')
def validate_data(doc_id: str, chunk_num: int, variation_num: int, is_synthetic: bool):
    """Validate data quality and consistency"""
    
    try:
        # Load the data
        if is_synthetic:
            source_dir = "synthetic"
            if variation_num == 0:
                # This shouldn't happen for synthetic, but handle gracefully
                return
            filename = f"chunk_{chunk_num:03d}_syn_{variation_num:03d}.json"
        else:
            source_dir = "annotated"
            filename = f"chunk_{chunk_num:03d}_annotations.json"
        
        data_path = os.path.join(settings.data_dir, source_dir, doc_id, filename)
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Validate using LLM
        validation_result = validate_with_llm(
            data.get('content', ''),
            data.get('annotations', {}),
            is_synthetic
        )
        
        # Create output directory
        output_dir = os.path.join(settings.data_dir, "validated", doc_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save validation results
        validated_data = {
            **data,
            "validation": validation_result,
            "validation_timestamp": datetime.now().isoformat(),
            "is_validated": True
        }
        
        # Determine output filename
        if is_synthetic:
            output_file = f"chunk_{chunk_num:03d}_syn_{variation_num:03d}_validated.json"
        else:
            output_file = f"chunk_{chunk_num:03d}_validated.json"
        
        output_path = os.path.join(output_dir, output_file)
        
        with open(output_path, 'w') as f:
            json.dump(validated_data, f, indent=2)
        
        logger.info(f"Validated {doc_id} chunk {chunk_num} {'synthetic' if is_synthetic else 'original'}")
        
    except Exception as e:
        logger.error(f"Failed to validate {doc_id}: {e}")
        raise
EOF
echo "Created src/workers/validation_worker.py"

# 16. Core Packaging Module
mkdir -p src/core
cat > src/core/packaging.py << 'EOF'
import os
import json
import glob
from typing import List, Dict, Any
import random

from config.settings import settings

class PackageBuilder:
    def __init__(self):
        self.data_dir = settings.data_dir
    
    def _load_validated_data(self) -> List[Dict[str, Any]]:
        """Load all validated data"""
        validated_data = []
        validated_dir = os.path.join(self.data_dir, "validated")
        
        # Walk through all validated files
        for root, dirs, files in os.walk(validated_dir):
            for file in files:
                if file.endswith('_validated.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        # Only include valid data
                        if data.get('validation', {}).get('valid', False):
                            validated_data.append(data)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        
        return validated_data
    
    def build_sft_dataset(self) -> List[Dict[str, Any]]:
        """Build SFT (Supervised Fine-Tuning) dataset"""
        validated_data = self._load_validated_data()
        sft_samples = []
        
        for data in validated_data:
            # Create SFT sample
            sft_sample = {
                "input": data.get('content', ''),
                "output": json.dumps(data.get('annotations', {})),
                "meta": {
                    "doc_id": data.get('doc_id', ''),
                    "chunk_num": data.get('chunk_num', 0),
                    "doc_type": data.get('doc_type', ''),
                    "is_synthetic": data.get('is_synthetic', False),
                    "validation_score": data.get('validation', {}).get('score', 0)
                }
            }
            sft_samples.append(sft_sample)
        
        return sft_samples
    
    def build_rlaif_dataset(self) -> List[Dict[str, Any]]:
        """Build RLAIF (Reinforcement Learning from AI Feedback) dataset"""
        validated_data = self._load_validated_data()
        rlaif_samples = []
        
        for data in validated_data:
            # Create RLAIF sample with scoring
            rlaif_sample = {
                "prompt": f"Analyze this construction document: {data.get('content', '')[:500]}...",
                "response": json.dumps(data.get('annotations', {})),
                "score": data.get('validation', {}).get('score', 50) / 100.0  # Normalize to 0-1
            }
            rlaif_samples.append(rlaif_sample)
        
        return rlaif_samples
    
    def build_pairwise_dataset(self) -> List[Dict[str, Any]]:
        """Build pairwise comparison dataset for RLHF"""
        validated_data = self._load_validated_data()
        pairwise_samples = []
        
        # Group by document type
        by_type = {}
        for data in validated_data:
            doc_type = data.get('doc_type', 'unknown')
            if doc_type not in by_type:
                by_type[doc_type] = []
            by_type[doc_type].append(data)
        
        # Create pairwise comparisons
        for doc_type, samples in by_type.items():
            if len(samples) >= 2:
                # Sort by validation score
                samples_sorted = sorted(
                    samples,
                    key=lambda x: x.get('validation', {}).get('score', 0),
                    reverse=True
                )
                
                # Create pairs (better vs worse)
                for i in range(min(5, len(samples_sorted) - 1)):  # Limit to 5 pairs per type
                    better = samples_sorted[i]
                    worse = samples_sorted[-(i+1)]
                    
                    pairwise_sample = {
                        "prompt": f"Analyze this {doc_type} document: {better.get('content', '')[:500]}...",
                        "chosen": json.dumps(better.get('annotations', {})),
                        "rejected": json.dumps(worse.get('annotations', {}))
                    }
                    pairwise_samples.append(pairwise_sample)
        
        return pairwise_samples
    
    def export_jsonl(self, dataset: List[Dict[str, Any]], output_path: str):
        """Export dataset to JSONL file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for sample in dataset:
                f.write(json.dumps(sample) + '\n')
        
        print(f"Exported {len(dataset)} samples to {output_path}")
    
    def export_all_formats(self):
        """Export all dataset formats"""
        # Export SFT
        sft_data = self.build_sft_dataset()
        self.export_jsonl(sft_data, os.path.join(self.data_dir, "train", "sft.jsonl"))
        
        # Export RLAIF
        rlaif_data = self.build_rlaif_dataset()
        self.export_jsonl(rlaif_data, os.path.join(self.data_dir, "train", "rlaif.jsonl"))
        
        # Export RLHF pairs
        pairwise_data = self.build_pairwise_dataset()
        self.export_jsonl(pairwise_data, os.path.join(self.data_dir, "train", "rlhf_pairs.jsonl"))
        
        # Create a summary
        summary = {
            "total_samples": len(sft_data),
            "sft_samples": len(sft_data),
            "rlaif_samples": len(rlaif_data),
            "pairwise_samples": len(pairwise_data),
            "export_timestamp": "auto_generated"
        }
        
        with open(os.path.join(self.data_dir, "train", "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
EOF
echo "Created src/core/packaging.py"

# 17. Utils - Logging
mkdir -p src/utils
cat > src/utils/logging.py << 'EOF'
import logging
import sys
from loguru import logger

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_logging():
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)
    
    # Remove default handler
    logger.remove()
    
    # Add custom handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Also log to file
    logger.add(
        "logs/pipeline.log",
        rotation="500 MB",
        retention="10 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO"
    )
    
    return logger
EOF
echo "Created src/utils/logging.py"

# 18. Utils - File Utilities
cat > src/utils/file_utils.py << 'EOF'
import os
import json
import hashlib
from typing import Dict, Any, Optional
import shutil

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def ensure_directory(path: str):
    """Ensure directory exists, create if not"""
    os.makedirs(path, exist_ok=True)

def save_json(data: Dict[str, Any], path: str, indent: int = 2):
    """Save dictionary as JSON file"""
    ensure_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)

def load_json(path: str) -> Optional[Dict[str, Any]]:
    """Load JSON file"""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def clean_directory(path: str):
    """Clean directory contents"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0
EOF
echo "Created src/utils/file_utils.py"

# 19. Create .env file
cat > .env << 'EOF'
# External Services
EXAOCR_URL=http://host.docker.internal:45001
VLLM_URL=http://host.docker.internal:45000

# Redis
REDIS_URL=redis://redis:6379/0

# Data paths
DATA_DIR=/app/data

# Processing settings
CHUNK_SIZE=2000
CHUNK_OVERLAP=200
MAX_CHUNK_LENGTH=3000

# Model settings
CLASSIFICATION_TEMPERATURE=0.1
ANNOTATION_TEMPERATURE=0.2
SYNTHESIS_TEMPERATURE=0.7

# Queue names (don't change unless you know what you're doing)
INGESTION_QUEUE=ingestion_queue
CLASSIFICATION_QUEUE=classification_queue
CHUNKING_QUEUE=chunking_queue
ANNOTATION_QUEUE=annotation_queue
SYNTHESIS_QUEUE=synthesis_queue
VALIDATION_QUEUE=validation_queue
EOF
echo "Created .env file"

# 20. Create setup script
cat > setup.sh << 'EOF'
#!/bin/bash

# setup.sh - Setup script for Construction AI Pipeline

echo "Setting up Construction AI Pipeline..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Create logs directory
mkdir -p logs

# Create data directories
mkdir -p data/{ingested,classified,chunks,annotated,synthetic,validated,train}

# Build and start services
echo "Building Docker images..."
docker-compose build

echo "Starting services..."
docker-compose up -d

echo ""
echo "✅ Setup complete!"
echo ""
echo "Services running:"
echo "  - API Gateway: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Flower Monitor: http://localhost:5555"
echo ""
echo "To upload PDFs:"
echo "  curl -X POST http://localhost:8000/api/v1/ingest -F 'files=@your-document.pdf'"
echo ""
echo "To check status:"
echo "  curl http://localhost:8000/api/v1/status/{doc_id}"
echo ""
echo "To export training data:"
echo "  curl http://localhost:8000/api/v1/export/training?format=sft"
echo ""
echo "To stop services:"
echo "  docker-compose down"
EOF

chmod +x setup.sh

# 21. Create quick test script
cat > test_pipeline.py << 'EOF'
#!/usr/bin/env python3
"""
Quick test script for the Construction AI Pipeline
"""

import requests
import os
import time

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("Testing Construction AI Pipeline API...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ API is healthy")
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False
    
    # Test root endpoint
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            print("✅ Root endpoint works")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Check if we can access external services
    print("\nChecking external services...")
    
    try:
        # Check exaOCR
        response = requests.get("http://localhost:45001/health")
        if response.status_code == 200:
            print("✅ exaOCR is accessible")
        else:
            print(f"⚠️  exaOCR responded with: {response.status_code}")
    except:
        print("❌ Cannot connect to exaOCR on port 45001")
    
    try:
        # Check vLLM
        response = requests.get("http://localhost:45000/v1/models")
        if response.status_code == 200:
            print("✅ vLLM is accessible")
        else:
            print(f"⚠️  vLLM responded with: {response.status_code}")
    except:
        print("❌ Cannot connect to vLLM on port 45000")
    
    return True

def test_queues():
    """Test Redis queue connection"""
    import redis
    from config.settings import settings
    
    print("\nTesting Redis connection...")
    
    try:
        r = redis.Redis.from_url(settings.redis_url)
        r.ping()
        print("✅ Redis is connected")
        
        # Test queue creation
        from src.services.queue_service import QueueService
        queue = QueueService()
        
        # Test enqueue
        result = queue.enqueue("test_queue", "test_task", ("arg1", "arg2"))
        print(f"✅ Can enqueue to Redis (result: {result})")
        
        # Test dequeue
        task = queue.dequeue("test_queue")
        if task:
            print("✅ Can dequeue from Redis")
        
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Construction AI Pipeline - Quick Test")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("docker-compose.yml"):
        print("❌ Please run this script from the project root directory")
        exit(1)
    
    # Test API
    if not test_api():
        print("\n❌ API tests failed. Make sure services are running.")
        print("Run: docker-compose up -d")
        exit(1)
    
    # Test Redis (optional - might need services running)
    try:
        test_queues()
    except:
        print("\n⚠️  Could not test Redis queues (services might not be running)")
    
    print("\n" + "=" * 60)
    print("✅ Quick test completed!")
    print("\nNext steps:")
    print("1. Upload a PDF: curl -X POST http://localhost:8000/api/v1/ingest -F 'files=@test.pdf'")
    print("2. Monitor progress: http://localhost:5555")
    print("3. Check API docs: http://localhost:8000/docs")
    print("=" * 60)
EOF

chmod +x test_pipeline.py

echo ""
echo "✅ All files created successfully!"
echo ""
echo "To get started:"
echo "1. Make sure exaOCR and vLLM are running on ports 45001 and 45000"
echo "2. Run: ./setup.sh"
echo "3. Test: python test_pipeline.py"
echo "4. Upload PDFs: curl -X POST http://localhost:8000/api/v1/ingest -F 'files=@your-document.pdf'"
echo ""
echo "The pipeline will process documents through all stages automatically!"
