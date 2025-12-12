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
