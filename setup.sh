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
