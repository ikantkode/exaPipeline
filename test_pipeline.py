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
