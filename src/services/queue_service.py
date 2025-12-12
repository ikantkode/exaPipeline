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
        """Enqueue a task to Redis queue in Celery-compatible format"""
        # Celery expects a specific message format
        task = {
            'task': task_name,
            'id': f'task_{queue_name}_{id(self)}',  # Unique ID
            'args': args,
            'kwargs': kwargs or {},
            'retries': 0,
            'eta': None,
            'expires': None,
            'utc': True,
            'callbacks': None,
            'errbacks': None,
            'timelimit': [None, None],
            'taskset': None,
            'chord': None,
        }
        
        # Push to Redis list (queue)
        result = self.redis_client.rpush(queue_name, json.dumps(task))
        
        logger.info(f"Enqueued task {task_name} to {queue_name}")
        return str(result)
    
    def enqueue_celery_format(self, queue_name: str, task_name: str, args: tuple) -> str:
        """Alternative: Use simpler format that matches Celery expectations"""
        # Simple format that avoids the 'properties' key issue
        message = {
            'body': json.dumps({
                'task': task_name,
                'args': args,
                'kwargs': {},
                'id': f'task_{queue_name}_{id(self)}'
            }),
            'content-type': 'application/json',
            'content-encoding': 'utf-8',
            'headers': {},
            'properties': {
                'body_encoding': 'base64',
                'correlation_id': f'task_{queue_name}_{id(self)}',
                'reply_to': None,
                'delivery_mode': 2,
                'delivery_info': {
                    'exchange': '',
                    'routing_key': queue_name,
                },
                'priority': 0,
                'delivery_tag': f'task_{queue_name}_{id(self)}',
            }
        }
        
        result = self.redis_client.rpush(queue_name, json.dumps(message))
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