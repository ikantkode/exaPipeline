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
