# src/core/packaging.py
import os
import json
import glob
from typing import List, Dict, Any
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

class PackageBuilder:
    def __init__(self):
        self.data_dir = settings.data_dir
        self.train_dir = os.path.join(self.data_dir, "train")
        os.makedirs(self.train_dir, exist_ok=True)

    def _load_synthetic_and_validated(self) -> List[Dict[str, Any]]:
        """Load all synthetic variations (primary) and validated originals"""
        data = []
        
        # Load synthetic (preferred for diversity)
        synthetic_dir = os.path.join(self.data_dir, "synthetic")
        if os.path.exists(synthetic_dir):
            for path in glob.glob(f"{synthetic_dir}/**/*.json", recursive=True):
                try:
                    with open(path, 'r') as f:
                        item = json.load(f)
                        item['source'] = 'synthetic'
                        data.append(item)
                except Exception as e:
                    logger.warning(f"Failed to load synthetic {path}: {e}")
        
        # Load validated originals as fallback
        validated_dir = os.path.join(self.data_dir, "validated")
        if os.path.exists(validated_dir):
            for path in glob.glob(f"{validated_dir}/**/*_validated.json", recursive=True):
                try:
                    with open(path, 'r') as f:
                        item = json.load(f)
                        if item.get('validation', {}).get('valid', False):
                            item['source'] = 'validated'
                            data.append(item)
                except Exception as e:
                    logger.warning(f"Failed to load validated {path}: {e}")
        
        logger.info(f"Loaded {len(data)} samples for export ({sum(1 for d in data if d['source']=='synthetic')} synthetic)")
        return data

    def export_qwen3_sft_chat(self, min_quality: float = 0.0) -> str:
        """Export in Qwen3 chat format: messages array with user/assistant"""
        samples = self._load_synthetic_and_validated()
        output_path = os.path.join(self.train_dir, "qwen3_sft_chat.jsonl")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in samples:
                content = item.get('content', '').strip()
                annotations = item.get('annotations', {})
                quality = item.get('validation', {}).get('score', 1.0)
                
                if quality < min_quality or not content:
                    continue
                
                # Qwen3 chat format
                chat_sample = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Extract structured information from this construction document:\n\n{content}"
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps(annotations, indent=2, ensure_ascii=False)
                        }
                    ]
                }
                f.write(json.dumps(chat_sample, ensure_ascii=False) + "\n")
        
        logger.info(f"Qwen3 SFT chat dataset exported: {output_path} ({len(samples)} samples)")
        return output_path

    def export_all(self):
        """Export all formats (call after validation complete)"""
        self.export_qwen3_sft_chat()
        # Add more formats later if needed
        logger.info("All training datasets exported")