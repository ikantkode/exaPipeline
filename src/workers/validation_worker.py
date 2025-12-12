# src/workers/validation_worker.py
from celery import Celery
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

from config.settings import settings
from src.services.llm_service import LLMService
# After saving validated_data
from src.core.packaging import PackageBuilder  # Changed comment to match your request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Celery('tasks', broker=settings.redis_url)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    broker_connection_retry_on_startup=True,
)

def validate_with_llm(content: str, annotations: Dict[str, Any], is_synthetic: bool) -> Dict[str, Any]:
    """Validate data consistency using LLM"""
    llm_service = LLMService()

    system_prompt = "You are a strict data validator. Return ONLY valid JSON, no extra text."

    prompt = f"""Validate this {'synthetic' if is_synthetic else 'original'} construction document data.

Content (truncated): {content[:1500]}
Annotations: {json.dumps(annotations, indent=2)}

Check:
- Are annotations factually supported by content?
- Any hallucinations or invented facts?
- Is formatting consistent and clean?
- Does it look realistic for real-world construction docs?

Respond with JSON only:
{{
  "valid": true/false,
  "score": 0-100,
  "errors": [...],
  "warnings": [...]
}}"""

    try:
        result = llm_service.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=600
        )

        # Extract JSON block
        start = result.find('{')
        end = result.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in LLM response")
        json_str = result[start:end]
        return json.loads(json_str)

    except Exception as e:
        logger.warning(f"LLM validation failed: {e}")
        return {
            "valid": False,
            "score": 0,
            "errors": ["LLM validation failed or returned invalid JSON"],
            "warnings": []
        }

@app.task(name='validate_data')
def validate_data(doc_id: str, chunk_num: int, variation_num: int, is_synthetic: bool) -> None:
    """Validate original or synthetic data quality"""
    try:
        if is_synthetic:
            if variation_num == 0:
                logger.warning(f"Invalid: is_synthetic=True but variation_num=0 for {doc_id}")
                return
            source_dir = "synthetic"
            # ← FIXED: Match synthesis format (4-digit chunk, 2-digit variation)
            filename = f"chunk_{chunk_num:04d}_syn_{variation_num:02d}.json"
        else:
            source_dir = "annotated"
            # ← FIXED: Use 4-digit for consistency
            filename = f"chunk_{chunk_num:04d}_annotations.json"

        data_path = os.path.join(settings.data_dir, source_dir, doc_id, filename)
        if not os.path.exists(data_path):
            logger.error(f"File not found for validation: {data_path}")
            raise FileNotFoundError(data_path)

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        validation_result = validate_with_llm(
            data.get('content', ''),
            data.get('annotations', {}),
            is_synthetic
        )

        output_dir = os.path.join(settings.data_dir, "validated", doc_id)
        os.makedirs(output_dir, exist_ok=True)

        validated_data = {
            **data,
            "validation": validation_result,
            "validation_timestamp": datetime.now().isoformat(),
            "is_validated": True
        }

        if is_synthetic:
            # ← FIXED: 4-digit chunk, 2-digit variation
            output_file = f"chunk_{chunk_num:04d}_syn_{variation_num:02d}_validated.json"
        else:
            output_file = f"chunk_{chunk_num:04d}_validated.json"

        output_path = os.path.join(output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(validated_data, f, indent=2)

        # After saving validated_data
        PackageBuilder().export_qwen3_sft_chat()
        logger.info("Qwen3 chat dataset auto-generated")

        status = "synthetic" if is_synthetic else "original"
        logger.info(f"Validated {doc_id} chunk {chunk_num} ({status} variation {variation_num}) → {output_path}")

    except Exception as e:
        logger.error(f"Validation failed for {doc_id} chunk {chunk_num} (var {variation_num}): {e}")
        raise