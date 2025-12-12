# src/services/llm_service.py
import requests
import json
from typing import Dict, Any, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.base_url = settings.vllm_url.rstrip("/")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True
    )
    def generate(self, 
                 prompt: str, 
                 temperature: float = 0.7,
                 max_tokens: int = 2000,
                 system_prompt: Optional[str] = None) -> str:
        """Generate text using vLLM OpenAI-compatible chat endpoint"""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": "/model",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=180
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            if 'response' in locals():
                logger.error(f"Response: {response.text}")
            raise

    def classify_document(self, content: str) -> str:
        """Classify document type using LLM"""
        system_prompt = "You are an expert construction document classifier. Return ONLY one word: the exact document type from the list."

        prompt = f"""Classify this document into EXACTLY ONE of these types (lowercase, single word):

certified_payroll, submittal, specification, contract, invoice, receipt, delay_report, email, check_copy

Document:
{content[:3000]}

Return ONLY the type."""
        
        try:
            result = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=20
            )
            
            doc_type = result.strip().lower().split()[0]
            valid_types = [t.lower() for t in settings.DOCUMENT_TYPES]
            
            if doc_type in valid_types:
                logger.info(f"Classification successful: {doc_type}")
                return doc_type
            else:
                logger.warning(f"Invalid classification '{doc_type}', using 'unknown'")
                return "unknown"
                
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return "unknown"

    def extract_annotations(self, content: str, doc_type: str) -> Dict[str, Any]:
        """Extract structured entities from document chunk"""
        system_prompt = f"""You are an expert construction document analyst. Extract structured information from this {doc_type} document.

Focus on:
- Dates (issue date, due date, period covered)
- Company names (contractor, owner, subcontractor)
- People names (project manager, contact person)
- Monetary amounts (bid amount, contract value, payments)
- Compliance status
- Action items
- Tables (if present)

Return ONLY valid JSON with these keys. Use empty list/string if not found."""

        prompt = f"""Document type: {doc_type}

Content:
{content}

Extract entities and return JSON only."""

        try:
            result = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=settings.annotation_temperature,
                max_tokens=3000
            )
            
            # Find JSON in response
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
                
            json_str = result[json_start:json_end]
            annotations = json.loads(json_str)
            
            logger.info(f"Annotation extraction successful for {doc_type}")
            return annotations
            
        except Exception as e:
            logger.error(f"Annotation extraction failed: {e}")
            logger.error(f"Raw response: {result[:500] if 'result' in locals() else 'N/A'}")
            return {"error": "extraction_failed", "raw_response": result if 'result' in locals() else ""}

    def generate_synthetic_variations(self, content: str, annotations: Dict[str, Any], num_variations: int = 3) -> List[Dict[str, Any]]:
        """Generate synthetic variations of a chunk with annotations"""
        system_prompt = "You are a construction document data synthesizer. Generate realistic variations of the given content while preserving meaning and structure."

        prompt = f"""Original content:
{content}

Original annotations:
{json.dumps(annotations, indent=2)}

Generate {num_variations} realistic variations of this construction document chunk.

For each variation:
1. Change company names, dates, amounts
2. Paraphrase wording
3. Maintain realistic construction document style
4. Keep the same structure
5. Preserve all key facts and intent

Return ONLY a JSON array of objects with 'content' and 'annotations' keys."""

        try:
            result = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=settings.synthesis_temperature,
                max_tokens=4000
            )
            
            # Find JSON array
            json_start = result.find('[')
            json_end = result.rfind(']') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON array found in response")
                
            json_str = result[json_start:json_end]
            variations = json.loads(json_str)
            
            if not isinstance(variations, list):
                raise ValueError("Response is not a list")
                
            logger.info(f"Generated {len(variations)} synthetic variations")
            return variations
            
        except Exception as e:
            logger.error(f"Synthetic generation failed: {e}")
            logger.error(f"Raw response: {result[:500] if 'result' in locals() else 'N/A'}")
            return []