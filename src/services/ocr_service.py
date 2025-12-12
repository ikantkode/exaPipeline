# src/services/ocr_service.py
import requests
import time
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
import os

from config.settings import settings

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self):
        self.base_url = settings.exaocr_url.rstrip("/")
        logger.info(f"OCRService initialized with base_url: {self.base_url}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, TimeoutError, Exception)),
        reraise=True
    )
    def process_pdf(self, file_path: str, force_ocr: bool = True) -> Dict[str, Any]:
        """Process PDF through exaOCR API with correct nested response handling"""
        filename = os.path.basename(file_path)
        logger.info(f"Processing PDF: {filename}")

        # Upload file
        with open(file_path, 'rb') as f:
            files = {'files': (filename, f, 'application/pdf')}
            params = {'force_ocr': str(force_ocr).lower()}
            
            response = requests.post(
                f"{self.base_url}/upload/",
                files=files,
                params=params,
                timeout=300
            )
            response.raise_for_status()

        result = response.json()
        logger.info(f"exaOCR upload response: {result}")

        # Critical Fix: Extract from results[0]
        if 'results' not in result or not result['results']:
            raise Exception(f"Invalid exaOCR response - no results: {result}")

        first_result = result['results'][0]
        file_id = first_result.get('file_id')
        md_id = first_result.get('markdown_id')  # Some responses have it directly

        if not file_id:
            raise Exception(f"No file_id in results[0]: {first_result}")

        logger.info(f"File uploaded successfully, file_id: {file_id}")

        # If markdown_id already present (fast path), skip polling
        if md_id:
            logger.info(f"Markdown already available, md_id: {md_id}")
        else:
            md_id = self._wait_for_processing(file_id)

        markdown = self._download_markdown(md_id)
        self._cleanup(file_id)

        return {
            'markdown': markdown,
            'file_id': file_id,
            'md_id': md_id,
            'engine': 'exaOCR'
        }

    def _wait_for_processing(self, file_id: str, timeout: int = 600) -> str:
        start_time = time.time()
        logger.info(f"Polling for completion: file_id={file_id}")

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/progress/{file_id}", timeout=30)
                if response.status_code == 200:
                    prog = response.json()
                    logger.debug(f"Progress: {prog}")
                    status = prog.get('status', '').lower()
                    if status == 'completed':
                        md_id = prog.get('md_id') or prog.get('markdown_id')
                        if not md_id:
                            raise Exception(f"Completed but no md_id: {prog}")
                        logger.info(f"Processing completed, md_id: {md_id}")
                        return md_id
                    elif status in ['failed', 'error']:
                        raise Exception(f"OCR failed: {prog}")
                elif response.status_code == 404:
                    logger.warning("Progress endpoint 404 - continuing poll")
            except Exception as e:
                logger.warning(f"Progress poll error: {e}")

            time.sleep(5)

        raise TimeoutError(f"OCR timeout for file_id {file_id}")

    def _download_markdown(self, md_id: str) -> str:
        logger.info(f"Downloading markdown: {md_id}")
        response = requests.get(f"{self.base_url}/download-markdown/{md_id}", timeout=60)
        response.raise_for_status()
        markdown = response.text
        if not markdown.strip():
            logger.warning("Downloaded markdown is empty")
        else:
            logger.info(f"Markdown downloaded ({len(markdown)} chars)")
        return markdown

    def _cleanup(self, file_id: str):
        try:
            resp = requests.delete(f"{self.base_url}/cleanup/{file_id}", timeout=30)
            if resp.status_code in [200, 204]:
                logger.info(f"Cleanup OK: {file_id}")
            else:
                logger.warning(f"Cleanup {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")