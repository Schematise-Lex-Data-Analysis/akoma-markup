"""Azure Document Intelligence (OCR) client used by the table-rescue path."""

import base64
import time
from pathlib import Path
from typing import Optional, Union

import requests


class AzureOCR:
    """Azure AI Document Intelligence (OCR) service wrapper."""

    def __init__(
        self,
        api_key: Optional[str],
        endpoint: Optional[str],
        model: Optional[str],
        timeout: int = 120,
        max_retries: int = 3
    ):
        if not api_key:
            raise ValueError(
                "Azure OCR key required. Provide via api_key parameter "
                "or set AZURE_OCR_KEY environment variable."
            )
        if not endpoint:
            raise ValueError(
                "Azure OCR endpoint required. Provide via endpoint parameter "
                "or set AZURE_OCR_ENDPOINT environment variable."
            )
        if not model:
            raise ValueError(
                "Azure OCR model required. Provide via model parameter "
                "or set AZURE_OCR_MODEL environment variable."
            )

        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def extract_text(self, pdf_path: Union[str, Path], include_images: bool = False) -> str:
        """Extract text from a PDF using Azure OCR. Returns markdown text."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"Extracting text from {pdf_path.name}...")

        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()

        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')

        payload = {
            "model": self.model,
            "document": {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{pdf_base64}"
            },
            "include_image_base64": include_images
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()

                    all_text = []
                    if 'pages' in result:
                        for page in result['pages']:
                            if 'markdown' in page and page['markdown']:
                                all_text.append(page['markdown'])

                    text = '\n'.join(all_text)

                    if text.strip():
                        print(f"Extracted {len(text)} characters from {len(result.get('pages', []))} pages")
                        return text
                    else:
                        raise ValueError("No text extracted from PDF")

                elif response.status_code == 502 and attempt < self.max_retries - 1:
                    print(f"502 error, retrying ({attempt + 1}/{self.max_retries})...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    response.raise_for_status()

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Error: {str(e)[:100]}..., retrying...")
                    time.sleep(2 ** attempt)
                else:
                    raise Exception(f"OCR extraction failed after {self.max_retries} attempts: {e}")

        raise Exception("OCR extraction failed")
