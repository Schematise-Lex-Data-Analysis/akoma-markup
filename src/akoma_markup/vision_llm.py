"""Minimal vision-LLM client for per-page image classification.

The deployment's API surface (chat/completions, Responses, or Azure AI
Inference) must be supplied explicitly — either via the ``api_mode`` constructor
argument or the ``AZURE_MULTIMODAL_API_STYLE`` env var. Used by ``tables.py``'s
``auto`` mode to spot table pages before paying for OCR.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from .azure_api import ApiMode, validate_api_mode
from .pdf_to_image import image_to_data_url


# Azure's Responses API enforces max_output_tokens >= 16.
_MIN_OUTPUT_TOKENS = 16


class VisionClient:
    """Thin wrapper over a vision-capable Azure inference endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        deployment: str | None = None,
        api_mode: str | None = None,
        max_tokens: int = _MIN_OUTPUT_TOKENS,
        temperature: float = 0.0,
    ):
        self.api_key = api_key or os.environ.get("AZURE_API_KEY")
        self.endpoint = endpoint or os.environ.get("AZURE_MULTIMODAL_ENDPOINT")
        self.deployment = deployment or os.environ.get("AZURE_MULTIMODAL_DEPLOYMENT")
        if not (self.api_key and self.endpoint and self.deployment):
            raise ValueError(
                "VisionClient requires api_key, endpoint, and deployment "
                "(or AZURE_API_KEY / AZURE_MULTIMODAL_ENDPOINT / "
                "AZURE_MULTIMODAL_DEPLOYMENT env vars)."
            )

        resolved_mode = api_mode or os.environ.get("AZURE_MULTIMODAL_API_STYLE")
        self.api_mode: ApiMode = validate_api_mode(
            resolved_mode, "AZURE_MULTIMODAL_API_STYLE"
        )

        self.max_tokens = max(max_tokens, _MIN_OUTPUT_TOKENS)
        self.temperature = temperature

        # Lazily constructed clients — pick the right SDK for the chosen mode.
        self._openai_client = None
        self._inference_client = None

    def _openai(self):
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(
                base_url=self.endpoint, api_key=self.api_key
            )
        return self._openai_client

    def _inference(self):
        if self._inference_client is None:
            from azure.ai.inference import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential

            self._inference_client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key),
            )
        return self._inference_client

    def ask(self, image: Image.Image, prompt: str, detail: str = "low") -> str:
        """Send ``prompt + image`` and return the raw text response."""
        data_url = image_to_data_url(image, format="PNG")

        if self.api_mode == "chat":
            response = self._openai().chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url, "detail": detail},
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return (response.choices[0].message.content or "").strip()

        if self.api_mode == "responses":
            response = self._openai().responses.create(
                model=self.deployment,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": data_url,
                                "detail": detail,
                            },
                        ],
                    }
                ],
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return (getattr(response, "output_text", None) or "").strip()

        # azure-inference
        response = self._inference().complete(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": detail},
                        },
                    ],
                }
            ],
            model=self.deployment,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def classify_pages(
        self,
        page_images: dict[int, Image.Image],
        prompt: str,
        max_workers: int = 8,
        detail: str = "low",
    ) -> dict[int, str]:
        """Run ``ask(image, prompt)`` over many pages in parallel.

        Per-page failures are recorded as ``"ERROR: <message>"`` so a single
        bad page never aborts the whole batch.
        """
        results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.ask, img, prompt, detail): pnum
                for pnum, img in page_images.items()
            }
            for fut in as_completed(futures):
                pnum = futures[fut]
                try:
                    results[pnum] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    results[pnum] = f"ERROR: {exc}"
        return results
