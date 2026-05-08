"""Multimodal vision-LLM client for page classification and extraction.

The deployment's API surface (chat/completions, Responses, or Azure AI
Inference) must be supplied explicitly — either via the ``api_mode`` constructor
argument or the ``AZURE_VISION_API_STYLE`` env var. Used by the table-rescue
path to (a) classify which pages contain tables (cheap YES/NO call) and
(b) extract page contents to markdown (heavier per-page call).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from .azure_api import ApiMode, validate_api_mode
from ..pdf.images import image_to_data_url

logger = logging.getLogger(__name__)


# Azure's Responses API enforces max_output_tokens >= 16.
_MIN_OUTPUT_TOKENS = 16
# Default per-page output budget for full-page markdown extraction. Dense
# multi-row schedule tables can spill past 4K tokens; 16K covers all but
# the most extreme cases. Tunable via AZURE_VISION_MAX_TOKENS / the
# ``extraction_max_tokens`` constructor arg.
_DEFAULT_EXTRACTION_TOKENS = 16384


def _to_int(s: str | None) -> int | None:
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


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
        extraction_max_tokens: int | None = None,
    ):
        self.api_key = api_key or os.environ.get("AZURE_VISION_KEY")
        self.endpoint = endpoint or os.environ.get("AZURE_VISION_ENDPOINT")
        self.deployment = deployment or os.environ.get("AZURE_VISION_MODEL")
        if not (self.api_key and self.endpoint and self.deployment):
            raise ValueError(
                "VisionClient requires api_key, endpoint, and deployment "
                "(or AZURE_VISION_KEY / AZURE_VISION_ENDPOINT / "
                "AZURE_VISION_MODEL env vars)."
            )

        resolved_mode = api_mode or os.environ.get("AZURE_VISION_API_STYLE")
        self.api_mode: ApiMode = validate_api_mode(
            resolved_mode, "AZURE_VISION_API_STYLE"
        )

        self.max_tokens = max(max_tokens, _MIN_OUTPUT_TOKENS)
        self.temperature = temperature
        self.extraction_max_tokens = max(
            (
                extraction_max_tokens
                or _to_int(os.environ.get("AZURE_VISION_MAX_TOKENS"))
                or _DEFAULT_EXTRACTION_TOKENS
            ),
            _MIN_OUTPUT_TOKENS,
        )

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

    def _complete(
        self,
        image: Image.Image,
        prompt: str,
        detail: str,
        max_tokens: int,
    ) -> tuple[str, str]:
        """Send ``prompt + image`` and return ``(text, finish_reason)``.

        ``finish_reason`` is normalised across the three API styles:
        ``"length"`` when the model stopped because of the output cap,
        otherwise the raw provider value (typically ``"stop"``).
        """
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
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
            choice = response.choices[0]
            text = (choice.message.content or "").strip()
            return text, (choice.finish_reason or "stop")

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
                max_output_tokens=max_tokens,
                temperature=self.temperature,
            )
            text = (getattr(response, "output_text", None) or "").strip()
            finish = "stop"
            if getattr(response, "status", None) == "incomplete":
                reason = getattr(
                    getattr(response, "incomplete_details", None),
                    "reason",
                    None,
                )
                finish = "length" if reason == "max_output_tokens" else (reason or "incomplete")
            return text, finish

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
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        return text, (str(choice.finish_reason) if choice.finish_reason else "stop")

    def ask(
        self,
        image: Image.Image,
        prompt: str,
        detail: str = "low",
        max_tokens: int | None = None,
    ) -> str:
        """Send ``prompt + image`` and return the raw text response."""
        budget = max(max_tokens or self.max_tokens, _MIN_OUTPUT_TOKENS)
        text, _ = self._complete(image, prompt, detail, budget)
        return text

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

    def extract_pages(
        self,
        page_images: dict[int, Image.Image],
        prompt: str,
        max_workers: int = 4,
        detail: str = "high",
        max_tokens: int | None = None,
        on_page_done=None,
    ) -> dict[int, str]:
        """Render each page image to text via the vision LLM, in parallel.

        ``max_tokens`` defaults to the client's ``extraction_max_tokens``
        (resolved from ``AZURE_VISION_MAX_TOKENS`` or 16384).

        ``on_page_done(page_number, response)`` is invoked from the worker
        thread immediately after a successful page so the caller can
        checkpoint to disk between pages. Must be thread-safe.

        Pages whose response was truncated by the output cap (finish
        reason ``"length"``) are reported in a single stderr warning at
        the end of the batch — the partial markdown is still kept and
        returned so the caller can inspect what came back.

        Raises if any page errors; partial results already handed to
        ``on_page_done`` are preserved on disk.
        """
        budget = max(max_tokens or self.extraction_max_tokens, _MIN_OUTPUT_TOKENS)
        results: dict[int, str] = {}
        truncated: list[int] = []

        def _do_one(img):
            return self._complete(img, prompt, detail, budget)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_do_one, img): pnum
                for pnum, img in page_images.items()
            }
            for fut in as_completed(futures):
                pnum = futures[fut]
                text, finish = fut.result()
                results[pnum] = text
                if finish == "length":
                    truncated.append(pnum)
                if on_page_done is not None:
                    on_page_done(pnum, text)

        if truncated:
            logger.warning(
                "%d page(s) hit the max_tokens=%d output cap and were "
                "truncated: pages %s. Set AZURE_VISION_MAX_TOKENS to a "
                "larger value (your deployment supports up to its own "
                "per-call output ceiling) or pass --azure-vision-max-tokens.",
                len(truncated), budget, sorted(truncated),
            )

        return results
