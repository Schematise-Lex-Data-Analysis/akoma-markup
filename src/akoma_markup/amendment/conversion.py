"""LLM-based conversion of gazette notifications to Akoma Ntoso markup."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image

from .prompts import (
    GAZETTE_CONVERSION_PROMPT_TEMPLATE,
    GAZETTE_HUMAN_PROMPT_TEMPLATE,
    GAZETTE_PAGE_CONVERSION_PROMPT,
    GAZETTE_MERGE_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

DEFAULT_RATE_CONFIG = {
    "delay_between_requests": 5,
    "max_retries": 3,
    "initial_backoff": 10,
}

RETRYABLE_KEYWORDS = [
    "429",
    "rate limit",
    "ReadTimeout",
    "read timed out",
    "timeout",
    "timed out",
    "ConnectionError",
    "connection error",
    "connection reset",
    "connection aborted",
    "broken pipe",
    "remote disconnected",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "500",
    "502",
    "503",
    "504",
    "internal server error",
]


def build_gazette_chain(llm: BaseChatModel, document_name: str = "Gazette Notification"):
    """Create a LangChain chain for gazette conversion.

    Args:
        llm: The language model to use for conversion.
        document_name: Name of the gazette document.

    Returns:
        A LangChain chain for converting gazettes to markup.
    """
    system_prompt = GAZETTE_CONVERSION_PROMPT_TEMPLATE
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", GAZETTE_HUMAN_PROMPT_TEMPLATE),
        ]
    )
    return prompt | llm | StrOutputParser()


def convert_gazette_text(
    chain,
    gazette_text: str,
    rate_config: dict | None = None,
) -> str:
    """Convert entire gazette text to markup.

    Args:
        chain: LangChain chain returned by ``build_gazette_chain``.
        gazette_text: Full text extracted from gazette PDF.
        rate_config: Override default rate-limiting settings.

    Returns:
        Converted markup text.
    """
    cfg = {**DEFAULT_RATE_CONFIG, **(rate_config or {})}
    retry_count = 0
    success = False

    while retry_count < cfg["max_retries"] and not success:
        try:
            result = chain.invoke({"gazette_text": gazette_text})
            success = True
            return result

        except Exception as exc:
            err_str = str(exc)
            is_retryable = any(
                kw.lower() in err_str.lower() for kw in RETRYABLE_KEYWORDS
            )
            if is_retryable and retry_count < cfg["max_retries"]:
                retry_count += 1
                wait = cfg["initial_backoff"] * (2 ** (retry_count - 1))
                logger.warning(
                    "Retryable error on gazette conversion (attempt %d/%d), "
                    "waiting %ds",
                    retry_count, cfg["max_retries"], wait,
                )
                time.sleep(wait)
            else:
                logger.error("Error converting gazette: %s", err_str[:120])
                raise

    raise RuntimeError(
        f"Failed to convert gazette after {retry_count} retries"
    )


def convert_gazette_page_with_vision(
    vision_client,
    page_image: Image.Image,
    page_num: int,
    total_pages: int,
    previous_context: str = "",
    rate_config: dict | None = None,
) -> str:
    """Convert a single gazette page using multimodal vision LLM.

    Args:
        vision_client: VisionClient instance for Azure OpenAI.
        page_image: PIL Image of the page.
        page_num: Current page number (1-indexed).
        total_pages: Total number of pages.
        previous_context: Text context from previous pages for continuity.
        rate_config: Rate limiting configuration.

    Returns:
        Converted markup text for this page.
    """
    cfg = {**DEFAULT_RATE_CONFIG, **(rate_config or {})}
    prompt = GAZETTE_PAGE_CONVERSION_PROMPT.format(
        page_num=page_num,
        total_pages=total_pages,
        previous_context=previous_context if previous_context else "None",
    )
    
    # DEBUG: Show prompt
    print(f"\n=== DEBUG: Prompt for page {page_num} ===")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Prompt preview (first 300 chars):\n{prompt[:300]}...")
    print("=== END DEBUG ===\n")

    retry_count = 0
    while retry_count < cfg["max_retries"]:
        try:
            # Use high token limit for text extraction (dense gazette pages)
            result = vision_client.ask(
                image=page_image,
                prompt=prompt,
                detail="high",
                max_tokens=8192,  # Increased from default 16 tokens
            )
            return result

        except Exception as exc:
            err_str = str(exc)
            is_retryable = any(
                kw.lower() in err_str.lower() for kw in RETRYABLE_KEYWORDS
            )
            if is_retryable and retry_count < cfg["max_retries"]:
                retry_count += 1
                wait = cfg["initial_backoff"] * (2 ** (retry_count - 1))
                logger.warning(
                    "Retryable error on page %d (attempt %d/%d), waiting %ds",
                    page_num, retry_count, cfg["max_retries"], wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Error on page %d: %s", page_num, err_str[:120]
                )
                raise

    raise RuntimeError(
        f"Failed to convert page {page_num} after {retry_count} retries"
    )


def convert_gazette_pages_with_vision(
    vision_client,
    page_images: dict[int, Image.Image],
    checkpoint_path: Path | None = None,
    rate_config: dict | None = None,
) -> dict[int, str]:
    """Convert multiple gazette pages using multimodal vision LLM.

    Processes pages sequentially to maintain context continuity.

    Args:
        vision_client: VisionClient instance for Azure OpenAI.
        page_images: Dict mapping page numbers to PIL Images.
        checkpoint_path: Path to save checkpoint for resuming.
        rate_config: Rate limiting configuration.

    Returns:
        Dict mapping page numbers to converted markup.
    """
    cfg = {**DEFAULT_RATE_CONFIG, **(rate_config or {})}
    results: dict[int, str] = {}
    page_nums = sorted(page_images.keys())
    total_pages = len(page_nums)

    # Try to load checkpoint
    if checkpoint_path and checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                cp = json.load(f)
                results = {int(k): v for k, v in cp.get("pages", {}).items()}
                logger.info(
                    "Resuming from checkpoint: %d/%d pages already processed",
                    len(results), total_pages,
                )
        except Exception as exc:
            logger.warning("Failed to load checkpoint: %s", exc)

    previous_context = ""

    for i, page_num in enumerate(page_nums):
        if page_num in results:
            # Use existing result to build context
            previous_context = results[page_num][-500:]  # Last 500 chars
            continue

        logger.info("Processing page %d/%d", page_num, total_pages)
        page_image = page_images[page_num]

        try:
            markup = convert_gazette_page_with_vision(
                vision_client=vision_client,
                page_image=page_image,
                page_num=page_num,
                total_pages=total_pages,
                previous_context=previous_context,
                rate_config=cfg,
            )
            results[page_num] = markup
            
            # DEBUG: Show what we got from the vision model
            print(f"\n=== DEBUG: Page {page_num} ===")
            print(f"Markup length: {len(markup)} chars")
            print(f"First 500 chars:\n{markup[:500]}")
            if len(markup) > 500:
                print(f"... (truncated, total {len(markup)} chars)")
            print("=== END DEBUG ===\n")

            # Update context for next page (last 500 chars for continuity)
            previous_context = markup[-500:] if markup else ""

            # Save checkpoint
            if checkpoint_path:
                _save_gazette_checkpoint(checkpoint_path, results, total_pages)

            # Rate limiting delay
            time.sleep(cfg["delay_between_requests"])

        except Exception as exc:
            logger.error("Failed to process page %d: %s", page_num, exc)
            raise

    return results


def _save_gazette_checkpoint(path: Path, pages: dict[int, str], total: int):
    """Save checkpoint for gazette conversion."""
    data = {
        "pages": {str(k): v for k, v in pages.items()},
        "timestamp": datetime.now().isoformat(),
        "total_pages": total,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def merge_gazette_pages(
    page_markups: dict[int, str],
    llm: BaseChatModel | None = None,
) -> str:
    """Merge page markups into a single coherent document.

    If llm is provided, uses it to intelligently merge and deduplicate.
    Otherwise, simple concatenation with page separators.

    Args:
        page_markups: Dict mapping page numbers to markup strings.
        llm: Optional LangChain model for intelligent merging.

    Returns:
        Merged markup text.
    """
    if not page_markups:
        return ""

    # Sort by page number
    sorted_pages = sorted(page_markups.items())

    # If only one page, return it directly
    if len(sorted_pages) == 1:
        return sorted_pages[0][1]

    # Without LLM, do simple concatenation with some basic deduplication
    if llm is None:
        lines: list[str] = []
        last_few_lines: set[str] = set()

        for page_num, markup in sorted_pages:
            page_lines = markup.strip().split("\n")
            for line in page_lines:
                # Simple deduplication: skip if exact match with recent lines
                if line.strip() and line not in last_few_lines:
                    lines.append(line)
                    # Keep last 20 lines for dedup check
                    last_few_lines.add(line)
                    if len(last_few_lines) > 20:
                        last_few_lines.pop()

        return "\n".join(lines)

    # With LLM, use intelligent merging
    combined = "\n\n---PAGE {}---\n\n".join(
        f"{k}\n{v}" for k, v in sorted_pages
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", GAZETTE_MERGE_PROMPT_TEMPLATE),
        ("human", "{page_markups}"),
    ])
    chain = prompt | llm | StrOutputParser()

    try:
        result = chain.invoke({
            "num_pages": len(sorted_pages),
            "page_markups": combined,
        })
        return result
    except Exception as exc:
        logger.warning("LLM merge failed (%s), using simple concatenation", exc)
        return "\n\n".join(v for _, v in sorted_pages)
