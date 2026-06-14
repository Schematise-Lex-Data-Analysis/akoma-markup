"""LLM-based amendment application engine.

Applies amendments to base legislation markup using LLM prompts.
Supports replace, insert, and delete operations.
"""

import logging
import re
import time
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# ============================================================================
# Amendment Operation Prompts
# ============================================================================

AMENDMENT_REPLACE_PROMPT = """You are a legal document editor applying amendments to legislation.

TASK: Replace section {section_num} in the base markup with the new amended text.

RULES:
1. Replace ONLY the specified section, leave all other content unchanged
2. Maintain proper indentation and formatting
3. Preserve Laws.Africa markup syntax (SECTION, SUBSECTION, etc.)
4. Keep the same section number and heading format
5. Ensure proper hierarchy (subsections, clauses, subclauses)

BASE MARKUP:
{base_markup}

NEW TEXT FOR SECTION {section_num}:
{new_text}

CONTEXT:
This amendment is from {amendment_act}, effective {effective_date}.

OUTPUT ONLY the complete amended markup (base markup with the section replaced), no explanations:"""

AMENDMENT_INSERT_PROMPT = """You are a legal document editor applying amendments to legislation.

TASK: Insert a new section {section_num} into the base markup.

RULES:
1. Insert the new section at the appropriate position (maintaining numeric order)
2. Maintain proper indentation and formatting
3. Preserve Laws.Africa markup syntax (SECTION, SUBSECTION, etc.)
4. Keep all existing sections unchanged
5. Ensure proper hierarchy (subsections, clauses, subclauses)

BASE MARKUP:
{base_markup}

NEW SECTION TO INSERT ({section_num}):
{new_text}

CONTEXT:
This amendment is from {amendment_act}, effective {effective_date}.

OUTPUT ONLY the complete amended markup with the new section inserted, no explanations:"""

AMENDMENT_DELETE_PROMPT = """You are a legal document editor applying amendments to legislation.

TASK: Mark section {section_num} as REPEALED (deleted) in the base markup.

RULES:
1. Keep the section heading but mark it as "[REPEALED]"
2. Replace section content with "This section has been repealed by {amendment_act}."
3. Maintain proper indentation
4. Keep all other sections unchanged

BASE MARKUP:
{base_markup}

SECTION TO REPEAL: {section_num}

CONTEXT:
This repeal is from {amendment_act}, effective {effective_date}.

OUTPUT ONLY the complete amended markup with the section marked as repealed, no explanations:"""

DEFAULT_RATE_CONFIG = {
    "delay_between_requests": 5,
    "max_retries": 3,
    "initial_backoff": 10,
}

RETRYABLE_KEYWORDS = [
    "429", "rate limit", "ReadTimeout", "read timed out",
    "timeout", "timed out", "ConnectionError", "connection error",
    "connection reset", "connection aborted", "broken pipe",
    "remote disconnected", "service unavailable", "bad gateway",
    "gateway timeout", "500", "502", "503", "504",
    "internal server error",
]


def _build_amendment_prompt(operation: Literal["replace", "insert", "delete"]):
    """Build prompt template for amendment operation."""
    if operation == "replace":
        system_prompt = AMENDMENT_REPLACE_PROMPT
    elif operation == "insert":
        system_prompt = AMENDMENT_INSERT_PROMPT
    elif operation == "delete":
        system_prompt = AMENDMENT_DELETE_PROMPT
    else:
        raise ValueError(f"Invalid operation: {operation}")

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
    ])


def _is_retryable_error(exc: Exception) -> bool:
    """Check if error is retryable."""
    err_str = str(exc).lower()
    return any(kw.lower() in err_str for kw in RETRYABLE_KEYWORDS)


def apply_amendment(
    llm: BaseChatModel,
    base_markup: str,
    section_num: str,
    operation: Literal["replace", "insert", "delete"],
    new_text: str,
    amendment_act: str,
    effective_date: str,
    rate_config: dict | None = None,
) -> str:
    """Apply a single amendment to base markup using LLM.

    Args:
        llm: LangChain LLM instance.
        base_markup: Original legislation markup.
        section_num: Section number to amend (e.g., "43" or "43A").
        operation: Amendment type - "replace", "insert", or "delete".
        new_text: New text for replace/insert operations.
        amendment_act: Name of the amending act (for context).
        effective_date: Effective date (for context).
        rate_config: Rate limiting configuration.

    Returns:
        Amended markup string.

    Raises:
        ValueError: If operation is invalid.
        RuntimeError: If amendment fails after retries.
    """
    cfg = {**DEFAULT_RATE_CONFIG, **(rate_config or {})}
    prompt = _build_amendment_prompt(operation)
    chain = prompt | llm | StrOutputParser()

    retry_count = 0
    while retry_count < cfg["max_retries"]:
        try:
            result = chain.invoke({
                "base_markup": base_markup,
                "section_num": section_num,
                "new_text": new_text,
                "amendment_act": amendment_act,
                "effective_date": effective_date,
            })
            return result

        except Exception as exc:
            if _is_retryable_error(exc) and retry_count < cfg["max_retries"]:
                retry_count += 1
                wait = cfg["initial_backoff"] * (2 ** (retry_count - 1))
                logger.warning(
                    "Retryable error on amendment to section %s (attempt %d/%d), waiting %ds",
                    section_num, retry_count, cfg["max_retries"], wait
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Error applying amendment to section %s: %s",
                    section_num, str(exc)[:120]
                )
                raise

    raise RuntimeError(
        f"Failed to apply amendment to section {section_num} after retries"
    )


def apply_multiple_amendments(
    llm: BaseChatModel,
    base_markup: str,
    amendments: list[dict],
    rate_config: dict | None = None,
) -> tuple[str, list[dict]]:
    """Apply multiple amendments sequentially.

    Args:
        llm: LangChain LLM instance.
        base_markup: Original legislation markup.
        amendments: List of amendment dicts with keys:
            - section: str
            - operation: "replace" | "insert" | "delete"
            - new_text: str
            - amendment_act: str
            - effective_date: str
        rate_config: Rate limiting configuration.

    Returns:
        Tuple of (final_markup, errors).
        errors is a list of dicts with amendment info and error message.
    """
    cfg = {**DEFAULT_RATE_CONFIG, **(rate_config or {})}
    current_markup = base_markup
    errors = []

    for i, amendment in enumerate(amendments):
        logger.info(
            "Applying amendment %d/%d: section %s (%s)",
            i + 1, len(amendments),
            amendment["section"], amendment["operation"]
        )

        try:
            current_markup = apply_amendment(
                llm=llm,
                base_markup=current_markup,
                section_num=amendment["section"],
                operation=amendment["operation"],
                new_text=amendment.get("new_text", ""),
                amendment_act=amendment["amendment_act"],
                effective_date=amendment["effective_date"],
                rate_config=cfg,
            )

            # Rate limiting delay
            if i < len(amendments) - 1:
                time.sleep(cfg["delay_between_requests"])

        except Exception as exc:
            logger.error(
                "Failed to apply amendment to section %s: %s",
                amendment["section"], exc
            )
            errors.append({
                "amendment_index": i,
                "section": amendment["section"],
                "operation": amendment["operation"],
                "error": str(exc),
            })

    return current_markup, errors


def extract_section_from_markup(markup: str, section_num: str) -> str | None:
    """Extract a specific section from markup.

    Args:
        markup: Full legislation markup.
        section_num: Section number to extract.

    Returns:
        Section text if found, None otherwise.
    """
    lines = markup.split("\n")
    section_start = None
    section_lines = []

    # Find section start
    for i, line in enumerate(lines):
        if re.match(rf"^SECTION\s+{re.escape(section_num)}[.:\s]", line, re.IGNORECASE):
            section_start = i
            break

    if section_start is None:
        return None

    section_lines.append(lines[section_start])

    # Collect until next section at same or higher level
    for line in lines[section_start + 1:]:
        # Stop at next SECTION marker
        if re.match(r"^SECTION\s+\d+", line, re.IGNORECASE):
            break
        section_lines.append(line)

    return "\n".join(section_lines)


def validate_amended_markup(
    original_markup: str,
    amended_markup: str,
    amended_sections: list[str],
) -> dict[str, list[str]]:
    """Validate that amendment was applied correctly.

    Args:
        original_markup: Original markup before amendment.
        amended_markup: Markup after amendment.
        amended_sections: List of section numbers that were amended.

    Returns:
        Dictionary of validation issues by category.
    """
    issues: dict[str, list[str]] = {
        "missing_sections": [],
        "corrupted_structure": [],
        "unintended_changes": [],
    }

    # Check amended sections exist in output
    for section_num in amended_sections:
        if not extract_section_from_markup(amended_markup, section_num):
            issues["missing_sections"].append(section_num)

    # Check for preserved structure
    original_sections = set(re.findall(r"^SECTION\s+(\w+)", original_markup, re.MULTILINE | re.IGNORECASE))
    amended_sections_all = set(re.findall(r"^SECTION\s+(\w+)", amended_markup, re.MULTILINE | re.IGNORECASE))

    missing = original_sections - amended_sections_all - set(amended_sections)
    if missing:
        issues["corrupted_structure"].extend([f"Missing section: {s}" for s in missing])

    return {k: v for k, v in issues.items() if v}
