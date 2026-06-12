"""LLM-based conversion of Gazette sections to Akoma Ntoso markup."""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# ============================================================================
# PROMPTS (mirroring conversion.py structure)
# ============================================================================

GAZETTE_CONVERSION_PROMPT_TEMPLATE = """You are converting Gazette sections to Laws.Africa plaintext markup format.

CONTEXT:
This is a section from a Gazette of India notification with hierarchical structure.

MARKUP RULES:
1. Rule format: RULE [num]. - [heading]
2. Indent content under rules with 2 spaces
3. Subrules: SUBRULE (1), SUBRULE (2), etc. - indented under RULE
4. Clauses: CLAUSE (a), CLAUSE (b), etc. - indented under SUBRULE
5. Subclauses: SUBCLAUSE (i), SUBCLAUSE (ii), etc.
6. Provisos: Start with "Provided that" as separate indented paragraph
7. Explanations: Start with "Explanation.—" as separate indented paragraph
8. Schedules: Begin with "SCHEDULE" heading
9. Table placeholders: ``<<TABLE_REGION:N>>`` are STRUCTURAL PLACEHOLDERS.
   - COPY each placeholder verbatim at the same indentation
   - Do NOT paraphrase, expand, remove, or convert these tokens
   - Post-processing will splice in actual TABLE blocks

HIERARCHY LEVEL:
This section is at hierarchy_level={hierarchy_level} where:
- 1 = Main rule (RULE)
- 2 = Subrule (SUBRULE)
- 3 = Clause (CLAUSE)
- 4 = Subclause (SUBCLAUSE)

PARENT CONTEXT:
{parent_context}

EXAMPLE OUTPUT:
```
RULE 3. - Definitions
  In these rules, unless the context otherwise requires,—

  SUBRULE (1)
    CLAUSE (a)
      "Act" means the Information Technology Act, 2000;

    CLAUSE (b)
      "Appellate Body" means...

  SUBRULE (2)
    Words and expressions used herein...
```

Preserve exact legal text. Do not paraphrase."""

GAZETTE_HUMAN_PROMPT_TEMPLATE = """Convert this Gazette section to Laws.Africa markup:

Section Number: {section_num}
Section Heading: {section_heading}
Hierarchy Level: {hierarchy_level}
Parent Section: {parent_section}

Section Content:
{section_content}

Output ONLY the markup, no explanations:"""


# ============================================================================
# CONFIGURATION (mirroring conversion.py structure)
# ============================================================================

DEFAULT_RATE_CONFIG = {
    "delay_between_requests": 5,
    "batch_size": 3,
    "batch_delay": 30,
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


# ============================================================================
# FUNCTIONS (mirroring conversion.py structure)
# ============================================================================

def build_gazette_chain(
    llm: BaseChatModel,
    document_name: str = "Gazette",
):
    """Create LangChain chain for gazette section conversion.

    Args:
        llm: The language model to use for conversion.
        document_name: Name of the document being converted.

    Returns:
        A LangChain chain for converting sections to markup.
    """
    system_prompt = GAZETTE_CONVERSION_PROMPT_TEMPLATE
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", GAZETTE_HUMAN_PROMPT_TEMPLATE),
        ]
    )
    return prompt | llm | StrOutputParser()


def _load_checkpoint(path: Path) -> dict | None:
    """Load checkpoint from disk."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_checkpoint(path: Path, last_index: int, results: list, total: int):
    """Save checkpoint to disk."""
    data = {
        "last_completed_index": last_index,
        "completed_sections": results,
        "timestamp": datetime.now().isoformat(),
        "total_sections": total,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def process_gazette_sections(
    chain,
    sections: list[dict],
    checkpoint_path: str | Path | None = None,
    rate_config: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Process gazette sections with checkpointing and rate limiting.

    Mirrors process_all_sections() from conversion.py.
    Handles hierarchy_level and parent_section from TSV.

    Args:
        chain: LangChain chain returned by build_gazette_chain.
        sections: List of section dicts with 'num', 'heading', 'content',
                  'hierarchy_level', 'parent_section'.
        checkpoint_path: Full path to checkpoint file. None disables checkpointing.
        rate_config: Override default rate-limiting settings.

    Returns:
        Tuple of (converted_sections, errors).
    """
    cfg = {**DEFAULT_RATE_CONFIG, **(rate_config or {})}
    batch_size = cfg["batch_size"]
    delay = cfg["delay_between_requests"]

    results: list[dict] = []
    errors: list[dict] = []
    start_index = 0

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    if checkpoint_path:
        cp = _load_checkpoint(checkpoint_path)
        if cp:
            results = cp["completed_sections"]
            start_index = cp["last_completed_index"] + 1
            logger.info(
                "Resuming from checkpoint: section %d/%d",
                start_index + 1, len(sections),
            )

    for i in range(start_index, len(sections)):
        section = sections[i]
        retry_count = 0
        success = False

        while retry_count < cfg["max_retries"] and not success:
            try:
                result = chain.invoke(
                    {
                        "section_num": section["num"],
                        "section_heading": section["heading"],
                        "section_content": section["content"],
                        "hierarchy_level": section.get("hierarchy_level", 1),
                        "parent_section": section.get("parent_section", ""),
                        "parent_context": _build_parent_context(section, sections),
                    }
                )
                results.append({"num": section["num"], "markup": result})
                success = True

                pct = ((i + 1) / len(sections)) * 100
                logger.info(
                    "[%d/%d %.0f%%] Section %s: %s",
                    i + 1, len(sections), pct,
                    section["num"], section["heading"][:60],
                )

                time.sleep(delay)

                if (i + 1) % batch_size == 0:
                    if checkpoint_path:
                        _save_checkpoint(
                            checkpoint_path, i, results, len(sections)
                        )
                    logger.info(
                        "Batch %d done, cooling %ds",
                        (i + 1) // batch_size, cfg["batch_delay"],
                    )
                    time.sleep(cfg["batch_delay"])

            except Exception as exc:
                err_str = str(exc)
                is_retryable = any(
                    kw.lower() in err_str.lower() for kw in RETRYABLE_KEYWORDS
                )
                if is_retryable and retry_count < cfg["max_retries"]:
                    retry_count += 1
                    wait = cfg["initial_backoff"] * (2 ** (retry_count - 1))
                    logger.warning(
                        "Retryable error on section %s (attempt %d/%d), "
                        "waiting %ds",
                        section["num"], retry_count, cfg["max_retries"], wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Error on section %s: %s",
                        section["num"], err_str[:120],
                    )
                    errors.append({"num": section["num"], "error": err_str})
                    if checkpoint_path:
                        _save_checkpoint(
                            checkpoint_path, i, results, len(sections)
                        )
                    break

        if not success:
            logger.error(
                "Skipping section %s after %d retries",
                section["num"], retry_count,
            )

    if checkpoint_path:
        _save_checkpoint(
            checkpoint_path, len(sections) - 1, results, len(sections)
        )

    return results, errors


def _build_parent_context(
    section: dict,
    all_sections: list[dict],
) -> str:
    """Build context string from parent sections.

    Args:
        section: Current section dict.
        all_sections: All sections list.

    Returns:
        Context string summarizing parent section content.
    """
    parent_num = section.get("parent_section")
    if not parent_num:
        return "Top-level section (no parent)"

    # Find parent section
    parent = None
    for s in all_sections:
        if s["num"] == parent_num:
            parent = s
            break

    if not parent:
        return f"Parent section {parent_num} not found"

    # Build brief context
    heading = parent.get("heading", "")
    content_preview = parent.get("content", "")[:200]
    return f"Parent {parent_num}: {heading}\n{content_preview}..."


def assemble_gazette_markup(
    results: list[dict],
    document_name: str,
    table_regions: list[dict],
) -> str:
    """Assemble final markup from processed sections.

    Args:
        results: List of converted section results with 'num' and 'markup'.
        document_name: Name of the document.
        table_regions: List of table region dicts for orphaned tables.

    Returns:
        Complete markup string.
    """
    markup_parts = []

    # Document preamble
    markup_parts.append(f"# {document_name}")
    markup_parts.append("")

    # Add all converted sections
    for result in results:
        markup_parts.append(result["markup"])
        markup_parts.append("")

    # Handle any orphaned table regions
    if table_regions:
        # Check which tables weren't inline-replaced
        processed_tables = set()
        for result in results:
            # Extract TABLE_REGION references from markup
            refs = re.findall(r"<<TABLE_REGION:(\d+)>>", result["markup"])
            processed_tables.update(int(r) for r in refs)

        # Add unprocessed tables at end
        unprocessed = [
            r for r in table_regions if r["id"] not in processed_tables
        ]
        if unprocessed:
            markup_parts.append("")
            markup_parts.append("# APPENDIX - TABLES")
            markup_parts.append("")
            for region in unprocessed:
                markup_parts.append(f"<<TABLE_REGION:{region['id']}>>")
                markup_parts.append("")

    return "\n".join(markup_parts)
