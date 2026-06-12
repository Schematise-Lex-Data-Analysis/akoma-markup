"""Gazette refine module for hierarchical section grouping.

Concatenates subsections under main sections and splits only when
token limits are crossed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..util.llm.factory import build_llm
from .conversion import DEFAULT_RATE_CONFIG, RETRYABLE_KEYWORDS

logger = logging.getLogger(__name__)

TokenizerType = Literal["tiktoken", "approx"]
ChunkStrategy = Literal["hierarchy", "semantic"]

GAZETTE_REFINE_PROMPT_TEMPLATE = """You are converting Gazette sections to Laws.Africa plaintext markup format.

CONTEXT:
This is a grouped section from a Gazette of India notification containing
multiple hierarchical subsections that have been concatenated for processing.

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

HIERARCHY HANDLING:
- The input contains multiple hierarchical levels concatenated together
- Maintain proper indentation for each nested level
- Use the section numbers to determine hierarchy (main numbers = RULE,
  (1)(2) = SUBRULE, (a)(b) = CLAUSE, (i)(ii) = SUBCLAUSE)
- Preserve parent-child relationships within the grouped content

EXAMPLE OUTPUT:
```
RULE 3. - Due diligence by an intermediary
  An intermediary shall observe the following due diligence...

  SUBRULE (1)
    Paragraph text here...

    CLAUSE (a)
      Sub-paragraph content...

    CLAUSE (b)
      Sub-paragraph content...

  SUBRULE (2)
    Grievance redressal mechanism...

    CLAUSE (a)
      Requirement details...
```

Preserve exact legal text. Do not paraphrase. Maintain hierarchical
structure with proper indentation."""

GAZETTE_REFINE_HUMAN_TEMPLATE = """Convert this grouped Gazette section to Laws.Africa markup:

Main Section: {main_section_num} - {main_section_heading}
Hierarchy Levels: {hierarchy_levels}
Child Sections Included: {child_count}

Grouped Content:
{content}

Output ONLY the markup, no explanations:"""


def calculate_token_count(text: str, tokenizer: TokenizerType = "approx") -> int:
    """Calculate approximate token count for text.

    Args:
        text: The text to count tokens for.
        tokenizer: The tokenizer method to use.

    Returns:
        Estimated token count.
    """
    if tokenizer == "tiktoken":
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            logger.warning("tiktoken not available, using approx tokenizer")
            return calculate_token_count(text, "approx")

    # Approximate: ~4 characters per token for English text
    return len(text) // 4


def _build_section_tree(tsv_df: pd.DataFrame) -> dict:
    """Build a tree structure from hierarchical sections.

    Args:
        tsv_df: DataFrame with section data including hierarchy_level
            and parent_section.

    Returns:
        Dict mapping section_num to section data with 'children' list.
    """
    sections = {}
    section_order = []

    for idx, row in tsv_df.iterrows():
        # Use section_num if available, otherwise generate ID from row index
        section_num = row.get("section_num")
        if pd.isna(section_num) or str(section_num).strip() == "":
            section_num = f"_row_{idx}"
        else:
            section_num = str(section_num).strip()

        # Use heading/content directly from TSV columns
        heading = row.get("heading") if pd.notna(row.get("heading")) else ""
        content = row.get("content") if pd.notna(row.get("content")) else ""
        parent = row.get("parent_section") if pd.notna(row.get("parent_section")) else ""
        hierarchy = int(row.get("hierarchy_level", 1)) if pd.notna(row.get("hierarchy_level")) else 1
        page = int(row.get("page", 0)) if pd.notna(row.get("page")) else 0

        sections[section_num] = {
            "num": section_num,
            "heading": str(heading),
            "content": str(content),
            "page": page,
            "hierarchy_level": hierarchy,
            "parent_section": str(parent),
            "children": [],
            "_original_index": idx,
        }
        section_order.append((idx, section_num))

    def _find_existing_parent(parent_num: str) -> str | None:
        """Find closest existing ancestor for a parent reference."""
        if not parent_num:
            return None
        if parent_num in sections:
            return parent_num
        # Try stripping trailing elements (e.g., "4(1)" -> "4.")
        for i in range(len(parent_num) - 1, 0, -1):
            prefix = parent_num[:i]
            if prefix in sections:
                return prefix
        # Try adding dot suffix (e.g., "4" -> "4.")
        if parent_num + "." in sections:
            return parent_num + "."
        # Try stripping then adding dot (e.g., "4(1)" -> "4" -> "4.")
        for i in range(len(parent_num) - 1, 0, -1):
            prefix = parent_num[:i]
            if prefix + "." in sections:
                return prefix + "."
        return None

    # Build parent-child relationships
    for section_num, section in sections.items():
        parent_num = section["parent_section"]
        existing_parent = _find_existing_parent(parent_num)
        if existing_parent:
            sections[existing_parent]["children"].append(section_num)

    # Sort children by their original index to maintain TSV order
    for section in sections.values():
        section["children"].sort(
            key=lambda x: sections[x]["_original_index"]
        )

    # Store order for later use
    sections["_section_order"] = section_order

    return sections


def _get_descendants(
    section_num: str,
    sections: dict,
    visited: set | None = None,
) -> list[str]:
    """Get all descendant sections in order.

    Args:
        section_num: The parent section number.
        sections: Dictionary of all sections.
        visited: Set of already-visited section numbers (for cycle detection).

    Returns:
        List of section numbers in hierarchical order.
    """
    result = []
    if section_num not in sections:
        return result

    if visited is None:
        visited = set()

    # Prevent cycles
    if section_num in visited:
        return result
    visited.add(section_num)

    section = sections[section_num]

    # Add children in order
    for child_num in section.get("children", []):
        if child_num not in visited:
            result.append(child_num)
            result.extend(_get_descendants(child_num, sections, visited))

    return result


def _concatenate_section_group(
    main_section: dict,
    descendant_nums: list[str],
    sections: dict,
) -> str:
    """Concatenate main section with all descendants.

    Args:
        main_section: The main section dict.
        descendant_nums: List of descendant section numbers.
        sections: Dictionary of all sections.

    Returns:
        Concatenated content string.
    """
    parts = []

    # Add main section
    parts.append(f"[{main_section['num']}] {main_section['heading']}")
    parts.append(main_section["content"])
    parts.append("")

    # Add descendants
    for desc_num in descendant_nums:
        if desc_num in sections:
            desc = sections[desc_num]
            parts.append(f"[{desc['num']}] {desc['heading']}")
            parts.append(desc["content"])
            parts.append("")

    return "\n".join(parts)


def group_sections_hierarchically(
    tsv_df: pd.DataFrame,
) -> list[dict]:
    """Group sections by hierarchy, concatenating children under parents.

    Args:
        tsv_df: DataFrame with TSV section data.

    Returns:
        List of group dicts with 'main_section', 'child_sections',
        'content', 'hierarchy_levels'. Groups are in original TSV order.
    """
    sections = _build_section_tree(tsv_df)

    # Get main sections (hierarchy_level=1 or no parent)
    main_sections = [
        (num, s) for num, s in sections.items()
        if num != "_section_order" and
        (s["hierarchy_level"] == 1 or not s["parent_section"])
    ]

    # Sort by original TSV index to preserve document order
    main_sections.sort(key=lambda x: x[1]["_original_index"])

    groups = []
    for section_num, main_section in main_sections:
        # Get all descendants
        descendant_nums = _get_descendants(section_num, sections)

        # Build content
        content = _concatenate_section_group(
            main_section, descendant_nums, sections
        )

        # Determine hierarchy levels in this group
        levels = {main_section["hierarchy_level"]}
        for desc_num in descendant_nums:
            if desc_num in sections:
                levels.add(sections[desc_num]["hierarchy_level"])

        groups.append({
            "main_section_num": section_num,
            "main_section_heading": main_section["heading"],
            "child_sections": descendant_nums,
            "content": content,
            "hierarchy_levels": sorted(levels),
            "page": main_section["page"],
        })

    logger.info(
        "Grouped %d sections into %d hierarchical groups",
        len(sections), len(groups)
    )
    return groups


def split_by_token_limit(
    groups: list[dict],
    limit: int = 8000,
    tokenizer: TokenizerType = "approx",
) -> list[dict]:
    """Split groups that exceed token limit.

    Args:
        groups: List of hierarchical group dicts.
        limit: Maximum tokens per chunk.
        tokenizer: Tokenizer method.

    Returns:
        List of chunk dicts respecting token limit.
    """
    chunks = []

    for group in groups:
        content = group["content"]
        token_count = calculate_token_count(content, tokenizer)

        if token_count <= limit:
            # Keep as single chunk
            chunks.append({
                "main_section_num": group["main_section_num"],
                "main_section_heading": group["main_section_heading"],
                "content": content,
                "hierarchy_levels": group["hierarchy_levels"],
                "page": group["page"],
                "chunk_index": 0,
                "total_chunks": 1,
                "token_count": token_count,
            })
        else:
            # Split by child sections
            child_chunks = _split_group_by_children(
                group, limit, tokenizer
            )
            chunks.extend(child_chunks)

    total_tokens = sum(c["token_count"] for c in chunks)
    logger.info(
        "Split %d groups into %d chunks, total tokens: %d",
        len(groups), len(chunks), total_tokens
    )
    return chunks


def _split_group_by_children(
    group: dict,
    limit: int,
    tokenizer: TokenizerType,
) -> list[dict]:
    """Split a large group by its child sections.

    Args:
        group: The group dict to split.
        limit: Token limit per chunk.
        tokenizer: Tokenizer method.

    Returns:
        List of chunk dicts.
    """
    chunks = []
    current_content = []
    current_tokens = 0

    # Start with main section header
    header = (
        f"[{group['main_section_num']}] "
        f"{group['main_section_heading']}"
    )
    header_tokens = calculate_token_count(header, tokenizer)

    # Rebuild sections from content
    content_lines = group["content"].split("\n")
    current_section = []
    current_section_tokens = 0

    for line in content_lines:
        line_tokens = calculate_token_count(line, tokenizer)

        # Check if this is a new section marker
        if line.startswith("[") and "]" in line:
            # Flush current section if exists
            if current_section:
                section_content = "\n".join(current_section)
                section_tokens = calculate_token_count(
                    section_content, tokenizer
                )

                # Check if adding would exceed limit
                if current_tokens + section_tokens > limit and current_content:
                    # Save current chunk
                    chunks.append({
                        "main_section_num": group["main_section_num"],
                        "main_section_heading": group["main_section_heading"],
                        "content": "\n\n".join(current_content),
                        "hierarchy_levels": group["hierarchy_levels"],
                        "page": group["page"],
                        "chunk_index": len(chunks),
                        "total_chunks": 0,  # Updated later
                        "token_count": current_tokens,
                    })
                    current_content = []
                    current_tokens = header_tokens  # Keep header

                    # Add header to new chunk
                    if header not in current_content:
                        current_content.append(header)

                current_content.append(section_content)
                current_tokens += section_tokens

            current_section = [line]
            current_section_tokens = line_tokens
        else:
            current_section.append(line)
            current_section_tokens += line_tokens

    # Flush last section
    if current_section:
        section_content = "\n".join(current_section)
        section_tokens = calculate_token_count(section_content, tokenizer)

        if current_tokens + section_tokens > limit and current_content:
            chunks.append({
                "main_section_num": group["main_section_num"],
                "main_section_heading": group["main_section_heading"],
                "content": "\n\n".join(current_content),
                "hierarchy_levels": group["hierarchy_levels"],
                "page": group["page"],
                "chunk_index": len(chunks),
                "total_chunks": 0,
                "token_count": current_tokens,
            })
            current_content = [header]
            current_tokens = header_tokens

        current_content.append(section_content)
        current_tokens += section_tokens

    # Flush final chunk
    if current_content:
        chunks.append({
            "main_section_num": group["main_section_num"],
            "main_section_heading": group["main_section_heading"],
            "content": "\n\n".join(current_content),
            "hierarchy_levels": group["hierarchy_levels"],
            "page": group["page"],
            "chunk_index": len(chunks),
            "total_chunks": 0,
            "token_count": current_tokens,
        })

    # Update total_chunks
    for chunk in chunks:
        chunk["total_chunks"] = len(chunks)

    return chunks


def build_refine_chain(llm: BaseChatModel):
    """Create LangChain chain for refined gazette conversion.

    Args:
        llm: The language model to use.

    Returns:
        A LangChain chain for converting grouped sections.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", GAZETTE_REFINE_PROMPT_TEMPLATE),
        ("human", GAZETTE_REFINE_HUMAN_TEMPLATE),
    ])
    return prompt | llm | StrOutputParser()


def _load_checkpoint(path: Path) -> dict | None:
    """Load checkpoint from disk."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_checkpoint(
    path: Path,
    last_index: int,
    results: list,
    total: int,
):
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


def process_refined_chunks(
    chain,
    chunks: list[dict],
    checkpoint_path: str | Path | None = None,
    rate_config: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Process refined chunks with checkpointing.

    Args:
        chain: LangChain chain from build_refine_chain.
        chunks: List of chunk dicts from split_by_token_limit.
        checkpoint_path: Path for checkpoint file.
        rate_config: Rate limiting configuration.

    Returns:
        Tuple of (converted_chunks, errors).
    """
    cfg = {**DEFAULT_RATE_CONFIG, **(rate_config or {})}
    batch_size = cfg["batch_size"]
    delay = cfg["delay_between_requests"]

    results = []
    errors = []
    start_index = 0

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    if checkpoint_path:
        cp = _load_checkpoint(checkpoint_path)
        if cp:
            results = cp["completed_sections"]
            start_index = cp["last_completed_index"] + 1
            logger.info(
                "Resuming from checkpoint: chunk %d/%d",
                start_index + 1, len(chunks)
            )

    import time

    for i in range(start_index, len(chunks)):
        chunk = chunks[i]
        retry_count = 0
        success = False

        while retry_count < cfg["max_retries"] and not success:
            try:
                result = chain.invoke({
                    "main_section_num": chunk["main_section_num"],
                    "main_section_heading": chunk["main_section_heading"],
                    "hierarchy_levels": ", ".join(
                        map(str, chunk["hierarchy_levels"])
                    ),
                    "child_count": chunk.get("child_count", 0),
                    "content": chunk["content"],
                })

                results.append({
                    "main_section_num": chunk["main_section_num"],
                    "chunk_index": chunk["chunk_index"],
                    "markup": result,
                })
                success = True

                pct = ((i + 1) / len(chunks)) * 100
                logger.info(
                    "[%d/%d %.0f%%] Section %s (chunk %d/%d): %s",
                    i + 1, len(chunks), pct,
                    chunk["main_section_num"],
                    chunk["chunk_index"] + 1,
                    chunk["total_chunks"],
                    chunk["main_section_heading"][:50],
                )

                time.sleep(delay)

                if (i + 1) % batch_size == 0:
                    if checkpoint_path:
                        _save_checkpoint(
                            checkpoint_path, i, results, len(chunks)
                        )
                    logger.info(
                        "Batch %d done, cooling %ds",
                        (i + 1) // batch_size, cfg["batch_delay"]
                    )
                    time.sleep(cfg["batch_delay"])

            except Exception as exc:
                err_str = str(exc)
                is_retryable = any(
                    kw.lower() in err_str.lower()
                    for kw in RETRYABLE_KEYWORDS
                )
                if is_retryable and retry_count < cfg["max_retries"]:
                    retry_count += 1
                    wait = cfg["initial_backoff"] * (2 ** (retry_count - 1))
                    logger.warning(
                        "Retryable error on chunk %d (attempt %d/%d), "
                        "waiting %ds",
                        i, retry_count, cfg["max_retries"], wait
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Error on chunk %d: %s",
                        i, err_str[:120]
                    )
                    errors.append({
                        "chunk_index": i,
                        "section": chunk["main_section_num"],
                        "error": err_str
                    })
                    if checkpoint_path:
                        _save_checkpoint(
                            checkpoint_path, i, results, len(chunks)
                        )
                    break

        if not success:
            logger.error(
                "Skipping chunk %d after %d retries",
                i, retry_count
            )

    if checkpoint_path:
        _save_checkpoint(checkpoint_path, len(chunks) - 1, results, len(chunks))

    return results, errors


def assemble_refined_markup(results: list[dict], document_name: str) -> str:
    """Assemble final markup from processed chunks.

    Args:
        results: List of converted chunk results.
        document_name: Name of the document.

    Returns:
        Complete markup string.
    """
    markup_parts = []

    # Document preamble
    markup_parts.append(f"# {document_name}")
    markup_parts.append("")

    # Group results by main section
    sections = {}
    for result in results:
        section_num = result["main_section_num"]
        if section_num not in sections:
            sections[section_num] = []
        sections[section_num].append(result)

    # Sort and concatenate
    for section_num in sorted(sections.keys()):
        chunks = sections[section_num]
        # Sort by chunk_index if available
        chunks.sort(key=lambda x: x.get("chunk_index", 0))

        for chunk in chunks:
            markup_parts.append(chunk["markup"])

    return "\n\n".join(markup_parts)


def refine_gazette_grouping(
    tsv_path: str | Path,
    output_path: str | Path,
    token_limit: int = 8000,
    tokenizer: TokenizerType = "approx",
) -> dict:
    """Group and split Gazette sections hierarchically, output TSV.

    This function concatenates subsections under their parent main sections
    and splits content only when token limits are crossed. Outputs a TSV
    that can be processed by gazette-convert.

    Args:
        tsv_path: Path to the TSV file from gazette-extract.
        output_path: Path for the output TSV file.
        token_limit: Maximum tokens per chunk (default: 8000).
        tokenizer: Tokenizer method ('tiktoken' or 'approx').

    Returns:
        Dictionary with grouping results and metadata.
    """
    logger.info("══ Gazette Refine Grouping ══")
    logger.info("TSV: %s", tsv_path)
    logger.info("Output: %s", output_path)
    logger.info("Token limit: %d", token_limit)

    tsv_path = Path(tsv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load TSV
    logger.info("Loading TSV...")
    tsv_df = pd.read_csv(tsv_path, sep="\t")
    logger.info("Loaded %d sections", len(tsv_df))

    # Phase 1: Group sections hierarchically
    logger.info("══ Phase 1: Grouping Sections ══")
    groups = group_sections_hierarchically(tsv_df)

    # Phase 2: Split by token limit
    logger.info("══ Phase 2: Token-Aware Splitting ══")
    chunks = split_by_token_limit(groups, token_limit, tokenizer)

    # Calculate stats
    total_tokens = sum(c["token_count"] for c in chunks)
    logger.info(
        "Created %d chunks from %d groups, avg %.0f tokens each",
        len(chunks), len(groups), total_tokens / len(chunks) if chunks else 0
    )

    # Phase 3: Build output TSV
    logger.info("══ Phase 3: Building Output TSV ══")

    # Build rows preserving order
    rows = []
    for i, chunk in enumerate(chunks):
        section_num = chunk["main_section_num"]
        # Add chunk suffix if multiple chunks for same section
        if chunk["total_chunks"] > 1:
            section_num = f"{section_num}_chunk{chunk['chunk_index'] + 1}"

        rows.append({
            "page": chunk["page"],
            "section_num": section_num,
            "heading": chunk["main_section_heading"],
            "content": chunk["content"],
            "chapter": "",  # Preserved from original
            "start_pos": 0,
            "end_pos": 0,
            "language": "en",
            "confidence": 1.0,
            "hierarchy_level": 1,  # All chunks are top-level for conversion
            "parent_section": "",
            "original_chunks": chunk["total_chunks"],
            "chunk_index": chunk["chunk_index"],
        })

    # Create DataFrame and save
    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_path, sep="\t", index=False)
    logger.info("Wrote TSV to: %s", output_path)

    # Create metadata
    metadata = {
        "tsv_path": str(tsv_path),
        "output_path": str(output_path),
        "token_limit": token_limit,
        "tokenizer": tokenizer,
        "total_sections": len(tsv_df),
        "groups_created": len(groups),
        "chunks_created": len(chunks),
        "total_tokens": total_tokens,
    }

    logger.info(
        "Grouping complete: %d groups -> %d chunks",
        len(groups), len(chunks)
    )

    return {
        "metadata": metadata,
        "chunks": chunks,
        "output_df": output_df,
    }
