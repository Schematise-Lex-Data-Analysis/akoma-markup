"""Intelligent Gazette refinement with contextual analysis.

This module implements an intelligent algorithm for combining sections by:
1. Reading each row and surrounding 4-5 rows to decide if a new section begins
2. If no new section begins, keep adding to same heading/section number
3. Handle section numbers that may be actual section numbers, subsection numbers, or both
4. Account for content repetition across different sub-sections due to parsing errors
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

TokenizerType = Literal["tiktoken", "approx"]


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


def _is_section_number(text: str) -> bool:
    """Check if text looks like a section number."""
    if pd.isna(text) or not isinstance(text, str):
        return False
    
    text = text.strip()
    
    # Common section patterns
    patterns = [
        r'^\d+\.$',  # "1.", "2.", etc.
        r'^\(\d+\)',  # "(1)", "(2)", etc.
        r'^\d+\.\(\d+\)',  # "1.(1)", "2.(2)", etc.
        r'^[A-Z]\.',  # "A.", "B.", etc.
        r'^\([a-z]\)',  # "(a)", "(b)", etc.
        r'^\([ivx]+\)',  # "(i)", "(ii)", etc.
        r'^[A-Z]+ \d+$',  # "CHAPTER I", "SECTION 1"
        r'^[A-Z]+ [IVXLCDM]+$',  # "CHAPTER IV", "PART II"
        r'^Sec\.? \d+',  # "Sec. 1", "Sec 1"
    ]
    
    return any(re.match(pattern, text) for pattern in patterns)


def _is_main_section_number(text: str) -> bool:
    """Check if text looks like a main section number (not subsection)."""
    if pd.isna(text) or not isinstance(text, str):
        return False
    
    text = text.strip()
    
    # Main section patterns (not sub-sections)
    main_patterns = [
        r'^\d+\.$',  # "1.", "2.", etc. (but not "1.(1)")
        r'^[A-Z]+ \d+$',  # "CHAPTER I", "SECTION 1"
        r'^CHAPTER [IVXLCDM]+$',  # "CHAPTER I", "CHAPTER II"
        r'^Sec\.? \d+',  # "Sec. 1", "Sec 1"
    ]
    
    # Exclude sub-section patterns
    exclude_patterns = [
        r'^\(\d+\)',  # "(1)", "(2)" are sub-sections
        r'^\d+\.\(\d+\)',  # "1.(1)" is sub-section
        r'^\([a-z]\)',  # "(a)", "(b)" are clauses
        r'^\([ivx]+\)',  # "(i)", "(ii)" are sub-clauses
    ]
    
    is_main = any(re.match(pattern, text) for pattern in main_patterns)
    is_sub = any(re.match(pattern, text) for pattern in exclude_patterns)
    
    return is_main and not is_sub


def _extract_base_section_number(section_num: str) -> str:
    """Extract base section number from complex section numbers.
    
    Examples:
        "1." -> "1."
        "(3)" -> "" (subsection, no base)
        "1.(1)" -> "1."
        "Sec. 1" -> "Sec. 1"
        "_row_34" -> "_row_34"
        "(a)" -> "" (clause, no base)
        "(i)" -> "" (subclause, no base)
    """
    if pd.isna(section_num) or not isinstance(section_num, str):
        return ""
    
    section_num = section_num.strip()
    
    # Handle compound patterns like "1.(1)"
    if re.match(r'^\d+\.\(\d+\)', section_num):
        base = section_num.split('(')[0]  # Get "1." from "1.(1)"
        return base if base.endswith('.') else f"{base}."
    
    # Handle patterns like "(3)", "(a)", "(i)"
    if re.match(r'^\([^)]+\)', section_num):
        return ""  # Subsection/clause only, no base
    
    # Handle patterns like "Sec. 1"
    if re.match(r'^Sec\.? \d+', section_num):
        return section_num
    
    # Keep main sections as-is
    if _is_main_section_number(section_num):
        return section_num
    
    # For _row_ markers or other text, keep as-is
    return section_num


def _is_new_section_start(
    current_row: dict,
    prev_row: dict,
    next_rows: list[dict],
    context_window: int = 5
) -> bool:
    """Determine if current row starts a new logical section.
    
    Analyzes context (previous row and next few rows) to decide.
    
    Args:
        current_row: Current row being analyzed.
        prev_row: Previous row in TSV.
        next_rows: List of next N rows for context.
        context_window: How many rows to look ahead.
    
    Returns:
        True if this appears to start a new section.
    """
    curr_section = str(current_row.get('section_num', ''))
    prev_section = str(prev_row.get('section_num', '')) if prev_row else ''
    
    # Rule 1: Clear section number change to a main section
    if _is_main_section_number(curr_section):
        # If previous was not a main section, this is likely new
        if not _is_main_section_number(prev_section):
            return True
        
        # If both are main sections, check if they're different
        curr_base = _extract_base_section_number(curr_section)
        prev_base = _extract_base_section_number(prev_section)
        if curr_base != prev_base and curr_base and prev_base:
            return True
    
    # Rule 2: Transition from _row_ marker to numbered section
    if prev_section.startswith('_row_') and _is_section_number(curr_section):
        return True
    
    # Rule 3: Check heading content for section breaks
    curr_heading = str(current_row.get('heading', '')).lower()
    prev_heading = str(prev_row.get('heading', '')).lower() if prev_row else ''
    
    # Common section break indicators in headings
    section_break_indicators = [
        'chapter',
        'part',
        'schedule',
        'appendix',
        'section',
        'article',
        'rule',
        'regulation',
    ]
    
    for indicator in section_break_indicators:
        if indicator in curr_heading and indicator not in prev_heading:
            return True
    
    # Rule 4: Analyze next few rows for consistency
    # If next rows continue with same base section, probably not new
    curr_base = _extract_base_section_number(curr_section)
    if curr_base:
        # Check next N rows
        lookahead = min(context_window, len(next_rows))
        for i in range(lookahead):
            next_section = str(next_rows[i].get('section_num', ''))
            next_base = _extract_base_section_number(next_section)
            
            # If we see a different base section soon, this might be a transition
            if next_base and next_base != curr_base:
                # Check if the different base is actually a subsection
                if not _is_main_section_number(next_section):
                    # Probably a continuation with subsections
                    return False
                else:
                    # Different main section starting soon
                    return True
    
    # Rule 5: Check for content patterns
    curr_content = str(current_row.get('content', ''))
    prev_content = str(prev_row.get('content', '')) if prev_row else ''
    
    # Look for section markers in content
    content_section_markers = [
        r'^\[\d+\.\]',  # [1.]
        r'^\[\(\d+\)\]',  # [(1)]
        r'^\[CHAPTER',  # [CHAPTER
        r'^\[Section',  # [Section
    ]
    
    for marker in content_section_markers:
        if re.search(marker, curr_content) and not re.search(marker, prev_content):
            return True
    
    # Default: Not a new section if we can't determine otherwise
    return False


def _is_content_duplicate(
    content1: str,
    content2: str,
    threshold: float = 0.7
) -> bool:
    """Check if content appears to be a duplicate/overlap.
    
    Args:
        content1: First content string.
        content2: Second content string.
        threshold: Similarity threshold (0-1).
    
    Returns:
        True if content appears to be duplicate.
    """
    if not content1 or not content2:
        return False
    
    # Simple approach: check for significant overlap
    shorter = content1 if len(content1) < len(content2) else content2
    longer = content2 if len(content1) < len(content2) else content1
    
    # Check if shorter is mostly contained in longer
    if len(shorter) > 100:  # Only check for substantial content
        # Look for common substantial substring
        max_overlap = 0
        for i in range(0, len(longer) - 100, 50):
            chunk = longer[i:i+200]
            if chunk in shorter:
                overlap = len(chunk)
                if overlap > max_overlap:
                    max_overlap = overlap
        
        similarity = max_overlap / len(shorter) if len(shorter) > 0 else 0
        return similarity > threshold
    
    return False


def intelligent_group_sections(
    tsv_df: pd.DataFrame,
    token_limit: int = 8000,
    tokenizer: TokenizerType = "approx",
    context_window: int = 5
) -> list[dict]:
    """Intelligently group sections with contextual analysis.
    
    Args:
        tsv_df: DataFrame with TSV section data.
        token_limit: Maximum tokens per group.
        tokenizer: Tokenizer method.
        context_window: How many rows to look ahead for context.
    
    Returns:
        List of group dicts with section metadata and combined content.
    """
    rows = tsv_df.to_dict('records')
    groups = []
    current_group = None
    current_tokens = 0
    
    for i, row in enumerate(rows):
        # Get context rows
        prev_row = rows[i-1] if i > 0 else None
        next_rows = rows[i+1:min(i+1+context_window, len(rows))]
        
        # Determine if this starts a new section
        is_new_section = _is_new_section_start(row, prev_row, next_rows, context_window)
        
        # Calculate content tokens
        content = str(row.get('content', ''))
        content_tokens = calculate_token_count(content, tokenizer)
        
        # Check for content duplication with previous row
        is_duplicate = False
        if prev_row and current_group:
            prev_content = str(prev_row.get('content', ''))
            if _is_content_duplicate(content, prev_content):
                is_duplicate = True
                logger.debug(f"Detected duplicate content at row {i}")
        
        # Start new group if:
        # 1. It's a new section start, OR
        # 2. Adding would exceed token limit, OR
        # 3. No current group exists
        if (is_new_section and not is_duplicate) or \
           (current_group and current_tokens + content_tokens > token_limit) or \
           not current_group:
            
            # Save previous group if exists
            if current_group:
                groups.append(current_group)
            
            # Start new group
            section_num = str(row.get('section_num', ''))
            base_section = _extract_base_section_number(section_num)
            
            current_group = {
                'main_section_num': base_section if base_section else section_num,
                'original_section_num': section_num,
                'main_section_heading': str(row.get('heading', '')),
                'page': int(row.get('page', 0)) if pd.notna(row.get('page')) else 0,
                'content_parts': [content],
                'row_indices': [i],
                'section_numbers': [section_num],
                'token_count': content_tokens,
                'start_row': i,
            }
            current_tokens = content_tokens
        
        else:
            # Add to current group
            current_group['content_parts'].append(content)
            current_group['row_indices'].append(i)
            current_group['section_numbers'].append(str(row.get('section_num', '')))
            current_group['token_count'] += content_tokens
            current_tokens += content_tokens
    
    # Add final group
    if current_group:
        groups.append(current_group)
    
    # Process groups into final format
    processed_groups = []
    for group in groups:
        # Combine content parts
        combined_content = []
        for j, (content, section_num) in enumerate(zip(group['content_parts'], group['section_numbers'])):
            if j == 0 or _is_section_number(section_num):
                # Add section marker for first item or numbered sections
                combined_content.append(f"[{section_num}]")
            combined_content.append(content)
            combined_content.append("")  # Blank line between parts
        
        full_content = "\n".join(combined_content).strip()
        
        # Determine hierarchy levels present
        hierarchy_levels = set()
        for section_num in group['section_numbers']:
            if _is_main_section_number(section_num):
                hierarchy_levels.add(1)
            elif re.match(r'^\(\d+\)', section_num):
                hierarchy_levels.add(2)
            elif re.match(r'^\([a-z]\)', section_num):
                hierarchy_levels.add(3)
            elif re.match(r'^\([ivx]+\)', section_num):
                hierarchy_levels.add(4)
            else:
                hierarchy_levels.add(1)
        
        processed_groups.append({
            'main_section_num': group['main_section_num'],
            'main_section_heading': group['main_section_heading'],
            'content': full_content,
            'hierarchy_levels': sorted(hierarchy_levels),
            'page': group['page'],
            'row_indices': group['row_indices'],
            'token_count': group['token_count'],
            'num_rows': len(group['row_indices']),
        })
    
    logger.info(
        "Intelligently grouped %d rows into %d groups",
        len(rows), len(processed_groups)
    )
    
    # Log group statistics
    for i, group in enumerate(processed_groups[:10]):  # First 10 groups
        logger.debug(
            "Group %d: %s (rows %d-%d, %d tokens, %d rows)",
            i, group['main_section_num'],
            group['row_indices'][0], group['row_indices'][-1],
            group['token_count'], group['num_rows']
        )
    
    return processed_groups


def intelligent_refine_gazette(
    tsv_path: str | Path,
    output_path: str | Path,
    token_limit: int = 8000,
    tokenizer: TokenizerType = "approx",
    context_window: int = 5,
) -> dict:
    """Intelligently refine Gazette TSV with contextual analysis.
    
    Args:
        tsv_path: Path to the TSV file from gazette-extract.
        output_path: Path for the output TSV file.
        token_limit: Maximum tokens per chunk (default: 8000).
        tokenizer: Tokenizer method ('tiktoken' or 'approx').
        context_window: How many rows to look ahead for context.
    
    Returns:
        Dictionary with grouping results and metadata.
    """
    logger.info("══ Intelligent Gazette Refinement ══")
    logger.info("TSV: %s", tsv_path)
    logger.info("Output: %s", output_path)
    logger.info("Token limit: %d, Context window: %d", token_limit, context_window)
    
    tsv_path = Path(tsv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load TSV
    logger.info("Loading TSV...")
    tsv_df = pd.read_csv(tsv_path, sep="\t")
    logger.info("Loaded %d sections", len(tsv_df))
    
    # Intelligent grouping
    logger.info("══ Intelligent Grouping ══")
    groups = intelligent_group_sections(
        tsv_df, 
        token_limit=token_limit,
        tokenizer=tokenizer,
        context_window=context_window
    )
    
    # Split groups that exceed token limit (should be rare with intelligent grouping)
    logger.info("══ Final Token Check ══")
    chunks = []
    for group in groups:
        if group['token_count'] <= token_limit:
            chunks.append({
                'main_section_num': group['main_section_num'],
                'main_section_heading': group['main_section_heading'],
                'content': group['content'],
                'hierarchy_levels': group['hierarchy_levels'],
                'page': group['page'],
                'chunk_index': 0,
                'total_chunks': 1,
                'token_count': group['token_count'],
                'row_indices': group['row_indices'],
            })
        else:
            # Need to split - use simple line-based splitting
            logger.warning(
                "Group %s exceeds token limit (%d > %d), splitting",
                group['main_section_num'], group['token_count'], token_limit
            )
            content_lines = group['content'].split('\n')
            current_chunk = []
            current_tokens = 0
            
            for line in content_lines:
                line_tokens = calculate_token_count(line, tokenizer)
                
                if current_tokens + line_tokens > token_limit and current_chunk:
                    # Save current chunk
                    chunk_content = '\n'.join(current_chunk)
                    chunks.append({
                        'main_section_num': group['main_section_num'],
                        'main_section_heading': group['main_section_heading'],
                        'content': chunk_content,
                        'hierarchy_levels': group['hierarchy_levels'],
                        'page': group['page'],
                        'chunk_index': len(chunks),
                        'total_chunks': 0,  # Will update later
                        'token_count': current_tokens,
                        'row_indices': [],  # Can't preserve for split chunks
                    })
                    current_chunk = [line]
                    current_tokens = line_tokens
                else:
                    current_chunk.append(line)
                    current_tokens += line_tokens
            
            # Add final chunk
            if current_chunk:
                chunk_content = '\n'.join(current_chunk)
                chunks.append({
                    'main_section_num': group['main_section_num'],
                    'main_section_heading': group['main_section_heading'],
                    'content': chunk_content,
                    'hierarchy_levels': group['hierarchy_levels'],
                    'page': group['page'],
                    'chunk_index': len(chunks),
                    'total_chunks': 0,
                    'token_count': current_tokens,
                    'row_indices': [],
                })
    
    # Update total_chunks for split groups
    section_chunks = {}
    for chunk in chunks:
        section = chunk['main_section_num']
        if section not in section_chunks:
            section_chunks[section] = []
        section_chunks[section].append(chunk)
    
    for section, section_chunk_list in section_chunks.items():
        for i, chunk in enumerate(section_chunk_list):
            chunk['chunk_index'] = i
            chunk['total_chunks'] = len(section_chunk_list)
    
    # Calculate stats
    total_tokens = sum(c["token_count"] for c in chunks)
    logger.info(
        "Created %d chunks from %d groups, avg %.0f tokens each",
        len(chunks), len(groups), total_tokens / len(chunks) if chunks else 0
    )
    
    # Build output TSV
    logger.info("══ Building Output TSV ══")
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
        "context_window": context_window,
        "total_sections": len(tsv_df),
        "groups_created": len(groups),
        "chunks_created": len(chunks),
        "total_tokens": total_tokens,
        "method": "intelligent_contextual",
    }
    
    logger.info(
        "Intelligent refinement complete: %d groups -> %d chunks",
        len(groups), len(chunks)
    )
    
    return {
        "metadata": metadata,
        "chunks": chunks,
        "output_df": output_df,
    }