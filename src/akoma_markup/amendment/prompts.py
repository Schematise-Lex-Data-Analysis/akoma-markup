"""Gazette-specific prompts for LLM conversion."""

GAZETTE_CONVERSION_PROMPT_TEMPLATE = """You are converting a Gazette of India notification to Laws.Africa plaintext markup format.

CONTEXT:
This is an official Gazette notification from the Government of India, typically containing:
- Rules, regulations, or amendments
- Ministerial notifications under specific Acts
- G.S.R. (Gazette Statutory Rules) numbers
- Complete text of new or amended rules

MARKUP RULES:
1. Rule format: RULE [num]. - [title/heading]
2. Indent content under rules with 2 spaces
3. Subrules: SUBRULE (1), SUBRULE (2), etc. - indented under RULE
4. Clauses within subrules: CLAUSE (a), CLAUSE (b), etc.
5. Subclauses: SUBCLAUSE (i), SUBCLAUSE (ii), etc.
6. Provisos: Start with "Provided that" as a separate indented paragraph
7. Explanations: Start with "Explanation.—" as a separate indented paragraph
8. Schedules: Begin with "SCHEDULE" heading, format as TABLE blocks when applicable
9. Table placeholders: tokens of the form ``<<TABLE_REGION:N>>`` are STRUCTURAL PLACEHOLDERS for tables.

    Rules for these tokens:
    - COPY each placeholder verbatim into the output, on its own line
    - Do NOT paraphrase, expand, remove, describe, comment on, or attempt to convert these tokens
    - If a placeholder token is the only content, output just that line at the right indent

EXAMPLE OUTPUT:
```
RULE 3. - Definitions
  In these rules, unless the context otherwise requires,—

  SUBRULE (1)
    CLAUSE (a)
      "Act" means the Information Technology Act, 2000 (21 of 2000);

    CLAUSE (b)
      "Appellate Body" means...

  SUBRULE (2)
    Words and expressions used herein and not defined...
```

Preserve the exact legal text. Do not paraphrase or summarize."""

GAZETTE_HUMAN_PROMPT_TEMPLATE = """Convert this gazette notification to Laws.Africa markup:

Gazette Text:
{gazette_text}

Output ONLY the markup, no explanations:"""

GAZETTE_PAGE_CONVERSION_PROMPT = """You are converting a page from a Gazette of India notification to Laws.Africa plaintext markup format.

CONTEXT:
This is page {page_num} of {total_pages} from an official Gazette notification from the Government of India. You will see this page as an image. Extract all text and convert to markup format.

The gazette typically contains:
- Rules, regulations, or amendments
- Ministerial notifications under specific Acts
- G.S.R. (Gazette Statutory Rules) numbers
- Complete text of new or amended rules

MARKUP RULES:
1. Rule format: RULE [num]. - [title/heading]
2. Indent content under rules with 2 spaces
3. Subrules: SUBRULE (1), SUBRULE (2), etc. - indented under RULE
4. Clauses within subrules: CLAUSE (a), CLAUSE (b), etc.
5. Subclauses: SUBCLAUSE (i), SUBCLAUSE (ii), etc.
6. Provisos: Start with "Provided that" as a separate indented paragraph
7. Explanations: Start with "Explanation.—" as a separate indented paragraph
8. Schedules: Begin with "SCHEDULE" heading, format as TABLE blocks when applicable
9. For tables: Convert to markdown table format with | delimiters, or use <<TABLE_REGION:N>> placeholder if table is complex
10. Preserve header/footer information like page numbers, gazette numbers, dates

PREVIOUS CONTEXT:
{previous_context}

IMPORTANT: Convert ALL text visible in the image, including:
- Main content (rules, subrules, clauses)
- Header information (Gazette numbers, dates, ministry names)
- Footer information (page numbers, publication details)
- Tables (as markdown tables or placeholders)

EXAMPLE OUTPUT:
```
RULE 3. - Definitions
  In these rules, unless the context otherwise requires,—

  SUBRULE (1)
    CLAUSE (a)
      "Act" means the Information Technology Act, 2000 (21 of 2000);

    CLAUSE (b)
      "Appellate Body" means...

  SUBRULE (2)
    Words and expressions used herein and not defined...
```

Preserve the exact legal text. Do not paraphrase or summarize.

Output ONLY the markup for this page, no explanations:"""

GAZETTE_MERGE_PROMPT_TEMPLATE = """You are merging markup from multiple pages of a Gazette notification into a single coherent document.

CONTEXT:
You will receive markup from {num_pages} pages of a Gazette of India notification. Your task is to merge them into a single, coherent markup document without duplication.

MERGE RULES:
1. Remove duplicate content that appears at page boundaries
2. Maintain proper indentation (2 spaces per level)
3. Ensure RULE, SUBRULE, CLAUSE hierarchy is preserved
4. Combine partial rules that span multiple pages
5. Remove page headers/footers (but keep the information if it's the first occurrence)
6. Ensure table placeholders like <<TABLE_REGION:N>> are preserved
7. Remove redundant Gazette headers that repeat across pages

PAGE MARKUP:
{page_markups}

Output ONLY the merged markup, no explanations:"""
