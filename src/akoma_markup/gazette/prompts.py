"""AI prompts for gazette section extraction (Phase 1).

Conversion prompts are now in conversion.py (Phase 3).
"""

GazetteSection = dict

GAZETTE_SECTION_ANALYSIS_PROMPT = """Analyze this page from an Indian Gazette notification.

TASK: Extract the complete structured content including all headings, sections, and text.

IDENTIFY:
1. Main headings (e.g., "1. Short title", "2. Definitions")
2. Sub-sections (e.g., "(1)", "(2)", "(a)", "(b)", "(i)", "(ii)")
3. Hierarchical relationships (parent/child sections)
4. Tables and structured data
5. Page headers/footers
6. Non-English content (Hindi, Tamil, etc.)

HEADING PATTERNS TO DETECT:
- Numbered sections: "1.", "2.A", "12."
- Lettered clauses: "(a)", "(b)", "(c)"
- Roman numerals: "(i)", "(ii)", "(iii)"
- Schedules: "SCHEDULE", "THE FIRST SCHEDULE", etc.

LANGUAGE DETECTION:
- "en" = Pure English text
- "mixed" = Mostly English with some non-English
- "other" = Predominantly non-English (Hindi, Tamil, etc.)

Return ONLY a JSON object in this exact format:
{
    "page_number": <number>,
    "sections": [
        {
            "number": "1.",
            "heading": "Short title and commencement",
            "content": "These rules may be called... (FULL TEXT)",
            "language": "en",
            "confidence": 0.95,
            "hierarchy_level": 1,
            "parent_section": null,
            "bbox": {"left": 100, "top": 200, "right": 500, "bottom": 400}
        }
    ],
    "hierarchies": [
        {"parent": "2.", "children": ["(1)", "(2)"]}
    ],
    "tables_detected": true,
    "non_english_regions": [
        {"type": "hindi", "bbox": {...}, "purpose": "description"}
    ],
    "headers": ["THE GAZETTE OF INDIA", "EXTRAORDINARY", ...],
    "footers": ["Page 1", ...],
    "document_info": {
        "gazette_date": "2021-02-25",
        "gazette_number": "CG-DL-E-25022021-228415",
        "ministry": "Ministry of Electronics and Information Technology"
    }
}

IMPORTANT:
- Include COMPLETE text content for each section (not summaries)
- Preserve exact numbering and punctuation
- Mark hierarchy_level: 1=main rule, 2=subsection, 3=clause, 4=subclause
- Set parent_section to the parent section number or null
- Include bounding boxes if visible
- Return ONLY the JSON, no other text"""


GAZETTE_BATCH_ANALYSIS_PROMPT = """Analyze these {num_pages} consecutive pages from an Indian Gazette notification.

TASK: Extract the complete structured content across all pages.

For each page, identify:
1. Section/rule headings with numbers
2. Sub-sections and clauses
3. Hierarchical relationships
4. Content that continues from previous pages
5. Tables and structured data
6. Non-English content regions

Return a JSON array with one object per page:
[
    {
        "page_number": <number>,
        "sections": [...],
        "continues_from_previous": true/false,
        "continues_to_next": true/false,
        "tables_detected": true/false,
        "non_english_regions": [...]
    }
]

Each section must include:
- number: section identifier (e.g., "1.", "(a)")
- heading: section heading text
- content: complete text content
- language: "en", "mixed", or "other"
- confidence: 0.0-1.0
- hierarchy_level: 1-4
- parent_section: parent number or null"""


GAZETTE_CONTINUATION_PROMPT = """This is page {page_num} of a Gazette notification. The previous page ended with:

{previous_context}

TASK: Identify if the current page continues this content or starts new sections.

Return JSON:
{
    "page_number": {page_num},
    "continues_from_previous": true/false,
    "continuation_section": "section number if continuing",
    "sections": [...],
    "completed_sections": ["sections that end on this page"]
}

Include all sections visible on this page."""


GAZETTE_LANGUAGE_FILTER_PROMPT = """Analyze this text from a Gazette notification and determine if it is primarily English or contains significant non-English content (Hindi, Tamil, Telugu, etc.).

TEXT TO ANALYZE:
{text}

Return ONLY a JSON object:
{
    "language": "en" | "mixed" | "other",
    "confidence": 0.0-1.0,
    "english_percentage": 0-100,
    "non_english_regions": [
        {"text": "...", "language": "hindi/tamil/etc", "purpose": "description"}
    ],
    "should_keep": true/false
}

Set "should_keep" to true if the content is at least 50% English AND contains legislative significance."""


GAZETTE_QUALITY_PROMPT = """Review the extracted sections from a Gazette page and assess quality.

EXTRACTED SECTIONS:
{extracted_data}

ORIGINAL PAGE REFERENCE:
Page {page_num}

Assess:
1. Completeness - are all sections captured?
2. Accuracy - is numbering correct?
3. Hierarchy - are parent-child relationships correct?
4. Formatting - is text legible and complete?

Return JSON:
{
    "overall_quality": 0.0-1.0,
    "issues": [
        {
            "type": "missing_section" | "wrong_numbering" | "broken_hierarchy" | "incomplete_text",
            "description": "...",
            "severity": "high" | "medium" | "low",
            "affected_section": "..."
        }
    ],
    "recommendations": ["..."],
    "needs_reprocessing": true/false
}"""


GAZETTE_TABLE_DETECTION_PROMPT = """Analyze this page and identify all tables, schedules, and structured data layouts.

TASK:
1. Locate all tables (ruled grids, schedules, forms)
2. Determine if tables span multiple pages
3. Extract table structure (headers, rows, columns)
4. Identify table type (schedule, form, appendix, etc.)

Return JSON:
{
    "page_number": <number>,
    "tables": [
        {
            "id": <number>,
            "type": "schedule" | "form" | "table" | "appendix",
            "title": "...",
            "bbox": {"left": ..., "top": ..., ...},
            "headers": ["col1", "col2", ...],
            "row_count": <number>,
            "spans_from_previous": true/false,
            "spans_to_next": true/false,
            "estimated_content": "brief description"
        }
    ],
    "total_tables": <number>
}"""
