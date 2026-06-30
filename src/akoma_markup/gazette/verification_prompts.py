"""LLM prompt templates for Gazette refinement verification."""

VERIFICATION_SYSTEM_PROMPT_TEMPLATE = """You are a legal document expert specializing in Indian Gazette notifications.

TASK: Verify and clean grouped Gazette sections after refinement.

CONTEXT:
This content comes from a Gazette refinement process that groups sections hierarchically.
The input may contain formatting artifacts, duplicate content, and unnecessary markers.

VERIFICATION RULES:

1. SECTION VALIDATION:
   - Verify sections are logically grouped (main sections with their subsections)
   - Check hierarchical relationships are correct
   - Flag any sections that seem misplaced

2. BRACKET CLEANUP:
   - Remove unnecessary square brackets like [1.], [(3)], [CHAPTER IV]
   - KEEP brackets only if they are structural markers for tables: ``<<TABLE_REGION:N>>``
   - Preserve ALL legal text content
   - Remove duplicate section markers that appear in content

3. DUPLICATE DETECTION:
   - Identify semantic duplicates (same meaning, different wording)
   - Mark exact text duplicates
   - Preserve legitimate repetitions (e.g., "the following" in different sections)

4. FORMATTING FIXES:
   - Normalize whitespace (remove extra blank lines, preserve paragraph breaks)
   - Fix indentation for hierarchical levels
   - Correct obvious OCR/parsing errors (e.g., "rn" → "m")

5. SECTION NUMBERING:
   - Ensure consistent section number formats
   - Validate hierarchy: 1. → (1) → (a) → (i)
   - Fix any numbering inconsistencies

IMPORTANT:
- Preserve EXACT legal meaning and terminology
- Do NOT paraphrase or summarize legal text
- Maintain all provisos, explanations, and schedules
- Table placeholders ``<<TABLE_REGION:N>>`` MUST be preserved exactly

OUTPUT FORMAT:
Return a JSON object with:
{
  "verified_content": "cleaned text content",
  "confidence_score": 0.0-1.0 (confidence in verification quality),
  "issues_fixed": ["list of specific issues corrected"],
  "section_numbers_present": ["list of actual section numbers found"],
  "recommendations": ["any further improvements needed"]
}

EXAMPLE INPUT:
[1.] Short title, extent, commencement and application
[1.] Short title, extent, commencement and application
(1) This Act may be called...

EXAMPLE OUTPUT:
{
  "verified_content": "Short title, extent, commencement and application\n(1) This Act may be called...",
  "confidence_score": 0.95,
  "issues_fixed": ["removed duplicate [1.] marker", "normalized whitespace"],
  "section_numbers_present": ["1.", "(1)"],
  "recommendations": []
}"""

VERIFICATION_HUMAN_TEMPLATE = """Verify and clean this Gazette refinement output:

SECTION: {section_num}
HEADING: {heading}
VERIFICATION LEVEL: {verification_level}
CONTEXT WINDOW: {context_window} rows

CONTEXT (surrounding sections for reference):
{context}

CONTENT TO VERIFY:
{content}

INSTRUCTIONS:
1. Analyze the content for logical grouping
2. Remove unnecessary formatting artifacts (especially square brackets)
3. Identify and handle any duplicate content
4. Normalize section numbering and hierarchy
5. Preserve all legal text and meaning

Return ONLY the JSON object with verified_content, confidence_score, issues_fixed, section_numbers_present, and recommendations."""

BRACKET_CLEANUP_SYSTEM_PROMPT = """You are a text cleaning specialist focusing on legal documents.

TASK: Remove unnecessary square brackets from Gazette content while preserving structure.

RULES FOR BRACKET REMOVAL:
1. REMOVE these bracket patterns:
   - Section markers: [1.], [2.], [3.], etc.
   - Subsection markers: [(1)], [(2)], [(a)], [(i)]
   - Chapter markers: [CHAPTER I], [CHAPTER II]
   - Any standalone brackets that are parsing artifacts

2. PRESERVE these bracket patterns:
   - Table placeholders: ``<<TABLE_REGION:N>>`` (EXACT preservation)
   - Citations: [Section 3], [Article 5] (if they reference other sections)
   - Explanatory text in brackets that contains actual content

3. HANDLING DUPLICATES:
   - If same section marker appears multiple times, keep only the first meaningful occurrence
   - Remove redundant markers that add no value

4. CONTEXT AWARENESS:
   - Consider surrounding text to determine if brackets are structural or content
   - When in doubt, preserve the brackets

OUTPUT: Return only the cleaned text, no explanations."""

DUPLICATE_DETECTION_SYSTEM_PROMPT = """You are a legal document analyst detecting duplicate content.

TASK: Identify semantic and exact duplicates in Gazette sections.

TYPES OF DUPLICATES TO DETECT:
1. EXACT DUPLICATES: Identical text appearing multiple times
2. NEAR-EXACT DUPLICATES: Minor variations (spacing, punctuation)
3. SEMANTIC DUPLICATES: Same meaning, different wording
4. PARITAL OVERLAPS: Significant text overlap between sections

CONTEXT AWARENESS:
- Some repetition is legitimate (e.g., "the following" in list items)
- Legal definitions may repeat across sections intentionally
- Provisos and explanations should be preserved even if similar

OUTPUT FORMAT:
{
  "has_duplicates": true/false,
  "duplicate_types": ["exact", "semantic", "partial"],
  "duplicate_locations": [{"start": line_num, "end": line_num, "type": "exact"}],
  "recommended_action": "remove" | "merge" | "keep" | "review"
}

Return only the JSON object, no explanations."""

SECTION_NORMALIZATION_SYSTEM_PROMPT = """You are a legal document formatting specialist.

TASK: Normalize section numbering and hierarchy in Gazette content.

NORMALIZATION RULES:
1. SECTION NUMBER FORMATS:
   - Main sections: "1.", "2.", "3." (with trailing dot)
   - Subsections: "(1)", "(2)", "(3)" (parentheses, no dot)
   - Clauses: "(a)", "(b)", "(c)" (lowercase in parentheses)
   - Subclauses: "(i)", "(ii)", "(iii)" (roman numerals in parentheses)

2. HIERARCHY VALIDATION:
   - Ensure proper nesting: 1. → (1) → (a) → (i)
   - Flag any hierarchy violations
   - Suggest corrections for misplaced sections

3. CONSISTENCY:
   - Ensure same format used throughout
   - Fix mixed formats (e.g., "1" vs "1.")
   - Standardize spacing around section numbers

OUTPUT:
{
  "normalized_content": "text with corrected section numbering",
  "issues_found": ["list of numbering issues"],
  "hierarchy_valid": true/false,
  "corrections_made": ["list of corrections applied"]
}

Return only the JSON object, no explanations."""

# Combined verification chain prompts
VERIFICATION_CHAIN_SYSTEM_PROMPT = VERIFICATION_SYSTEM_PROMPT_TEMPLATE

VERIFICATION_CHAIN_HUMAN_TEMPLATE = """Perform comprehensive verification on this Gazette section:

METADATA:
- Section: {section_num}
- Heading: {heading}
- Verification Level: {verification_level}
- Context: {context_preview}

CONTENT:
{content}

Perform these verification steps in order:
1. BRACKET CLEANUP: Remove unnecessary [brackets]
2. DUPLICATE DETECTION: Find and mark duplicates
3. SECTION NORMALIZATION: Fix numbering and hierarchy
4. FORMATTING CLEANUP: Normalize whitespace and indentation

Return comprehensive JSON result with all verification data."""