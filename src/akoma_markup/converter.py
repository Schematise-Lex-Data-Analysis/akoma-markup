"""LLM-based conversion of legislative sections to Akoma Ntoso markup."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

CONVERSION_PROMPT_TEMPLATE = """You are converting {document_name} sections to Laws.Africa plaintext markup format.

MARKUP RULES:
1. Section format: SEC [num]. - [heading]
2. Indent content under sections with 2 spaces
3. Subsections: SUBSEC (1), SUBSEC (2), etc. - indented under SEC
4. Paragraphs within subsections: PARA (a), PARA (b), etc.
5. Subparagraphs: SUBPARA (i), SUBPARA (ii), etc.
6. Explanations: Start with "Explanation.—" as a separate indented paragraph
7. Illustrations: Start with "Illustration" or "(a)", "(b)" as separate indented paragraphs
8. Exceptions: Start with "Exception" as a separate indented paragraph
9. Provisos: Start with "Provided that" as a separate indented paragraph

EXAMPLE OUTPUT:
```
SEC 41. - When police may arrest without warrant
  SUBSEC (1)
    Any police officer may without an order from a Magistrate and without a warrant, arrest any person—

    PARA (a)
      who commits, in the presence of a police officer, a cognizable offence;

    PARA (b)
      against whom a reasonable complaint has been made...

  SUBSEC (2)
    Any officer in charge of a police station may...

  Explanation.—For the purposes of this section...
```

PRESERVE the exact legal text. Do not paraphrase or summarize."""

HUMAN_PROMPT_TEMPLATE = """Convert this section to Laws.Africa markup:

Section Number: {section_num}
Section Heading: {section_heading}
Section Content:
{section_content}

Output ONLY the markup, no explanations:"""

# Defaults matching the Azure AI S0 tier rate limiting.
DEFAULT_RATE_CONFIG = {
    "delay_between_requests": 5,
    "batch_size": 3,
    "batch_delay": 30,
    "max_retries": 3,
    "initial_backoff": 60,
}

RETRYABLE_KEYWORDS = [
    "429",
    "rate limit",
    "ReadTimeout",
    "read timed out",
    "ConnectionError",
    "Connection reset",
]


def build_chain(llm: BaseChatModel, document_name: str = "Legislative Document"):
    """Create a LangChain chain from a chat model and the conversion prompt.

    Args:
        llm: The language model to use for conversion.
        document_name: Name of the document being converted.

    Returns:
        A LangChain chain for converting sections to markup.
    """
    system_prompt = CONVERSION_PROMPT_TEMPLATE.format(document_name=document_name)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", HUMAN_PROMPT_TEMPLATE),
        ]
    )
    return prompt | llm | StrOutputParser()


def _load_checkpoint(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_checkpoint(path: Path, last_index: int, results: list, total: int):
    data = {
        "last_completed_index": last_index,
        "completed_sections": results,
        "timestamp": datetime.now().isoformat(),
        "total_sections": total,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def process_all_sections(
    chain,
    sections: list[dict],
    checkpoint_path: str | Path | None = None,
    rate_config: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Process all sections through the LLM chain with rate limiting and checkpointing.

    Args:
        chain: LangChain chain returned by ``build_chain``.
        sections: List of section dicts with 'num', 'heading', 'content'.
        checkpoint_path: Full path to checkpoint file. ``None`` disables checkpointing.
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
            print(
                f"Resuming from checkpoint: section {start_index + 1}/{len(sections)}",
                file=sys.stderr,
            )

    section_times: list[float] = []

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
                    }
                )
                results.append({"num": section["num"], "markup": result})
                success = True

                elapsed = time.time()
                section_times.append(elapsed)
                remaining = len(sections) - (i + 1)
                pct = ((i + 1) / len(sections)) * 100
                print(
                    f"[{i + 1}/{len(sections)} {pct:.0f}%] Section {section['num']}: "
                    f"{section['heading'][:60]}",
                    file=sys.stderr,
                )

                time.sleep(delay)

                if (i + 1) % batch_size == 0:
                    if checkpoint_path:
                        _save_checkpoint(checkpoint_path, i, results, len(sections))
                    print(
                        f"Batch {(i + 1) // batch_size} done, cooling {cfg['batch_delay']}s ...",
                        file=sys.stderr,
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
                    print(
                        f"Retryable error on section {section['num']} "
                        f"(attempt {retry_count}/{cfg['max_retries']}), "
                        f"waiting {wait}s ...",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                else:
                    print(
                        f"Error on section {section['num']}: {err_str[:120]}",
                        file=sys.stderr,
                    )
                    errors.append({"num": section["num"], "error": err_str})
                    if checkpoint_path:
                        _save_checkpoint(checkpoint_path, i, results, len(sections))
                    break

        if not success:
            print(
                f"Skipping section {section['num']} after {retry_count} retries",
                file=sys.stderr,
            )

    if checkpoint_path:
        _save_checkpoint(checkpoint_path, len(sections) - 1, results, len(sections))

    return results, errors
