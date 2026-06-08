# akoma-markup

`akoma-markup` is a tool for converting law
PDFs into markup files following the
[Akoma Ntoso](https://www.oasis-open.org/standard/akn-v1-0/) format.

## Workflow
The following diagram explains the workflow from a high-level

![roadmap](image.png)

## Installation

Install in editable mode with the Azure extra:

```bash
pip install -e ".[azure]"
```

## Usage

### As a library

Endpoint, credential, and API style are picked up from the corresponding
`AZURE_INFERENCE_*` environment variables when omitted from the config:

```python
from akoma_markup import convert

result = convert(
    pdf_path="input.pdf",
    llm_config={
        "provider": "azure",
        "model": "<deployment-name>",
    },
    output_path="output/markup.txt",
    document_name="Bharatiya Nagarik Suraksha Sanhita 2023",
    act_number="46 of 2023",
    replaces="Criminal Procedure Code (CrPC) 1973",
)
print(result)  # path to the markup file
```

### As a CLI

```bash
# Read all credentials from a .env file (recommended)
akoma-markup input.pdf --llm-env .env -o output/markup.txt

# Or load the LLM config from a JSON file
akoma-markup input.pdf --llm-json llm_config.json
```

See [`.env.sample`](.env.sample) for the full set of supported environment
variables.

## LLM configuration

`llm_config` is a dict with two required fields — `provider` (always
`"azure"`) and `model` — plus Azure credentials. **There are no built-in
model defaults**: an unspecified `model` raises `ValueError` immediately.
Optional knobs: `temperature` (default `0`), `max_tokens` (default `4096`).

```python
{
    "provider": "azure",
    "endpoint": "https://<your-resource>.cognitiveservices.azure.com/openai/v1/",
    "credential": "<api-key>",
    "model": "<deployment-name>",   # must match a deployed model
    "api_style": "azure-inference", # one of: chat | responses | azure-inference
}
```

Environment-variable equivalents: `AZURE_INFERENCE_ENDPOINT`,
`AZURE_INFERENCE_KEY`, `AZURE_INFERENCE_MODEL_ID`, `AZURE_INFERENCE_API_STYLE`.

## Table rescue (optional)

PDFs with embedded tables come out garbled from `pdfplumber` because column
structure is lost when the page is flattened to text. `convert` can
optionally re-extract table-bearing pages through a multimodal vision LLM
(which renders the page as markdown with tables in pipe format), then
convert those tables deterministically into bluebell `TABLE` blocks and
splice the result back into the per-page text stream at sentinel positions
— so table data never passes through the section-conversion LLM.

Three modes — all use the same vision LLM:

- `declared` — you list the table pages yourself. Cheapest; use when you
  already know which pages contain tables.
- `auto` — the vision LLM scans every page image and flags table pages
  (cheap YES/NO call); only flagged pages are re-extracted (heavier
  per-page call). Best cost / coverage tradeoff.
- `full` — every page is re-extracted. Most expensive but guaranteed not to
  miss anything.

All three modes require `AZURE_VISION_KEY`, `AZURE_VISION_ENDPOINT`,
`AZURE_VISION_MODEL`, and `AZURE_VISION_API_STYLE`.

### As a library

```python
result = convert(
    pdf_path="banking_act.pdf",
    llm_config={"provider": "azure", "model": "<deployment-name>"},
    table_mode="declared",                       # or "auto", "full"
    table_pages=[12, 18, 19, 25],                # required when "declared"
    azure_vision_key="<azure-key>",              # or via AZURE_VISION_KEY
    azure_vision_endpoint="https://...",         # or via AZURE_VISION_ENDPOINT
    azure_vision_model="<vision-deployment>",    # or via AZURE_VISION_MODEL
    azure_vision_api_style="chat",               # or via AZURE_VISION_API_STYLE
)
```

### As a CLI

```bash
akoma-markup banking_act.pdf \
  --llm-env .env \
  --table-mode declared \
  --table-pages "12,18-19,25"
```

`--table-pages` accepts comma + range syntax. With `--llm-env`, the vision
credentials are read from the `.env` file; explicit flags
(`--azure-vision-key`, `--azure-vision-endpoint`, `--azure-vision-model`)
override the file values when needed.

## Output

Running `convert` writes:

- `<output_path>` — the Akoma Ntoso plaintext markup
- `<output_path>.meta.json` — conversion summary (sections, chapters, errors,
  document metadata)
- `<output_dir>/.akoma_cache/` — per-PDF resume state:
  `<pdf_stem>_conversion_checkpoint.json` (so a failed run skips
  already-converted sections) and, when `--table-mode` is set,
  `<pdf_stem>_table_ocr.json` (so re-runs skip already-extracted pages).
- `<output_dir>/.akoma_debug/` — inspection-only artefacts, prefixed with
  the PDF stem so multiple PDFs in the same output directory don't
  collide: `<pdf_stem>_raw_text.txt`, `<pdf_stem>_ocr.txt`,
  `<pdf_stem>_parser_summary.json`, `<pdf_stem>_sections.tsv`, and —
  when `--table-mode` is set — `<pdf_stem>_table_regions.txt`. Useful for
  diagnosing parser misalignment; safe to delete.

### Metadata file structure

```json
{
  "conversion_date": "2024-01-15T10:30:00",
  "sections_converted": 531,
  "chapters": 20,
  "errors": 0,
  "document": "Bharatiya Nagarik Suraksha Sanhita 2023",
  "act_number": "46 of 2023",
  "replaces": "Criminal Procedure Code (CrPC) 1973"
}
```

`document`, `act_number`, and `replaces` are included only when provided.

## Project layout

```
src/akoma_markup/
  __init__.py            # public convert() entry point
  cli.py                 # click-based CLI
  conversion.py          # section → markup chain + checkpointed runner
  output.py              # markup / metadata / OCR-text writers
  parsing/
    text/
      chapter_section_mapping.py   # TOC + section-body parsing
    tables/
      rescue.py          # table-rescue orchestrator (declared/auto/full)
      render.py          # markdown → bluebell TABLE
  util/
    pdf/
      text.py            # pdfplumber text extraction
      images.py          # pypdfium2 rasterization + data URLs
    llm/
      factory.py         # provider config → LangChain chat model
      vision.py          # vision-LLM client (page classification + extraction)
      azure_api.py       # ApiMode literal + validator
```

## Troubleshooting

- **`ValueError: ... must include a 'model' field`** — there are no model
  defaults. Set `AZURE_INFERENCE_MODEL_ID` (for the section-conversion LLM)
  or `AZURE_VISION_MODEL` (for table rescue) in your `.env`, or pass
  `model` / the matching `azure_vision_model` arg explicitly.
- **`Error code: 404 - Resource not found` on every section (Azure):** the
  `model` value does not match an active deployment. Check Azure AI Foundry /
  AI Studio and use the exact deployment name.
- **Empty output file with "N sections failed conversion":** every LLM call
  failed. Check credentials, model name, and network access, then rerun —
  the checkpoint cache will skip sections that did succeed.

## License

Licensed under the
[GNU Affero General Public License v3.0 or later](LICENSE)
(AGPL-3.0-or-later). If you run a modified version of this software as a
network service, you must make the modified source code available to its
users.
