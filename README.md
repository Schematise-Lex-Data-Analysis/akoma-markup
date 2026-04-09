# akoma-markup

Convert the law PDFs(source - https://www.indiacode.nic.in/) into
[Akoma Ntoso](https://www.akomantoso.org/) / [Laws.Africa](https://laws.africa/)
plaintext markup using any LangChain chat model.

The package extracts text from the law PDF, parses its table of contents and
sections, then uses an LLM to rewrite each section into Akoma Ntoso markup.
Output is a single `.txt` file plus a `.meta.json` sidecar with conversion
metadata.

## Installation

Install the package in editable mode along with the extra for the LLM provider
you want to use:

```bash
# Anthropic
pip install -e ".[anthropic]"

# Azure AI Inference (Llama, etc.)
pip install -e ".[azure]"
```

## Usage

### As a library

```python
from akoma_markup import convert

result = convert(
    pdf_path="input_pdf.pdf",
    llm_config={
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        # api_key picked up from ANTHROPIC_API_KEY if omitted
    },
    output_path="output/bnss_markup.txt",
)
print(result)  # path to the markup file
```

### As a CLI

```bash
# Inline config
akoma-markup input_pdf.pdf \
  --llm-inline '{"provider": "anthropic", "model": "claude-sonnet-4-20250514"}' \
  -o output/bnss_markup.txt

# From a JSON file
akoma-markup input_pdf.pdf --llm-json llm_config.json

# From a .env file
akoma-markup input_pdf.pdf --llm-env .env
```

## LLM configuration

`llm_config` is a dict with a required `provider` field plus provider-specific
credentials. Common fields: `model`, `temperature` (default `0`), `max_tokens`
(default `4096`).

### Anthropic

```python
{
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514",
    "api_key": "sk-ant-...",  # or set ANTHROPIC_API_KEY
}
```

### Azure AI Inference

```python
{
    "provider": "azure",
    "endpoint": "https://<your-resource>.cognitiveservices.azure.com/openai/v1/",
    "credential": "<api-key>",  # or set AZURE_INFERENCE_CREDENTIAL
    "model": "Llama-3.3-70B-Instruct",  # must match a deployed model
}
```

Environment variable fallbacks:
`AZURE_INFERENCE_ENDPOINT`, `AZURE_INFERENCE_CREDENTIAL`.

When using `--llm-env`, the CLI reads `PROVIDER`, `AZURE_INFERENCE_ENDPOINT`,
`AZURE_INFERENCE_CREDENTIAL`, `AZURE_MODEL_ID`, `ANTHROPIC_API_KEY`, and
`ANTHROPIC_MODEL_ID` from the file.

## Output

Running `convert` produces:

- `<output>.txt` — the Akoma Ntoso plaintext markup
- `<output>.meta.json` — per-section conversion status, errors, and chapter info
- `<output-dir>/.akoma_checkpoints/` — intermediate per-section cache that lets
  a failed run resume without re-calling the LLM for already-converted sections

## Project layout

```
src/akoma_markup/
  __init__.py     # convert() entry point
  extractor.py    # pdfplumber text extraction
  parser.py       # TOC + section parsing
  converter.py    # LangChain chain + per-section processing
  llm.py          # provider config → chat model
  writer.py       # markup + metadata output
  cli.py          # click-based CLI
```

## Troubleshooting

- **`Error code: 404 - Resource not found` on every section (Azure):** the
  `model` value does not match an active deployment on your endpoint. Check
  your Azure AI Foundry / AI Studio deployments and use the exact deployment
  name.
- **Empty output file with "N sections failed conversion":** every LLM call
  failed. Check credentials, model name, and network access, then rerun — the
  checkpoint cache will skip any sections that did succeed.

## License

Licensed under the [GNU Affero General Public License v3.0 or later](LICENSE)
(AGPL-3.0-or-later). If you run a modified version of this software as a
network service, you must make the modified source code available to its
users.
