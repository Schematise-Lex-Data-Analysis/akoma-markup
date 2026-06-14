"""CLI entrypoint for akoma-markup."""

import json
import logging
import os

import click

import dotenv


def _config_from_env(env_path: str) -> dict:
    """Build an LLM config dict from a .env file."""
    env_vars = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            env_vars[key.strip()] = value.strip().strip('"').strip("'")

    provider = env_vars.get("PROVIDER", "").lower()
    if "azure" in provider or env_vars.get("AZURE_INFERENCE_ENDPOINT"):
        return {
            "provider": "azure",
            "endpoint": env_vars.get("AZURE_INFERENCE_ENDPOINT"),
            "credential": env_vars.get("AZURE_INFERENCE_KEY"),
            "model": env_vars.get("AZURE_INFERENCE_MODEL_ID"),
            "api_style": env_vars.get("AZURE_INFERENCE_API_STYLE"),
        }
    if "anthropic" in provider or env_vars.get("ANTHROPIC_API_KEY"):
        return {
            "provider": "anthropic",
            "api_key": env_vars.get("ANTHROPIC_API_KEY"),
            "model": env_vars.get("ANTHROPIC_MODEL_ID"),
        }

    raise click.ClickException(
        f"Could not determine LLM provider from env file: {env_path}"
    )


@click.group()
def main():
    """Convert IndiaCode law PDFs to Akoma Ntoso markup."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@main.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(),
    default=None,
    help="Destination for the markup file. Defaults to <pdf_stem>_markup.txt.",
)
@click.option(
    "--llm-inline",
    "llm_inline",
    type=str,
    default=None,
    help='Inline JSON LLM config, e.g. \'{"provider": "azure", "model": "<deployment-name>"}\'',
)
@click.option(
    "--llm-json",
    "llm_json_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to a JSON file with LLM config.",
)
@click.option(
    "--llm-env",
    "llm_env_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to a .env file with LLM credentials.",
)
@click.option(
    "--table-mode",
    type=click.Choice(["declared", "auto", "full"]),
    default=None,
    help="Enable vision-LLM table rescue. "
         "'declared' takes --table-pages. "
         "'auto' classifies every page (cheap), re-extracting only flagged "
         "ones. 'full' re-extracts every page (no classification step — "
         "most expensive but guaranteed not to miss anything). Off by default.",
)
@click.option(
    "--table-pages",
    type=str,
    default=None,
    help='Comma/range page list, e.g. "10,12-15". Required with --table-mode=declared.',
)
@click.option(
    "--azure-vision-key",
    type=str,
    default=None,
    help="Vision-LLM API key. Required with --table-mode. "
         "Falls back to AZURE_VISION_KEY env var.",
)
@click.option(
    "--azure-vision-endpoint",
    type=str,
    default=None,
    help="Vision-LLM endpoint. Required with --table-mode. "
         "Falls back to AZURE_VISION_ENDPOINT env var.",
)
@click.option(
    "--azure-vision-model",
    type=str,
    default=None,
    help="Vision-LLM model/deployment name. Required with --table-mode. "
         "Falls back to AZURE_VISION_MODEL env var.",
)
@click.option(
    "--azure-vision-max-tokens",
    type=int,
    default=None,
    help="Per-page output token budget for the vision-LLM extraction call. "
         "Falls back to AZURE_VISION_MAX_TOKENS env var, then 16384. Bump "
         "this if you see truncation warnings on dense schedule pages.",
)
def convert(pdf_path, output_path, llm_inline, llm_json_path, llm_env_path,
            table_mode, table_pages, azure_vision_key,
            azure_vision_endpoint, azure_vision_model,
            azure_vision_max_tokens):
    """Convert a IndiaCode law PDF to Akoma Ntoso markup."""
    sources = [s for s in [llm_inline, llm_json_path, llm_env_path] if s]
    if len(sources) == 0:
        raise click.ClickException("Provide one of: --llm-inline, --llm-json, or --llm-env")
    if len(sources) > 1:
        raise click.ClickException("Use only one of: --llm-inline, --llm-json, or --llm-env")

    if llm_env_path:
        config = _config_from_env(llm_env_path)
    elif llm_json_path:
        with open(llm_json_path) as f:
            config = json.load(f)
    else:
        try:
            config = json.loads(llm_inline)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Invalid JSON in --llm-inline: {exc}")

    parsed_table_pages = None
    if table_mode is not None:
        from .parsing.tables.rescue import parse_page_spec

        if table_mode == "declared":
            if not table_pages:
                raise click.ClickException(
                    "--table-mode=declared requires --table-pages"
                )
            try:
                parsed_table_pages = parse_page_spec(table_pages)
            except ValueError as exc:
                raise click.ClickException(f"Invalid --table-pages: {exc}")
        elif table_pages:
            raise click.ClickException(
                f"--table-pages is only valid with --table-mode=declared "
                f"(got --table-mode={table_mode})"
            )

        env_file_vars: dict[str, str] = {}
        if llm_env_path:
            with open(llm_env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    key, _, value = line.partition("=")
                    env_file_vars[key.strip()] = value.strip().strip('"').strip("'")

        resolved_vision_key = (
            azure_vision_key
            or env_file_vars.get("AZURE_VISION_KEY")
            or os.environ.get("AZURE_VISION_KEY")
        )
        if not resolved_vision_key:
            raise click.ClickException(
                "--table-mode requires --azure-vision-key or "
                "AZURE_VISION_KEY in env"
            )
        resolved_vision_endpoint = (
            azure_vision_endpoint
            or env_file_vars.get("AZURE_VISION_ENDPOINT")
            or os.environ.get("AZURE_VISION_ENDPOINT")
        )
        if not resolved_vision_endpoint:
            raise click.ClickException(
                "--table-mode requires --azure-vision-endpoint or "
                "AZURE_VISION_ENDPOINT in env"
            )
        resolved_vision_model = (
            azure_vision_model
            or env_file_vars.get("AZURE_VISION_MODEL")
            or os.environ.get("AZURE_VISION_MODEL")
        )
        if not resolved_vision_model:
            raise click.ClickException(
                "--table-mode requires --azure-vision-model or "
                "AZURE_VISION_MODEL in env"
            )
        resolved_vision_api_style = (
            env_file_vars.get("AZURE_VISION_API_STYLE")
            or os.environ.get("AZURE_VISION_API_STYLE")
        )
        if not resolved_vision_api_style:
            raise click.ClickException(
                "--table-mode requires AZURE_VISION_API_STYLE in env. "
                "Set it to one of: 'chat', 'responses', 'azure-inference'."
            )
        resolved_vision_max_tokens = azure_vision_max_tokens
        if resolved_vision_max_tokens is None:
            raw = (
                env_file_vars.get("AZURE_VISION_MAX_TOKENS")
                or os.environ.get("AZURE_VISION_MAX_TOKENS")
            )
            if raw:
                try:
                    resolved_vision_max_tokens = int(raw)
                except ValueError:
                    raise click.ClickException(
                        f"AZURE_VISION_MAX_TOKENS must be an integer; got {raw!r}"
                    )
    else:
        resolved_vision_key = None
        resolved_vision_endpoint = None
        resolved_vision_model = None
        resolved_vision_api_style = None
        resolved_vision_max_tokens = None
        if table_pages:
            raise click.ClickException(
                "--table-pages requires --table-mode=declared"
            )

    from . import convert as convert_func

    result = convert_func(
        pdf_path,
        llm_config=config,
        output_path=output_path,
        table_mode=table_mode,
        table_pages=parsed_table_pages,
        azure_vision_key=resolved_vision_key,
        azure_vision_endpoint=resolved_vision_endpoint,
        azure_vision_model=resolved_vision_model,
        azure_vision_api_style=resolved_vision_api_style,
        azure_vision_max_tokens=resolved_vision_max_tokens,
    )
    click.echo(result)


@main.command(name="gazette-convert")
@click.argument("tsv_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, writable=True),
    help="Output markup file path.",
)
@click.option(
    "--llm-env",
    "llm_env_file",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .env file with LLM configuration.",
)
@click.option(
    "--llm-inline",
    type=str,
    help='JSON string with LLM configuration.',
)
@click.option(
    "--llm-json",
    "llm_json_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a JSON file with LLM config.",
)
@click.option(
    "--document-name",
    default=None,
    help="Document name for the gazette (defaults to TSV stem).",
)
def gazette_convert(
    tsv_file,
    output_path,
    llm_env_file,
    llm_inline,
    llm_json_path,
    document_name,
):
    """Convert a Gazette TSV to legislative markup (Phase 3).

    Takes the TSV output from gazette-extract and converts each section
    to Akoma Ntoso markup using LLM.

    Example:
        akoma-markup gazette-extract gazette.pdf -o ./output --llm-env .env
        akoma-markup gazette-convert ./output/gazette_sections.tsv -o markup.txt --llm-env .env
    """
    sources = [s for s in [llm_inline, llm_json_path, llm_env_file] if s]
    if len(sources) == 0:
        raise click.ClickException(
            "Provide one of: --llm-inline, --llm-json, or --llm-env"
        )
    if len(sources) > 1:
        raise click.ClickException(
            "Use only one of: --llm-inline, --llm-json, or --llm-env"
        )

    # Parse LLM config
    if llm_env_file:
        config = _config_from_env(llm_env_file)
    elif llm_json_path:
        with open(llm_json_path) as f:
            config = json.load(f)
    else:
        try:
            config = json.loads(llm_inline)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Invalid JSON in --llm-inline: {exc}")

    from . import convert_gazette_from_tsv

    result = convert_gazette_from_tsv(
        tsv_path=tsv_file,
        output_path=output_path,
        llm_config=config,
        document_name=document_name,
    )
    click.echo(result)


@main.command(name="gazette-extract")
@click.argument("gazette_pdf", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False, writable=True),
    help="Output directory for TSV and intermediate files.",
)
@click.option(
    "--llm-env",
    "llm_env_file",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .env file with LLM configuration.",
)
@click.option(
    "--llm-inline",
    type=str,
    help='JSON string with LLM configuration.',
)
@click.option(
    "--llm-json",
    "llm_json_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a JSON file with LLM config.",
)
@click.option(
    "--table-mode",
    type=click.Choice(["declared", "auto", "full"]),
    default=None,
    help="Enable table rescue mode.",
)
@click.option(
    "--table-pages",
    type=str,
    default=None,
    help='Comma/range page list for table rescue, e.g. "10,12-15". '
         'Required with --table-mode=declared.',
)
@click.option(
    "--azure-vision-key",
    type=str,
    default=None,
    help="Vision-LLM API key. Falls back to AZURE_VISION_KEY env var.",
)
@click.option(
    "--azure-vision-endpoint",
    type=str,
    default=None,
    help="Vision-LLM endpoint. Falls back to AZURE_VISION_ENDPOINT env var.",
)
@click.option(
    "--azure-vision-model",
    type=str,
    default=None,
    help="Vision-LLM model/deployment name. "
         "Falls back to AZURE_VISION_MODEL env var.",
)
@click.option(
    "--azure-vision-api-style",
    type=str,
    default=None,
    help="Vision-LLM API style - 'chat', 'responses', or 'azure-inference'. "
         "Falls back to AZURE_VISION_API_STYLE env var.",
)
@click.option(
    "--azure-vision-max-tokens",
    type=int,
    default=None,
    help="Per-page output token budget for the vision-LLM. "
         "Falls back to AZURE_VISION_MAX_TOKENS env var, then 16384.",
)
@click.option(
    "--skip-extraction",
    is_flag=True,
    help="Skip extraction and reuse existing TSV from cache.",
)
@click.option(
    "--dpi",
    type=int,
    default=200,
    help="DPI for rendering PDF pages to images (default: 200).",
)
@click.option(
    "--max-workers",
    type=int,
    default=4,
    help="Number of parallel workers for page analysis (default: 4).",
)
def gazette_extract(
    gazette_pdf,
    output_dir,
    llm_env_file,
    llm_inline,
    llm_json_path,
    table_mode,
    table_pages,
    azure_vision_key,
    azure_vision_endpoint,
    azure_vision_model,
    azure_vision_api_style,
    azure_vision_max_tokens,
    skip_extraction,
    dpi,
    max_workers,
):
    """Extract sections from a Gazette PDF using AI-powered analysis.

    Phase 1 of the Gazette conversion pipeline: Uses multimodal vision LLM
    to intelligently identify document structure, extract sections with
    hierarchies, filter non-English content, and optionally rescue tables.

    Outputs a TSV file compatible with akoma-markup's standard format, plus
    intermediate JSON cache for resumable processing.

    Example:
        akoma-markup gazette-extract gazette.pdf -o ./output --llm-env .env

        akoma-markup gazette-extract gazette.pdf -o ./output --llm-env .env \\
            --table-mode auto
    """
    sources = [s for s in [llm_inline, llm_json_path, llm_env_file] if s]
    if len(sources) == 0:
        raise click.ClickException(
            "Provide one of: --llm-inline, --llm-json, or --llm-env"
        )
    if len(sources) > 1:
        raise click.ClickException(
            "Use only one of: --llm-inline, --llm-json, or --llm-env"
        )

    # Parse LLM config (needed for table conversion later)
    if llm_env_file:
        dotenv.load_dotenv(llm_env_file)
        config = _config_from_env(llm_env_file)
    elif llm_json_path:
        with open(llm_json_path) as f:
            config = json.load(f)
    else:
        try:
            config = json.loads(llm_inline)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Invalid JSON in --llm-inline: {exc}")

    # Resolve vision credentials
    resolved_vision_key = azure_vision_key or os.environ.get(
        "AZURE_VISION_KEY"
    )
    resolved_vision_endpoint = azure_vision_endpoint or os.environ.get(
        "AZURE_VISION_ENDPOINT"
    )
    resolved_vision_model = azure_vision_model or os.environ.get(
        "AZURE_VISION_MODEL"
    )
    resolved_vision_api_style = azure_vision_api_style or os.environ.get(
        "AZURE_VISION_API_STYLE", "chat"
    )
    resolved_vision_max_tokens = azure_vision_max_tokens
    if resolved_vision_max_tokens is None:
        raw = os.environ.get("AZURE_VISION_MAX_TOKENS")
        if raw:
            try:
                resolved_vision_max_tokens = int(raw)
            except ValueError:
                raise click.ClickException(
                    f"AZURE_VISION_MAX_TOKENS must be an integer; got {raw!r}"
                )

    if not resolved_vision_key:
        raise click.ClickException(
            "Vision LLM key required. Provide --azure-vision-key or "
            "set AZURE_VISION_KEY env var."
        )
    if not resolved_vision_endpoint:
        raise click.ClickException(
            "Vision LLM endpoint required. Provide --azure-vision-endpoint or "
            "set AZURE_VISION_ENDPOINT env var."
        )
    if not resolved_vision_model:
        raise click.ClickException(
            "Vision LLM model required. Provide --azure-vision-model or "
            "set AZURE_VISION_MODEL env var."
        )

    # Parse table pages if declared mode
    parsed_table_pages = None
    if table_mode == "declared":
        if not table_pages:
            raise click.ClickException(
                "--table-mode=declared requires --table-pages"
            )
        from .parsing.tables.rescue import parse_page_spec

        try:
            parsed_table_pages = parse_page_spec(table_pages)
        except ValueError as exc:
            raise click.ClickException(f"Invalid --table-pages: {exc}")

    # Initialize vision client
    from .util.llm.vision import VisionClient

    vision_client = VisionClient(
        api_key=resolved_vision_key,
        endpoint=resolved_vision_endpoint,
        deployment=resolved_vision_model,
        api_mode=resolved_vision_api_style,
        extraction_max_tokens=resolved_vision_max_tokens,
    )

    # Run extraction
    from .gazette import extract_sections_with_ai
    from pathlib import Path

    pdf_path = Path(gazette_pdf)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if skip_extraction:
        click.echo(f"Skipping extraction (using cache): {output_path}")

    tsv_df = extract_sections_with_ai(
        pdf_path=pdf_path,
        vision_client=vision_client,
        output_dir=output_path,
        max_workers=max_workers,
        dpi=dpi,
    )

    # Save TSV
    tsv_path = output_path / f"{pdf_path.stem}_sections.tsv"
    tsv_df.to_csv(tsv_path, sep="\t", index=False)

    click.echo(f"Extraction complete!")
    click.echo(f"  TSV: {tsv_path}")
    click.echo(f"  Sections: {len(tsv_df)}")
    click.echo(f"  Cache: {output_path / '.akoma_cache'}")


@main.command(name="gazette-refine")
@click.argument("tsv_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, writable=True),
    help="Output TSV file path.",
)
@click.option(
    "--token-limit",
    type=int,
    default=8000,
    help="Maximum tokens per chunk (default: 8000).",
)
@click.option(
    "--tokenizer",
    type=click.Choice(["tiktoken", "approx"]),
    default="approx",
    help="Tokenizer method (default: approx).",
)
def gazette_refine(
    tsv_file,
    output_path,
    token_limit,
    tokenizer,
):
    """Group and split Gazette TSV sections hierarchically.

    This command concatenates subsections under their parent main sections
    and splits content only when token limits are crossed. Outputs a TSV
    that can be processed by gazette-convert.

    Example:
        akoma-markup gazette-extract gazette.pdf -o ./output --llm-env .env
        akoma-markup gazette-refine ./output/gazette_sections.tsv \\
            -o refined_sections.tsv --token-limit 8000
        akoma-markup gazette-convert refined_sections.tsv -o markup.txt \\
            --llm-env .env
    """
    from .gazette.refine import refine_gazette_grouping

    result = refine_gazette_grouping(
        tsv_path=tsv_file,
        output_path=output_path,
        token_limit=token_limit,
        tokenizer=tokenizer,
    )
    click.echo(f"Created: {result['metadata']['output_path']}")
    click.echo(f"Groups: {result['metadata']['groups_created']}")
    click.echo(f"Chunks: {result['metadata']['chunks_created']}")


@main.command()
@click.argument("base_markup", type=click.Path(exists=True, dir_okay=False))
@click.argument("amendments_csv", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, writable=True),
    help="Output amended markup file path.",
)
@click.option(
    "--llm-env",
    "llm_env_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to .env file with LLM configuration.",
)
@click.option(
    "--llm-inline",
    type=str,
    default=None,
    help='JSON string with LLM configuration.',
)
@click.option(
    "--llm-json",
    "llm_json_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a JSON file with LLM config.",
)
@click.option(
    "--act-name",
    type=str,
    default=None,
    help="Name of the act (defaults to base markup filename).",
)
@click.option(
    "--base-version",
    type=str,
    default="original",
    help="Base version identifier.",
)
def amend(
    base_markup,
    amendments_csv,
    output_path,
    llm_env_path,
    llm_inline,
    llm_json_path,
    act_name,
    base_version,
):
    """Apply amendments to base legislation markup.

    Reads amendments from CSV and applies them to base markup using LLM.

    CSV format (expected columns):
        section,operation,new_text,effective_date,amendment_act

    Operations: replace, insert, delete

    Example:
        akoma-markup amend it_act_2000.txt amendments.csv -o it_act_2008.txt --llm-env .env
    """
    sources = [s for s in [llm_inline, llm_json_path, llm_env_path] if s]
    if len(sources) == 0:
        raise click.ClickException("Provide one of: --llm-inline, --llm-json, or --llm-env")
    if len(sources) > 1:
        raise click.ClickException("Use only one of: --llm-inline, --llm-json, or --llm-env")

    # Parse LLM config
    if llm_env_path:
        config = _config_from_env(llm_env_path)
    elif llm_json_path:
        with open(llm_json_path) as f:
            config = json.load(f)
    else:
        try:
            config = json.loads(llm_inline)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Invalid JSON in --llm-inline: {exc}")

    # Parse amendments CSV
    import pandas as pd
    import csv
    
    try:
        df = pd.read_csv(amendments_csv)
        required = ["section", "operation", "new_text", "effective_date", "amendment_act"]
        missing = set(required) - set(df.columns)
        if missing:
            raise click.ClickException(f"Missing columns in CSV: {sorted(missing)}")
        
        amendments = df.to_dict(orient="records")
    except Exception as exc:
        raise click.ClickException(f"Error parsing amendments CSV: {exc}")

    # Set act name
    if not act_name:
        from pathlib import Path
        act_name = Path(base_markup).stem

    # Call amend function
    from . import amend as amend_func
    
    try:
        output_file, errors = amend_func(
            base_markup=base_markup,
            amendments=amendments,
            llm_config=config,
            output_path=output_path,
            act_name=act_name,
            base_version=base_version,
        )
        
        click.echo(f"Applied {len(amendments) - len(errors)} amendments")
        click.echo(f"Amended markup written to: {output_file}")
        
        if errors:
            click.echo(f"\nErrors ({len(errors)}):")
            for err in errors[:5]:  # Show first 5 errors
                click.echo(f"  Section {err['section']}: {err['error'][:60]}...")
            if len(errors) > 5:
                click.echo(f"  ... and {len(errors) - 5} more")
    except Exception as exc:
        raise click.ClickException(f"Amendment failed: {exc}")


@main.command(name="batch-amend")
@click.argument("registry_csv", type=click.Path(exists=True, dir_okay=False))
@click.argument("gazette_folder", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-o",
    "--output",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False, writable=True),
    help="Output directory for amended versions.",
)
@click.option(
    "--llm-env",
    "llm_env_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to .env file with LLM configuration.",
)
@click.option(
    "--llm-inline",
    type=str,
    default=None,
    help='JSON string with LLM configuration.',
)
@click.option(
    "--llm-json",
    "llm_json_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a JSON file with LLM config.",
)
def batch_amend(
    registry_csv,
    gazette_folder,
    output_dir,
    llm_env_path,
    llm_inline,
    llm_json_path,
):
    """Batch process amendments from registry CSV.

    Processes all amendments in registry, applying them to base acts
    and saving versioned outputs.

    Example:
        akoma-markup batch-amend amendments.csv gazettes/ -o versions/ --llm-env .env
    """
    sources = [s for s in [llm_inline, llm_json_path, llm_env_path] if s]
    if len(sources) == 0:
        raise click.ClickException("Provide one of: --llm-inline, --llm-json, or --llm-env")
    if len(sources) > 1:
        raise click.ClickException("Use only one of: --llm-inline, --llm-json, or --llm-env")

    # Parse LLM config
    if llm_env_path:
        config = _config_from_env(llm_env_path)
    elif llm_json_path:
        with open(llm_json_path) as f:
            config = json.load(f)
    else:
        try:
            config = json.loads(llm_inline)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Invalid JSON in --llm-inline: {exc}")

    # Call batch_amend function
    from . import batch_amend as batch_amend_func
    
    try:
        results = batch_amend_func(
            registry_csv=registry_csv,
            gazette_folder=gazette_folder,
            llm_config=config,
            output_dir=output_dir,
        )
        
        click.echo(f"Processed {len(results)} acts:")
        for act_name, paths in results.items():
            click.echo(f"  {act_name}:")
            for path in paths:
                click.echo(f"    - {path}")
    except Exception as exc:
        raise click.ClickException(f"Batch amendment failed: {exc}")


@main.command()
@click.argument("markup1", type=click.Path(exists=True, dir_okay=False))
@click.argument("markup2", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, writable=True),
    help="Output file for diff report (optional).",
)
@click.option(
    "--context-lines",
    type=int,
    default=3,
    help="Number of context lines in diff (default: 3).",
)
@click.option(
    "--format",
    "format_type",
    type=click.Choice(["text", "section", "summary"]),
    default="summary",
    help="Diff format: text (unified), section (structured), summary (report).",
)
def diff(
    markup1,
    markup2,
    output_path,
    context_lines,
    format_type,
):
    """Generate diff between two legislation versions.

    Compares two markup files and shows differences.

    Example:
        akoma-markup diff it_act_2000.txt it_act_2008.txt --format summary
    """
    from pathlib import Path
    from .amendment import (
        generate_text_diff,
        generate_section_diff,
        generate_summary_report,
    )
    
    # Read files
    try:
        with open(markup1, "r", encoding="utf-8") as f:
            content1 = f.read()
        with open(markup2, "r", encoding="utf-8") as f:
            content2 = f.read()
    except Exception as exc:
        raise click.ClickException(f"Error reading files: {exc}")

    # Generate diff
    if format_type == "text":
        result = generate_text_diff(content1, content2, context_lines)
    elif format_type == "section":
        result = generate_section_diff(content1, content2)
        # Pretty print dict
        import json
        result = json.dumps(result, indent=2)
    elif format_type == "summary":
        section_diff = generate_section_diff(content1, content2)
        result = generate_summary_report(section_diff)

    # Output
    if output_path:
        Path(output_path).write_text(result, encoding="utf-8")
        click.echo(f"Diff written to: {output_path}")
    else:
        click.echo(result)