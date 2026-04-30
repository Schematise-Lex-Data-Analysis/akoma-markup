"""CLI entrypoint for akoma-markup."""

import json
import os
import sys
from pathlib import Path

import click


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
            "credential": env_vars.get("AZURE_INFERENCE_CREDENTIAL"),
            "model": env_vars.get("AZURE_MODEL_ID"),
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


def _azure_ai_config_from_env(env_path: str) -> dict:
    """Build Azure AI config dict from a .env file."""
    env_vars = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            env_vars[key.strip()] = value.strip().strip('"').strip("'")
    
    # Extract Azure AI configuration
    config = {
        "api_key": env_vars.get("AZURE_API_KEY"),
        "document_intelligence_endpoint": env_vars.get(
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
            "https://***.services.ai.azure.com/providers/mistral/azure/ocr"
        ),
        "multimodal_endpoint": env_vars.get(
            "AZURE_MULTIMODAL_ENDPOINT",
            "https://***.services.ai.azure.com/openai/v1/"
        ),
        "multimodal_deployment": env_vars.get(
            "AZURE_MULTIMODAL_DEPLOYMENT",
            "grok-4-1-fast-non-reasoning"
        )
    }
    
    if not config["api_key"]:
        raise click.ClickException(
            f"AZURE_API_KEY not found in env file: {env_path}"
        )
    
    return config


@click.group()
def main():
    """Convert IndiaCode law PDFs to Akoma Ntoso markup."""
    pass


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
    help='Inline JSON LLM config, e.g. \'{"provider": "azure", "model": "Llama-3.3-70B-Instruct"}\'',
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
    help="Enable Azure-OCR table rescue. "
         "'declared' takes --table-pages. "
         "'auto' uses a vision LLM to find table pages, then OCRs only those. "
         "'full' OCRs every page (no detection step — most expensive but "
         "guaranteed not to miss anything). Off by default.",
)
@click.option(
    "--table-pages",
    type=str,
    default=None,
    help='Comma/range page list, e.g. "10,12-15". Required with --table-mode=declared.',
)
@click.option(
    "--azure-api-key",
    type=str,
    default=None,
    help="Azure API key for table OCR. Falls back to AZURE_API_KEY env var.",
)
@click.option(
    "--azure-ocr-endpoint",
    type=str,
    default=None,
    help="Override Azure Document Intelligence endpoint. "
         "Falls back to AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT env var.",
)
@click.option(
    "--azure-multimodal-endpoint",
    type=str,
    default=None,
    help="Vision-LLM endpoint for --table-mode=auto. "
         "Falls back to AZURE_MULTIMODAL_ENDPOINT env var.",
)
@click.option(
    "--azure-multimodal-deployment",
    type=str,
    default=None,
    help="Vision-LLM deployment name for --table-mode=auto. "
         "Falls back to AZURE_MULTIMODAL_DEPLOYMENT env var.",
)
def convert(pdf_path, output_path, llm_inline, llm_json_path, llm_env_path,
            table_mode, table_pages, azure_api_key, azure_ocr_endpoint,
            azure_multimodal_endpoint, azure_multimodal_deployment):
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
        from .tables import parse_page_spec

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

        resolved_api_key = (
            azure_api_key
            or env_file_vars.get("AZURE_API_KEY")
            or os.environ.get("AZURE_API_KEY")
        )
        if not resolved_api_key:
            raise click.ClickException(
                "--table-mode requires --azure-api-key, "
                "AZURE_API_KEY in --llm-env, or AZURE_API_KEY env var"
            )
        resolved_ocr_endpoint = (
            azure_ocr_endpoint
            or env_file_vars.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
            or os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        )
        resolved_multimodal_endpoint = (
            azure_multimodal_endpoint
            or env_file_vars.get("AZURE_MULTIMODAL_ENDPOINT")
            or os.environ.get("AZURE_MULTIMODAL_ENDPOINT")
        )
        resolved_multimodal_deployment = (
            azure_multimodal_deployment
            or env_file_vars.get("AZURE_MULTIMODAL_DEPLOYMENT")
            or os.environ.get("AZURE_MULTIMODAL_DEPLOYMENT")
        )
        resolved_multimodal_api_style = (
            env_file_vars.get("AZURE_MULTIMODAL_API_STYLE")
            or os.environ.get("AZURE_MULTIMODAL_API_STYLE")
        )
        if table_mode == "auto" and not (
            resolved_multimodal_endpoint and resolved_multimodal_deployment
        ):
            raise click.ClickException(
                "--table-mode=auto requires --azure-multimodal-endpoint and "
                "--azure-multimodal-deployment (or AZURE_MULTIMODAL_ENDPOINT "
                "and AZURE_MULTIMODAL_DEPLOYMENT in env)"
            )
        if table_mode == "auto" and not resolved_multimodal_api_style:
            raise click.ClickException(
                "--table-mode=auto requires AZURE_MULTIMODAL_API_STYLE in env. "
                "Set it to one of: 'chat', 'responses', 'azure-inference'."
            )
    else:
        resolved_api_key = None
        resolved_ocr_endpoint = None
        resolved_multimodal_endpoint = None
        resolved_multimodal_deployment = None
        resolved_multimodal_api_style = None
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
        azure_api_key=resolved_api_key,
        azure_ocr_endpoint=resolved_ocr_endpoint,
        azure_multimodal_endpoint=resolved_multimodal_endpoint,
        azure_multimodal_deployment=resolved_multimodal_deployment,
        azure_multimodal_api_style=resolved_multimodal_api_style,
    )
    click.echo(result)


# Azure AI commands (optional)
try:
    import os
    from pathlib import Path
    from .table_ocr_ai import (
        AzureOCR,
        IndiaCodeAnalyzer,
        test_azure_connectivity
    )
    
    @main.group()
    def azure():
        """Azure AI services for IndiaCode document analysis."""
        pass
    
    @azure.command()
    @click.option("--llm-env",
                  "llm_env_path",
                  type=click.Path(exists=True),
                  default=None,
                  help="Path to a .env file with Azure AI credentials")
    @click.option("--api-key", help="Azure API key (default: AZURE_API_KEY env var)")
    @click.option("--multimodal-endpoint",
                  help="Multimodal AI endpoint (default: AZURE_MULTIMODAL_ENDPOINT env var)")
    @click.option("--multimodal-deployment",
                  help="Multimodal deployment name (default: AZURE_MULTIMODAL_DEPLOYMENT env var)")
    def test(llm_env_path: str, api_key: str, multimodal_endpoint: str, multimodal_deployment: str):
        """Test connectivity to Azure AI services."""
        # Configuration precedence: llm-env > command-line args > environment variables
        if llm_env_path:
            config = _azure_ai_config_from_env(llm_env_path)
            actual_api_key = config["api_key"]
            actual_multimodal_endpoint = config.get("multimodal_endpoint", multimodal_endpoint)
            actual_multimodal_deployment = config.get("multimodal_deployment", multimodal_deployment)
        else:
            actual_api_key = api_key or os.environ.get("AZURE_API_KEY")
            actual_multimodal_endpoint = multimodal_endpoint or os.environ.get("AZURE_MULTIMODAL_ENDPOINT")
            actual_multimodal_deployment = multimodal_deployment or os.environ.get("AZURE_MULTIMODAL_DEPLOYMENT")
        
        if not actual_api_key:
            raise click.ClickException(
                "Azure API key required. Provide via --llm-env, --api-key, or set AZURE_API_KEY environment variable."
            )
        
        success = test_azure_connectivity(actual_api_key, actual_multimodal_endpoint, actual_multimodal_deployment)
        
        if success:
            click.echo(click.style("✅ All Azure AI services accessible", fg="green"))
            sys.exit(0)
        else:
            click.echo(click.style("❌ Azure AI services not accessible", fg="red"))
            sys.exit(1)
    
    @azure.command()
    @click.argument("pdf_file", type=click.Path(exists=True))
    @click.option("--api-key", help="Azure API key (default: AZURE_API_KEY env var)")
    @click.option("--document-intelligence-endpoint", 
                  help="Document Intelligence (OCR) endpoint (default: AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT env var)")
    @click.option("--output", "-o", type=click.Path(), help="Output directory")
    def document_intelligence(pdf_file: str, api_key: str, document_intelligence_endpoint: str, 
                             output: str):
        """Extract text from PDF using Azure Document Intelligence."""
        pdf_path = Path(pdf_file)
        
        if output:
            output_dir = Path(output)
        else:
            output_dir = pdf_path.parent
        
        # Use environment variables if not provided
        actual_api_key = api_key or os.environ.get("AZURE_API_KEY")
        if not actual_api_key:
            raise click.ClickException(
                "Azure API key required. Provide via --api-key or set AZURE_API_KEY environment variable."
            )
        actual_endpoint = (
            document_intelligence_endpoint
            or os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        )
        if not actual_endpoint:
            raise click.ClickException(
                "Azure OCR endpoint required. Provide via --document-intelligence-endpoint "
                "or set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT environment variable."
            )

        ocr_client = AzureOCR(
            api_key=actual_api_key,
            endpoint=actual_endpoint,
        )
        
        click.echo(f"📄 Extracting text from: {pdf_path.name}")
        
        # Extract and save text
        output_file = ocr_client.extract_and_save(pdf_path, output_dir)
        
        click.echo(click.style(f"✓ Extraction complete", fg="green"))
        click.echo(f"📁 Saved to: {output_file}")
    
    @azure.command()
    @click.argument("pdf_file", type=click.Path(exists=True))
    @click.option("--api-key", help="Azure API key (default: AZURE_API_KEY env var)")
    @click.option("--analysis", "-a", multiple=True,
                  type=click.Choice(["summary", "structure", "content", "json"]),
                  default=["summary", "structure", "content"],
                  help="Types of analysis to perform")
    @click.option("--output", "-o", type=click.Path(), help="Output directory")
    @click.option("--html-report", is_flag=True, help="Generate HTML report")
    def indiacode_tables(pdf_file: str, api_key: str, analysis: list, output: str,
                         html_report: bool):
        """Complete analysis of IndiaCode legislative tables."""
        pdf_path = Path(pdf_file)
        
        if output:
            output_dir = Path(output)
        else:
            output_dir = pdf_path.parent / "indiacode_analysis"
        
        # Use environment variables if not provided
        actual_api_key = api_key or os.environ.get("AZURE_API_KEY")
        if not actual_api_key:
            raise click.ClickException(
                "Azure API key required. Provide via --api-key or set AZURE_API_KEY environment variable."
            )
        
        click.echo(f"\n{'='*60}")
        click.echo(f"INDIA CODE LEGISLATIVE TABLE ANALYSIS")
        click.echo(f"PDF: {pdf_path.name}")
        click.echo(f"Output: {output_dir}")
        click.echo(f"{'='*60}")
        
        # Initialize analyzer (uses env vars for endpoints if not provided)
        analyzer = IndiaCodeAnalyzer(api_key=actual_api_key)
        
        # Run complete analysis
        results = analyzer.analyze_pdf(
            pdf_path=pdf_path,
            analysis_types=list(analysis),
            output_dir=output_dir
        )
        
        # Create HTML report if requested
        if html_report and results.get("files"):
            html_file = output_dir / f"{pdf_path.stem}_report.html"
            analyzer.create_html_report(results, html_file)
            results["files"]["html_report"] = html_file
        
        click.echo(click.style(f"\n✅ ANALYSIS COMPLETE!", fg="green"))
        click.echo(f"📊 Files generated:")
        
        for name, filepath in results.get("files", {}).items():
            if hasattr(filepath, 'name'):
                click.echo(f"  • {name}: {filepath.name}")
            else:
                click.echo(f"  • {name}: {filepath}")
        
        if results.get("errors"):
            click.echo(click.style(f"\n⚠️  Errors encountered:", fg="yellow"))
            for error in results["errors"]:
                click.echo(f"  • {error}")
    
    @main.command()
    @click.argument("pdf_path", type=click.Path(exists=True))
    @click.option(
        "--llm-env",
        "llm_env_path",
        type=click.Path(exists=True),
        default=None,
        help="Path to a .env file with Azure AI credentials"
    )
    @click.option(
        "--api-key",
        help="Azure API key for Document Intelligence and Multimodal AI (default: AZURE_API_KEY env var)"
    )
    @click.option(
        "--document-intelligence-endpoint",
        help="Document Intelligence (OCR) endpoint (default: AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT env var)"
    )
    @click.option(
        "--multimodal-endpoint",
        help="Multimodal AI endpoint for analysis (default: AZURE_MULTIMODAL_ENDPOINT env var)"
    )
    @click.option(
        "--multimodal-deployment",
        help="Multimodal deployment name (default: AZURE_MULTIMODAL_DEPLOYMENT env var)"
    )
    @click.option(
        "-o",
        "--output",
        "output_path",
        type=click.Path(),
        default=None,
        help="Output directory for analysis results"
    )
    def analyze_tables(pdf_path: str, llm_env_path: str, api_key: str, document_intelligence_endpoint: str,
                       multimodal_endpoint: str, multimodal_deployment: str,
                       output_path: str):
        """Extract and analyze tables from IndiaCode PDF using Azure AI."""
        pdf_path_obj = Path(pdf_path)
        
        if output_path:
            output_dir = Path(output_path)
        else:
            output_dir = pdf_path_obj.parent / "table_analysis"
        
        # Configuration precedence: llm-env > command-line args > environment variables
        if llm_env_path:
            config = _azure_ai_config_from_env(llm_env_path)
            actual_api_key = config["api_key"]
            actual_document_intelligence_endpoint = config.get("document_intelligence_endpoint")
            actual_multimodal_endpoint = config.get("multimodal_endpoint")
            actual_multimodal_deployment = config.get("multimodal_deployment")
        else:
            # Use command-line arguments with environment variable fallbacks
            actual_api_key = api_key or os.environ.get("AZURE_API_KEY")
            actual_document_intelligence_endpoint = document_intelligence_endpoint or os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
            actual_multimodal_endpoint = multimodal_endpoint or os.environ.get("AZURE_MULTIMODAL_ENDPOINT")
            actual_multimodal_deployment = multimodal_deployment or os.environ.get("AZURE_MULTIMODAL_DEPLOYMENT")
        
        if not actual_api_key:
            raise click.ClickException(
                "Azure API key required. Provide via --llm-env, --api-key, or set AZURE_API_KEY environment variable."
            )
        
        click.echo(f"\n{'='*60}")
        click.echo(f"INDIA CODE TABLE ANALYSIS")
        click.echo(f"PDF: {pdf_path_obj.name}")
        click.echo(f"Document Intelligence: {actual_document_intelligence_endpoint or 'default'}")
        click.echo(f"Multimodal AI: {actual_multimodal_endpoint or 'default'}/{actual_multimodal_deployment or 'default'}")
        click.echo(f"Output: {output_dir}")
        click.echo(f"{'='*60}")
        
        # Initialize analyzer with config
        analyzer = IndiaCodeAnalyzer(
            api_key=actual_api_key,
            ocr_endpoint=actual_document_intelligence_endpoint,
            multimodal_endpoint=actual_multimodal_endpoint,
            multimodal_deployment=actual_multimodal_deployment
        )
        
        # Run analysis
        results = analyzer.analyze_pdf(
            pdf_path=pdf_path_obj,
            output_dir=output_dir
        )
        
        click.echo(click.style(f"\n✅ ANALYSIS COMPLETE!", fg="green"))
        
        # Create HTML report
        html_file = output_dir / f"{pdf_path_obj.stem}_report.html"
        analyzer.create_html_report(results, html_file)
        
        click.echo(f"📊 Files generated in {output_dir}:")
        for name, filepath in results.get("files", {}).items():
            if hasattr(filepath, 'name'):
                click.echo(f"  • {name}: {filepath.name}")
        
        click.echo(f"  • html_report: {html_file.name}")
        
        click.echo(f"\n💡 Open the report: file://{html_file.absolute()}")

except ImportError:
    # Azure AI module not available - skip adding commands
    pass
