"""
Table OCR and AI CLI module for akoma-markup.
"""

import sys
from pathlib import Path
import click

try:
    from .table_ocr_ai import (
        AzureOCR,
        AzureMultimodalAnalyzer,
        IndiaCodeAnalyzer,
        extract_indiacode_tables,
        test_azure_connectivity
    )
    AZURE_AI_AVAILABLE = True
except ImportError:
    AZURE_AI_AVAILABLE = False


@click.group()
def azure():
    """Azure AI services for IndiaCode document analysis."""
    if not AZURE_AI_AVAILABLE:
        click.echo(click.style("❌ Azure AI module not available", fg="red"))
        click.echo("Install required dependencies: pip install langchain-openai langchain-core")
        sys.exit(1)


@azure.command()
@click.option("--api-key", required=True, help="Azure API key")
@click.option("--multimodal-endpoint", 
              default="https://***.services.ai.azure.com/openai/v1/",
              help="Multimodal AI endpoint (default: Grok)")
@click.option("--deployment", default="grok-4-1-fast-non-reasoning",
              help="Multimodal deployment name")
def test(api_key: str, multimodal_endpoint: str, deployment: str):
    """Test connectivity to Azure AI services."""
    success = test_azure_connectivity(api_key)
    
    if success:
        click.echo(click.style("✅ All Azure AI services accessible", fg="green"))
        sys.exit(0)
    else:
        click.echo(click.style("❌ Azure AI services not accessible", fg="red"))
        sys.exit(1)


@azure.command()
@click.argument("pdf_file", type=click.Path(exists=True))
@click.option("--api-key", required=True, help="Azure API key")
@click.option("--document-intelligence-endpoint", 
              default="https://***.services.ai.azure.com/providers/mistral/azure/ocr",
              help="Document Intelligence (OCR) endpoint")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
def ocr(pdf_file: str, api_key: str, document_intelligence_endpoint: str, output: str):
    """Extract text from PDF using Azure Document Intelligence."""
    pdf_path = Path(pdf_file)
    
    if output:
        output_dir = Path(output)
    else:
        output_dir = pdf_path.parent
    
    ocr_client = AzureOCR(
        api_key=api_key,
        endpoint=document_intelligence_endpoint
    )
    
    click.echo(f"📄 Extracting text from: {pdf_path.name}")
    
    # Extract and save text
    output_file = ocr_client.extract_and_save(pdf_path, output_dir)
    
    click.echo(click.style(f"✓ Extraction complete", fg="green"))
    click.echo(f"📁 Saved to: {output_file}")


@azure.command()
@click.argument("pdf_file", type=click.Path(exists=True))
@click.option("--api-key", required=True, help="Azure API key")
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
    
    click.echo(f"\n{'='*60}")
    click.echo(f"INDIA CODE LEGISLATIVE TABLE ANALYSIS")
    click.echo(f"PDF: {pdf_path.name}")
    click.echo(f"Output: {output_dir}")
    click.echo(f"{'='*60}")
    
    # Initialize analyzer
    analyzer = IndiaCodeAnalyzer(api_key=api_key)
    
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


if __name__ == "__main__":
    azure()