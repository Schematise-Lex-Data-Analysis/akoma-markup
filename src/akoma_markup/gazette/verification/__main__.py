"""CLI command for Gazette refinement verification."""

import asyncio
import json
import logging
from pathlib import Path
import click

from ..util.llm.factory import build_llm
from .engine import LLMVerificationEngine

logger = logging.getLogger(__name__)


def _config_from_env(env_path: Path) -> dict:
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
        config = {
            "provider": "azure",
            "endpoint": env_vars.get("AZURE_INFERENCE_ENDPOINT"),
            "credential": env_vars.get("AZURE_INFERENCE_KEY"),
            "model": env_vars.get("AZURE_INFERENCE_MODEL_ID"),
            "api_style": env_vars.get("AZURE_INFERENCE_API_STYLE"),
        }
    elif "anthropic" in provider or env_vars.get("ANTHROPIC_API_KEY"):
        config = {
            "provider": "anthropic",
            "api_key": env_vars.get("ANTHROPIC_API_KEY"),
            "model": env_vars.get("ANTHROPIC_MODEL_ID"),
        }
    else:
        raise ValueError(f"Could not determine LLM provider from env file: {env_path}")
    
    # Remove None values
    return {k: v for k, v in config.items() if v is not None}


@click.command(name="gazette-refinement-verify")
@click.argument("tsv_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, writable=True),
    help="Output TSV file path",
)
@click.option(
    "--llm-env",
    "llm_env_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .env file with LLM configuration",
)
@click.option(
    "--llm-inline",
    type=str,
    help="Inline JSON string with LLM configuration",
)
@click.option(
    "--llm-json",
    "llm_json_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to JSON file with LLM configuration",
)
@click.option(
    "--verification-level",
    type=click.Choice(["strict", "moderate", "light"], case_sensitive=False),
    default="moderate",
    show_default=True,
    help="Verification strictness level",
)
@click.option(
    "--context-window",
    type=int,
    default=3,
    show_default=True,
    help="Number of surrounding sections to include for context",
)
@click.option(
    "--remove-brackets/--no-remove-brackets",
    default=True,
    show_default=True,
    help="Remove unnecessary square brackets",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Resume from checkpoint if available",
)
@click.option(
    "--batch-size",
    type=int,
    default=5,
    show_default=True,
    help="Number of sections to process in parallel",
)
@click.option(
    "--clear-checkpoint",
    is_flag=True,
    default=False,
    help="Clear existing checkpoint before starting",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging",
)
def gazette_refinement_verify(
    tsv_file,
    output_path,
    llm_env_path,
    llm_inline,
    llm_json_path,
    verification_level,
    context_window,
    remove_brackets,
    resume,
    batch_size,
    clear_checkpoint,
    verbose,
):
    """Verify and clean Gazette refinement output using LLM.
    
    This command performs LLM-powered verification of Gazette refinement output,
    removing unnecessary formatting artifacts, detecting duplicates, and
    normalizing section numbering.
    
    \b
    Examples:
        akoma-markup gazette-refinement-verify refined.tsv -o verified.tsv --llm-env .env
        akoma-markup gazette-refinement-verify input.tsv -o output.tsv --llm-env .env --verification-level strict
        akoma-markup gazette-refinement-verify input.tsv -o output.tsv --llm-env .env --clear-checkpoint --no-resume
    """
    import os
    import sys
    
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Convert paths
    tsv_path = Path(tsv_file)
    output_path = Path(output_path)
    
    # Load LLM configuration
    try:
        if llm_env_path:
            config = _config_from_env(Path(llm_env_path))
        elif llm_json_path:
            with open(llm_json_path) as f:
                config = json.load(f)
        elif llm_inline:
            config = json.loads(llm_inline)
        else:
            raise click.UsageError(
                "Must provide LLM configuration via --llm-env, --llm-json, or --llm-inline"
            )
        
        logger.info(f"Loaded LLM config: provider={config.get('provider')}")
        
    except Exception as e:
        logger.error(f"Failed to load LLM configuration: {e}")
        click.echo(f"Error: Failed to load LLM configuration: {e}", err=True)
        sys.exit(1)
    
    # Build LLM
    try:
        llm = build_llm(config)
        logger.info(f"Built LLM: {llm.__class__.__name__}")
    except Exception as e:
        logger.error(f"Failed to build LLM: {e}")
        click.echo(f"Error: Failed to build LLM: {e}", err=True)
        sys.exit(1)
    
    # Clear checkpoint if requested
    if clear_checkpoint:
        from .checkpoint import VerificationCheckpointManager
        checkpoint_manager = VerificationCheckpointManager()
        if checkpoint_manager.clear_checkpoint(tsv_path, output_path):
            logger.info("Cleared existing checkpoint")
        else:
            logger.info("No checkpoint to clear")
    
    # Create verification engine
    try:
        engine = LLMVerificationEngine(
            llm=llm,
            verification_level=verification_level,
            context_window=context_window,
            remove_brackets=remove_brackets,
            batch_size=batch_size,
        )
        
        logger.info(
            f"Starting verification: "
            f"level={verification_level}, "
            f"context={context_window}, "
            f"batch={batch_size}, "
            f"resume={resume}"
        )
        
    except Exception as e:
        logger.error(f"Failed to create verification engine: {e}")
        click.echo(f"Error: Failed to create verification engine: {e}", err=True)
        sys.exit(1)
    
    # Run verification
    try:
        summary = asyncio.run(engine.verify_tsv(
            tsv_path=tsv_path,
            output_path=output_path,
            resume=resume,
        ))
        
        # Print summary
        click.echo("\n" + "=" * 50)
        click.echo("VERIFICATION SUMMARY")
        click.echo("=" * 50)
        click.echo(f"Input: {summary['input_path']}")
        click.echo(f"Output: {summary['output_path']}")
        click.echo(f"Total sections: {summary['total_sections']}")
        click.echo(f"Verified: {summary['verified_sections'] - summary['failed_sections']}")
        click.echo(f"Failed: {summary['failed_sections']}")
        click.echo(f"Success rate: {summary['success_rate']:.1%}")
        click.echo(f"Average confidence: {summary['avg_confidence']:.2f}")
        click.echo(f"Verification level: {summary['verification_level']}")
        click.echo("=" * 50)
        
        # Save summary to JSON file
        summary_path = output_path.with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        click.echo(f"Error: Verification failed: {e}", err=True)
        sys.exit(1)


# This module provides the gazette_refinement_verify function
# It's imported and used by the main CLI in cli.py