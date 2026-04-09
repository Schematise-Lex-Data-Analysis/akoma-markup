"""CLI entrypoint for akoma-markup."""

import json
import os

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


@click.command()
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
def main(pdf_path, output_path, llm_inline, llm_json_path, llm_env_path):
    """Convert a BNSS 2023 PDF to Akoma Ntoso markup."""
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

    from . import convert

    result = convert(pdf_path, llm_config=config, output_path=output_path)
    click.echo(result)
