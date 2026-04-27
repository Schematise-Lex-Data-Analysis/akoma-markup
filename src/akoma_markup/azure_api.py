"""Shared API-mode constants and validation for Azure LLM call sites.

The three transports we support:
    - ``chat``            OpenAI-compat chat/completions via openai SDK
                          (Bearer auth, no api-version).
    - ``responses``       OpenAI-compat Responses API via openai SDK
                          (Bearer auth, no api-version).
    - ``azure-inference`` Older Azure AI Inference path via azure-ai-inference
                          SDK (api-key header + ?api-version=... query param).

Both ``vision_llm.py`` (per-page table classification) and ``llm.py`` (main
section-conversion LLM) read their ``AZURE_*_API_STYLE`` env var through
``validate_api_mode`` so a single ``.env`` swap changes both consistently.
"""

from __future__ import annotations

from typing import Literal

ApiMode = Literal["chat", "responses", "azure-inference"]
_VALID_API_MODES: tuple[str, ...] = ("chat", "responses", "azure-inference")


def validate_api_mode(value: str | None, env_var_name: str) -> ApiMode:
    """Validate an API-mode string read from a config dict or env var.

    Raises:
        ValueError: with a clear remediation message if missing or invalid.
    """
    if not value:
        raise ValueError(
            f"{env_var_name} is required. Set it in your .env file to one of:\n"
            f"  - 'chat'             OpenAI-compat chat/completions "
            f"(Phi-4, GPT-4o, Llama on newer resources)\n"
            f"  - 'responses'        OpenAI-compat Responses API "
            f"(some Grok deployments)\n"
            f"  - 'azure-inference'  older Azure AI Inference path "
            f"(api-key header + ?api-version query param; common for Llama "
            f"on Azure AI Foundry)"
        )
    if value not in _VALID_API_MODES:
        raise ValueError(
            f"{env_var_name}={value!r} is not valid. "
            f"Must be one of: {list(_VALID_API_MODES)}"
        )
    return value  # type: ignore[return-value]
