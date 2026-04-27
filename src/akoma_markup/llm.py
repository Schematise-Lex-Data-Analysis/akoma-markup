"""Build a LangChain chat model from a config dict."""

import os

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_AZURE_MODEL = "Llama-3.3-70B-Instruct"


def build_llm(config: dict):
    """Instantiate a LangChain chat model from a provider config.

    Args:
        config: Dict with 'provider' (required) and provider-specific fields.
            Supported providers: 'anthropic', 'azure'.

            Common fields:
                provider: str — 'anthropic' or 'azure'
                model: str — model name (has sensible defaults per provider)
                temperature: float — defaults to 0
                max_tokens: int — defaults to 4096

            Provider-specific fields:
                anthropic: api_key (or ANTHROPIC_API_KEY env var)
                azure: endpoint, credential (or AZURE_INFERENCE_ENDPOINT,
                       AZURE_INFERENCE_CREDENTIAL env vars)

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ValueError: If provider is missing or unsupported, or required credentials
            are not provided.
    """
    config = dict(config)  # don't mutate the caller's dict
    provider = config.pop("provider", None)
    if not provider:
        raise ValueError("LLM config must include a 'provider' field")

    model = config.pop("model", None)
    temperature = config.pop("temperature", 0)
    max_tokens = config.pop("max_tokens", 4096)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "Install the anthropic extra: pip install akoma-markup[anthropic]"
            )
        api_key = config.pop("api_key", None) or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Provide api_key in config or set ANTHROPIC_API_KEY env var"
            )
        return ChatAnthropic(
            model=model or DEFAULT_ANTHROPIC_MODEL,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "azure":
        endpoint = config.pop("endpoint", None) or os.environ.get(
            "AZURE_INFERENCE_ENDPOINT"
        )
        credential = (
            config.pop("credential", None)
            or config.pop("api_key", None)
            or os.environ.get("AZURE_INFERENCE_CREDENTIAL")
        )
        if not endpoint or not credential:
            raise ValueError(
                "Provide endpoint and credential in config or set "
                "AZURE_INFERENCE_ENDPOINT and AZURE_INFERENCE_CREDENTIAL env vars"
            )

        deployment = model or DEFAULT_AZURE_MODEL

        # The deployment's API surface must be specified explicitly — different
        # Azure deployments expose different transports (chat/completions,
        # Responses API, or older Azure-Inference path) and we don't probe.
        from .azure_api import validate_api_mode

        api_style = (
            config.pop("api_style", None)
            or os.environ.get("AZURE_INFERENCE_API_STYLE")
        )
        api_mode = validate_api_mode(api_style, "AZURE_INFERENCE_API_STYLE")

        if api_mode == "azure-inference":
            try:
                from langchain_azure_ai.chat_models import (
                    AzureAIChatCompletionsModel,
                )
            except ImportError:
                raise ImportError(
                    "Install the azure extra: pip install akoma-markup[azure]"
                )
            return AzureAIChatCompletionsModel(
                endpoint=endpoint,
                credential=credential,
                model=deployment,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # OpenAI-compat path: chat/completions or Responses
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "Install the azure extra: pip install akoma-markup[azure]"
            )
        return ChatOpenAI(
            base_url=endpoint,
            api_key=credential,
            model=deployment,
            use_responses_api=(api_mode == "responses"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # add support for huggingface provider
    
    raise ValueError(
        f"Unknown provider '{provider}'. Supported: anthropic, azure"
    )
