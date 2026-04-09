"""Build a LangChain chat model from a config dict."""

import os


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
            model=model or "claude-sonnet-4-20250514",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
# hugggingface
    if provider == "azure":
        try:
            from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
        except ImportError:
            raise ImportError(
                "Install the azure extra: pip install akoma-markup[azure]"
            )
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
        return AzureAIChatCompletionsModel(
            endpoint=endpoint,
            credential=credential,
            model=model or "Llama-3.3-70B-Instruct",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(
        f"Unknown provider '{provider}'. Supported: anthropic, azure"
    )
