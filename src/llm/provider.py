"""
provider.py

Responsável por inicializar o LLM correto com base nas variáveis de ambiente.

Providers suportados (via LLM_PROVIDER):
  - groq
  - ollama
  - microsoft / azure / azure_openai
  - huggingface
  - aws
  - google
  - anthropic
  - openai
  - deepseek
  - grok

O agente NUNCA instala libs automaticamente:
se o pacote do provider não estiver instalado, este módulo lança um erro
explicando qual `pip install` você precisa rodar.
"""

import os
from typing import Any


class ProviderConfigError(RuntimeError):
    """Erro de configuração de provider (variável de ambiente faltando)."""


def _get_base_params() -> dict:
    """Lê parâmetros genéricos do modelo a partir do .env."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL_NAME")
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0.1"))
    max_tokens = int(os.getenv("MODEL_MAX_TOKENS", "8000"))
    return {
        "provider": provider,
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _require_env(var_name: str, provider_label: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise ProviderConfigError(
            f"[{provider_label}] Variável de ambiente obrigatória não encontrada: {var_name}"
        )
    return value


def _maybe_set_env(var_name: str, value: str | None) -> None:
    """Define uma env var se não existir ainda, usado para libs que esperam nomes específicos."""
    if value and not os.getenv(var_name):
        os.environ[var_name] = value


def get_llm() -> Any:
    """
    Retorna uma instância de chat model do LangChain de acordo com LLM_PROVIDER.

    Exemplo de uso:
        from provider import get_llm
        llm = get_llm()
        llm.invoke("Olá!")
    """
    cfg = _get_base_params()
    provider = cfg["provider"]
    api_key = cfg["api_key"]
    model = cfg["model"]
    temperature = cfg["temperature"]
    max_tokens = cfg["max_tokens"]

    # -------------------------
    # OpenAI
    # -------------------------
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Para usar 'openai', instale o pacote: pip install -U langchain-openai"
            ) from exc

        if not api_key:
            # permite também usar OPENAI_API_KEY diretamente no ambiente
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderConfigError(
                "[OpenAI] Defina LLM_API_KEY ou OPENAI_API_KEY no .env."
            )

        openai_base_url = os.getenv("OPENAI_BASE_URL")
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "model": model or "gpt-4.1-mini",
            "temperature": temperature,
        }
        if max_tokens > 0:
            kwargs["max_tokens"] = max_tokens
        if openai_base_url:
            kwargs["base_url"] = openai_base_url

        return ChatOpenAI(**kwargs)

    # -------------------------
    # Groq
    # -------------------------
    if provider == "groq":
        try:
            from langchain_groq import ChatGroq  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Para usar 'groq', instale o pacote: pip install -U langchain-groq"
            ) from exc

        if not api_key:
            # compatível com docs oficiais (GROQ_API_KEY)
            api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ProviderConfigError(
                "[Groq] Defina LLM_API_KEY ou GROQ_API_KEY no .env."
            )

        return ChatGroq(
            api_key=api_key,
            model=model or "llama-3.3-70b-versatile",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # -------------------------
    # Ollama (local)
    # -------------------------
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Para usar 'ollama', instale o pacote: pip install -U langchain-ollama"
            ) from exc

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        return ChatOllama(
            model=model or "llama3",
            temperature=temperature,
            base_url=base_url,
            # num_predict equivalente aproximado a max_tokens
            num_predict=max_tokens if max_tokens > 0 else None,
        )

    # -------------------------
    # Microsoft / Azure OpenAI
    # -------------------------
    if provider in {"microsoft", "azure", "azure_openai"}:
        try:
            from langchain_openai import AzureChatOpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Para usar 'microsoft/azure', instale: pip install -U langchain-openai"
            ) from exc

        azure_endpoint = _require_env("AZURE_ENDPOINT", "Azure")
        api_key = api_key or os.getenv("AZURE_API_KEY")
        if not api_key:
            raise ProviderConfigError(
                "[Azure] Defina LLM_API_KEY ou AZURE_API_KEY no .env."
            )
        api_version = _require_env("AZURE_API_VERSION", "Azure")
        deployment = os.getenv("AZURE_DEPLOYMENT") or model
        if not deployment:
            raise ProviderConfigError(
                "[Azure] Defina AZURE_DEPLOYMENT ou LLM_MODEL_NAME no .env."
            )

        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=deployment,
            temperature=temperature,
            max_tokens=max_tokens if max_tokens > 0 else None,
        )

    # -------------------------
    # Hugging Face (Inference / Endpoint)
    # -------------------------
    if provider == "huggingface":
        try:
            from langchain_huggingface import (
                ChatHuggingFace,
                HuggingFaceEndpoint,
            )  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Para usar 'huggingface', instale: pip install -U langchain-huggingface"
            ) from exc

        # Usa LLM_API_KEY como fallback para o token da HF
        if api_key:
            _maybe_set_env("HUGGINGFACEHUB_API_TOKEN", api_key)

        if not model:
            raise ProviderConfigError(
                "[HuggingFace] Defina LLM_MODEL_NAME com o repo_id do modelo "
                "(ex: meta-llama/Meta-Llama-3.3-70B-Instruct)."
            )

        endpoint_llm = HuggingFaceEndpoint(
            repo_id=model,
            task="text-generation",
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        return ChatHuggingFace(llm=endpoint_llm)

    # -------------------------
    # AWS Bedrock
    # -------------------------
    if provider == "aws":
        try:
            from langchain_aws import ChatBedrockConverse  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Para usar 'aws', instale: pip install -U langchain-aws boto3"
            ) from exc

        # modelo pode vir de AWS_BEDROCK_MODEL ou LLM_MODEL_NAME
        bedrock_model = os.getenv("AWS_BEDROCK_MODEL") or model
        if not bedrock_model:
            raise ProviderConfigError(
                "[AWS] Defina AWS_BEDROCK_MODEL ou LLM_MODEL_NAME no .env "
                "(ex: anthropic.claude-3-sonnet-20240229-v1:0)."
            )

        # credenciais normalmente vêm de AWS_ACCESS_KEY_ID / SECRET ou profile;
        # aqui apenas garantimos que, se estiverem no .env, o boto3 consiga ler.
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"):
            val = os.getenv(name)
            if val:
                os.environ[name] = val

        return ChatBedrockConverse(
            model=bedrock_model,
            temperature=temperature,
            max_tokens=max_tokens if max_tokens > 0 else None,
        )

    # -------------------------
    # Google (Gemini)
    # -------------------------
    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Para usar 'google', instale: pip install -U langchain-google-genai"
            ) from exc

        # Usa LLM_API_KEY como GOOGLE_API_KEY se necessário
        if api_key:
            _maybe_set_env("GOOGLE_API_KEY", api_key)

        if not api_key and not os.getenv("GOOGLE_API_KEY"):
            raise ProviderConfigError(
                "[Google] Defina LLM_API_KEY ou GOOGLE_API_KEY no .env."
            )

        return ChatGoogleGenerativeAI(
            model=model or "gemini-1.5-flash",
            temperature=temperature,
            max_output_tokens=max_tokens if max_tokens > 0 else None,
        )

    # -------------------------
    # Anthropic (Claude)
    # -------------------------
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Para usar 'anthropic', instale: pip install -U langchain-anthropic"
            ) from exc

        if not api_key:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderConfigError(
                "[Anthropic] Defina LLM_API_KEY ou ANTHROPIC_API_KEY no .env."
            )

        return ChatAnthropic(
            api_key=api_key,
            model=model or "claude-3-5-sonnet",
            temperature=temperature,
            max_tokens=max_tokens if max_tokens > 0 else None,
        )

    # -------------------------
    # DeepSeek
    # -------------------------
    if provider == "deepseek":
        try:
            from langchain_deepseek import ChatDeepSeek  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Para usar 'deepseek', instale: pip install -U langchain-deepseek"
            ) from exc

        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ProviderConfigError(
                "[DeepSeek] Defina LLM_API_KEY ou DEEPSEEK_API_KEY no .env."
            )

        return ChatDeepSeek(
            api_key=api_key,
            model=model or "deepseek-chat",
            temperature=temperature,
            max_tokens=max_tokens if max_tokens > 0 else None,
        )

    # -------------------------
    # Grok (xAI)
    # -------------------------
    if provider == "grok":
        try:
            from langchain_xai import ChatXAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Para usar 'grok', instale: pip install -U langchain-xai"
            ) from exc

        if not api_key:
            api_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
        if not api_key:
            raise ProviderConfigError(
                "[Grok/xAI] Defina LLM_API_KEY ou GROK_API_KEY ou XAI_API_KEY no .env."
            )

        return ChatXAI(
            api_key=api_key,
            model=model or "grok-2-latest",
            temperature=temperature,
            max_tokens=max_tokens if max_tokens > 0 else None,
        )

    # -------------------------
    # Provider desconhecido
    # -------------------------
    raise ProviderConfigError(
        f"LLM_PROVIDER='{provider}' não é suportado neste provider.py."
    )
