"""
LLM Provider Factory — builds LangChain LLM objects for any supported provider.

Supported providers
-------------------
  openai      → ChatOpenAI
  google      → ChatGoogleGenerativeAI
  anthropic   → ChatAnthropic
  groq        → ChatOpenAI (Groq-compatible base_url)
  nvidia      → ChatNVIDIA

Model catalogues are also exposed so the frontend can populate dropdowns.
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model catalogues (displayed in the frontend dropdown)
# ---------------------------------------------------------------------------
PROVIDER_CATALOGUE: Dict[str, Dict[str, Any]] = {
    "openai": {
        "display_name": "OpenAI",
        "default_main": "gpt-5.2",
        "default_fast": "gpt-5-mini",
        "models": [
            {"id": "gpt-5.2",              "label": "GPT-5.2 (Recommended)"},
            {"id": "gpt-5.2-pro",          "label": "GPT-5.2 Pro"},
            {"id": "gpt-5",                "label": "GPT-5"},
            {"id": "gpt-5-mini",           "label": "GPT-5 Mini (Recommended Fast)"},
            {"id": "gpt-5-nano",           "label": "GPT-5 Nano"},
            {"id": "o3",                   "label": "o3 (Reasoning)"},
            {"id": "o3-pro",               "label": "o3 Pro"},
            {"id": "gpt-4.1",              "label": "GPT-4.1"},
            {"id": "gpt-4o",               "label": "GPT-4o"},
            {"id": "gpt-4o-mini",          "label": "GPT-4o Mini"},
            {"id": "o3-mini",              "label": "o3 Mini"},
            {"id": "gpt-4-turbo",          "label": "GPT-4 Turbo"},
            {"id": "gpt-3.5-turbo",        "label": "GPT-3.5 Turbo"}
        ]
    },
    "google": {
        "display_name": "Google Gemini",
        "default_main": "gemini-3.1-pro-preview",
        "default_fast": "gemini-2.5-flash",
        "models": [
            {"id": "gemini-3.1-pro-preview",         "label": "Gemini 3.1 Pro Preview (Latest Recommended)"},
            {"id": "gemini-2.5-pro",                "label": "Gemini 2.5 Pro"},
            {"id": "gemini-2.5-flash",              "label": "Gemini 2.5 Flash (Recommended Fast)"},
            {"id": "gemini-3.1-flash-lite-preview", "label": "Gemini 3.1 Flash-Lite Preview"},
            {"id": "gemini-2.0-flash",              "label": "Gemini 2.0 Flash"},
            {"id": "gemini-1.5-pro",                "label": "Gemini 1.5 Pro"},
            {"id": "gemini-1.5-flash",              "label": "Gemini 1.5 Flash"}
        ]
    },
    "anthropic": {
        "display_name": "Anthropic Claude",
        "default_main": "claude-sonnet-4-6",
        "default_fast": "claude-haiku-4-5-20251001",
        "models": [
            {"id": "claude-opus-4-6",           "label": "Claude Opus 4.6"},
            {"id": "claude-sonnet-4-6",          "label": "Claude Sonnet 4.6 (Recommended)"},
            {"id": "claude-haiku-4-5-20251001",  "label": "Claude Haiku 4.5"},
        ],
    },
    "groq": {
        "display_name": "Groq",
        "default_main": "llama-3.3-70b-versatile",
        "default_fast": "openai/gpt-oss-120b",
        "models": [
            {"id": "llama-3.3-70b-versatile",  "label": "Llama 3.3 70B (Recommended)"},
            {"id": "qwen/qwen3-32b",     "label": "Qwen 3 32B"},
            {"id": "openai/gpt-oss-120b",       "label": "GPT OSS 120B"},
            {"id": "llama-3.1-8b-instant",             "label": "llama 3.1 8B Instant"},
            {"id": "openai/gpt-oss-20b", "label": "GPT OSS 20B"},
        ],
    },
    "nvidia": {
        "display_name": "NVIDIA NIM",
        "default_main": "qwen/qwen3-next-80b-a3b-instruct",
        "default_fast": "meta/llama-3.3-70b-instruct",
        "models": [
            {"id": "qwen/qwen3-next-80b-a3b-instruct",  "label": "Qwen3 Next 80B (Recommended)"},
            {"id": "meta/llama-3.3-70b-instruct",       "label": "Llama 3.3 70B"},
            {"id": "google/gemma-4-31b-it",             "label": "Gemma 4 31B"},
            {"id": "mistralai/mistral-large-2-instruct","label": "Mistral Large 2"},
            {"id": "meta/llama-4-maverick-70b-instruct", "label": "Llama 4 Maverick 70B"},
            {"id": "deepseek/deepseek-v3-2-236b-instruct","label": "DeepSeek V3.2 236B"},
            {"id": "qwen/qwen3-32b-instruct",           "label": "Qwen3 32B"},
            {"id": "mistralai/mixtral-8x22b-instruct-v2","label": "Mixtral 8x22B v2"},
            {"id": "microsoft/phi-4-14b-instruct",      "label": "Phi-4 14B"},
            {"id": "google/gemma-3-27b-it",             "label": "Gemma 3 27B"},
            {"id": "cohere/command-rplus-104b-instruct", "label": "Command R+ 104B"},
            {"id": "meta/llama-4-405b-instruct",        "label": "Llama 4 405B"},
            {"id": "qwen/qwen3-next-235b-a22b-instruct","label": "Qwen3 Next 235B"},
            {"id": "deepseek/deepseek-coder-v3-130b",   "label": "DeepSeek Coder V3 130B"}
        ]
    },
    "huggingface": {
        "display_name": "HuggingFace",
        "default_main": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "default_fast": "Qwen/Qwen3.5-9B",
        "models": [
            {"id": "Qwen/Qwen2.5-72B-Instruct",                "label": "Qwen 2.5 72B Instruct (Recommended)"},
            {"id": "meta-llama/Meta-Llama-3.1-70B-Instruct",   "label": "Llama 3.1 70B Instruct"},
            {"id": "meta-llama/Meta-Llama-3.1-8B-Instruct",    "label": "Llama 3.1 8B Instruct (Fast)"},
            {"id": "mistralai/Mistral-7B-Instruct-v0.3",        "label": "Mistral 7B Instruct v0.3"},
            {"id": "mistralai/Mixtral-8x7B-Instruct-v0.1",      "label": "Mixtral 8x7B Instruct"},
            {"id": "microsoft/Phi-3.5-mini-instruct",           "label": "Phi-3.5 Mini Instruct"},
            {"id": "google/gemma-2-27b-it",                     "label": "Gemma 2 27B IT"},
            {"id": "google/gemma-2-9b-it",                      "label": "Gemma 2 9B IT"},
            {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  "label": "DeepSeek R1 Distill 32B"},
            {"id": "Qwen/Qwen2.5-Coder-32B-Instruct",           "label": "Qwen 2.5 Coder 32B"},
            {"id": "zai-org/GLM-5.1",                           "label": "GLM-5.1 (Top Trending)"},
            {"id": "google/gemma-4-31B-it",                     "label": "Gemma 4 31B IT"},
            {"id": "google/gemma-4-26B-A4B-it",                 "label": "Gemma 4 26B A4B IT"},
            {"id": "Qwen/Qwen3.5-9B",                           "label": "Qwen3.5 9B (Fast Analysis)"},
            {"id": "zai-org/GLM-5.1-FP8",                       "label": "GLM-5.1 FP8"},
            {"id": "Qwen/Qwen3.5-27B",                          "label": "Qwen3.5 27B"},
            {"id": "zai-org/GLM-5",                             "label": "GLM-5"},
            {"id": "Qwen/Qwen3.5-35B-A3B",                      "label": "Qwen3.5 35B A3B"},
            {"id": "openai/gpt-oss-120b",                       "label": "GPT-OSS 120B"},
            {"id": "moonshotai/Kimi-K2.5",                      "label": "Kimi K2.5"},
            {"id": "MiniMaxAI/MiniMax-M2.5",                    "label": "MiniMax M2.5"},
            {"id": "Qwen/Qwen3-Coder-Next",                     "label": "Qwen3 Coder Next (SQL)"},
            {"id": "deepseek-ai/DeepSeek-R1",                   "label": "DeepSeek R1"},
            {"id": "Qwen/Qwen3.5-397B-A17B",                    "label": "Qwen3.5 397B A17B"},
            {"id": "openai/gpt-oss-20b",                        "label": "GPT-OSS 20B"},
            {"id": "deepseek-ai/DeepSeek-V3.2",                 "label": "DeepSeek V3.2"},
            {"id": "Qwen/Qwen3.5-122B-A10B",                    "label": "Qwen3.5 122B A10B"},
            {"id": "meta-llama/Llama-3.3-70B-Instruct",         "label": "Llama 3.3 70B Instruct"},
            {"id": "deepseek-ai/DeepSeek-V3",                   "label": "DeepSeek V3"},
            {"id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",         "label": "Qwen3 Coder 30B A3B"},
            {"id": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "label": "Llama 4 Scout 17B"},
            {"id": "Qwen/Qwen3-32B",                            "label": "Qwen3 32B"},
            {"id": "Qwen/Qwen3-235B-A22B-Instruct-2507",        "label": "Qwen3 235B A22B (Premier)"},
            {"id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "label": "Llama 4 Maverick 17B FP8"},
            {"id": "Qwen/Qwen3-Coder-480B-A35B-Instruct",       "label": "Qwen3 Coder 480B A35B"}
        ]
    },
}


def get_provider_models(provider: str) -> List[Dict[str, str]]:
    """Return the model list for a provider (for the frontend dropdown)."""
    catalogue = PROVIDER_CATALOGUE.get(provider, {})
    return catalogue.get("models", [])


def get_defaults(provider: str) -> Tuple[str, str]:
    """Return (default_main, default_fast) model IDs for a provider."""
    cat = PROVIDER_CATALOGUE.get(provider, {})
    return cat.get("default_main", ""), cat.get("default_fast", "")


# ---------------------------------------------------------------------------
# LLM construction
# ---------------------------------------------------------------------------
def build_llm_pair(provider: str, api_key: str,
                   main_model: str, fast_model: str):
    """
    Build (_main_llm, _fast_llm) LangChain objects for the given provider.

    Raises ValueError on unknown provider.
    Raises RuntimeError if the required langchain package is missing.
    """
    kwargs_common = {"temperature": 0, "max_tokens": 7000}

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise RuntimeError("uv pip install langchain-openai")
        main = ChatOpenAI(model=main_model, api_key=api_key, **kwargs_common)
        fast = ChatOpenAI(model=fast_model, api_key=api_key, **kwargs_common)

    elif provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise RuntimeError("uv pip install langchain-google-genai")
        main = ChatGoogleGenerativeAI(model=main_model, google_api_key=api_key,
                                      temperature=0)
        fast = ChatGoogleGenerativeAI(model=fast_model, google_api_key=api_key,
                                      temperature=0)

    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise RuntimeError("uv pip install langchain-anthropic")
        main = ChatAnthropic(model=main_model, api_key=api_key,
                             temperature=0, max_tokens=8000)
        fast = ChatAnthropic(model=fast_model, api_key=api_key,
                             temperature=0, max_tokens=8000)

    elif provider == "groq":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise RuntimeError("uv pip install langchain-openai")  # groq uses openai-compat
        groq_url = "https://api.groq.com/openai/v1"
        main = ChatOpenAI(model=main_model, api_key=api_key,
                          base_url=groq_url, **kwargs_common)
        fast = ChatOpenAI(model=fast_model, api_key=api_key,
                          base_url=groq_url, **kwargs_common)

    elif provider == "nvidia":
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError:
            raise RuntimeError("uv pip install langchain-nvidia-ai-endpoints")
        main = ChatNVIDIA(model=main_model, api_key=api_key,
                          temperature=0, max_tokens=8000)
        fast = ChatNVIDIA(model=fast_model, api_key=api_key,
                          temperature=0, max_tokens=8000)

    elif provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        except ImportError:
            raise RuntimeError("uv pip install langchain-huggingface huggingface_hub")

        def _hf_llm(model_id: str):
            endpoint = HuggingFaceEndpoint(
                repo_id=model_id,
                huggingfacehub_api_token=api_key,
                temperature=0.01,   # some HF models reject temperature=0
                max_new_tokens=4096,
                task="text-generation",
            )
            return ChatHuggingFace(llm=endpoint)

        main = _hf_llm(main_model)
        fast = _hf_llm(fast_model)

    else:
        raise ValueError(f"Unknown provider: {provider!r}. "
                         f"Choose from: {list(PROVIDER_CATALOGUE)}")

    logger.info("Built LLM pair: provider=%s main=%s fast=%s", provider, main_model, fast_model)
    return main, fast


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------
def validate_api_key(provider: str, api_key: str,
                     main_model: str, fast_model: str) -> Tuple[bool, str]:
    """
    Actually invoke the LLM with a tiny prompt to verify the key works.
    Returns (is_valid, error_message).
    """
    try:
        main_llm, _ = build_llm_pair(provider, api_key, main_model, fast_model)
        from langchain_core.messages import HumanMessage
        resp = main_llm.invoke([HumanMessage(content="Reply with the single word: OK")])
        text = resp.content.strip().lower() if hasattr(resp, "content") else str(resp)
        logger.info("Key validation passed: %s", text[:40])
        return True, ""
    except Exception as exc:
        msg = str(exc)
        # Extract a clean error (hide full stack from client)
        if "401" in msg or "authentication" in msg.lower() or "api key" in msg.lower():
            clean = "Invalid API key — authentication failed."
        elif "429" in msg or "rate" in msg.lower():
            clean = "Rate limit hit — please wait a moment and try again."
        elif "404" in msg or "model" in msg.lower():
            clean = f"Model not found: {main_model}. Check the model ID."
        else:
            clean = f"Connection error: {msg[:200]}"
        logger.warning("Key validation failed: %s", msg[:200])
        return False, clean
