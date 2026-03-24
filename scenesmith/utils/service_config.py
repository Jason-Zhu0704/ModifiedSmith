"""Centralized service endpoint/credential resolution."""

from __future__ import annotations

import os

from typing import Any

from openai import AsyncOpenAI, OpenAI


def _safe_get(mapping: Any, *keys: str, default: Any = None) -> Any:
    current = mapping
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return default if current is None else current


def resolve_openai_connection(
    service_cfg: Any | None,
    section: str,
) -> dict[str, str | None]:
    """Resolve OpenAI endpoint and key for a service section."""
    section_cfg = _safe_get(service_cfg, "providers", section, default={}) or {}

    api_key_env = section_cfg.get("openai_api_key_env", "OPENAI_API_KEY")
    base_url_env = section_cfg.get("openai_base_url_env", "OPENAI_BASE_URL")
    base_url_default = section_cfg.get(
        "openai_base_url_default", "https://api.openai.com/v1"
    )

    api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get(base_url_env) or os.environ.get(
        "OPENAI_BASE_URL", base_url_default
    )

    return {
        "api_key": api_key,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "base_url_env": base_url_env,
    }


def resolve_gemini_api_key(service_cfg: Any | None, section: str) -> tuple[str | None, str]:
    """Resolve Gemini API key and source env var for a service section."""
    section_cfg = _safe_get(service_cfg, "providers", section, default={}) or {}
    api_key_env = section_cfg.get("gemini_api_key_env", "GOOGLE_API_KEY")
    api_key = os.environ.get(api_key_env) or os.environ.get("GOOGLE_API_KEY")
    return api_key, api_key_env


def resolve_gemini_connection(
    service_cfg: Any | None, section: str
) -> dict[str, str | None]:
    """Resolve Gemini endpoint/model/key for a service section."""
    section_cfg = _safe_get(service_cfg, "providers", section, default={}) or {}

    api_key_env = section_cfg.get("gemini_api_key_env", "GOOGLE_API_KEY")
    base_url_env = section_cfg.get("gemini_base_url_env", "SCENESMITH_GEMINI_BASE_URL")
    model_env = section_cfg.get("gemini_model_env", "SCENESMITH_GEMINI_IMAGE_MODEL")

    base_url_default = section_cfg.get(
        "gemini_base_url_default", "https://generativelanguage.googleapis.com/v1beta"
    )
    model_default = section_cfg.get(
        "gemini_model_default", "gemini-3-pro-image-preview"
    )

    api_key = os.environ.get(api_key_env) or os.environ.get("GOOGLE_API_KEY")
    base_url = os.environ.get(base_url_env, base_url_default)
    model = os.environ.get(model_env, model_default)

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "api_key_env": api_key_env,
        "base_url_env": base_url_env,
        "model_env": model_env,
    }


def build_openai_client(service_cfg: Any | None, section: str) -> OpenAI:
    conn = resolve_openai_connection(service_cfg=service_cfg, section=section)
    kwargs: dict[str, str] = {}
    if conn["api_key"]:
        kwargs["api_key"] = str(conn["api_key"])
    if conn["base_url"]:
        kwargs["base_url"] = str(conn["base_url"])
    return OpenAI(**kwargs)


def build_async_openai_client(service_cfg: Any | None, section: str) -> AsyncOpenAI:
    conn = resolve_openai_connection(service_cfg=service_cfg, section=section)
    kwargs: dict[str, str] = {}
    if conn["api_key"]:
        kwargs["api_key"] = str(conn["api_key"])
    if conn["base_url"]:
        kwargs["base_url"] = str(conn["base_url"])
    return AsyncOpenAI(**kwargs)
