from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    title: str
    input_price: float
    output_price: float
    supports_temperature: bool = True
    token_param: str = "max_tokens"
    reasoning_effort: str | None = None
    notes: str = ""


MODELS: dict[str, ModelSpec] = {
    "openai/gpt-4.1-mini": ModelSpec(
        id="openai/gpt-4.1-mini",
        title="GPT-4.1 mini",
        input_price=104.0,
        output_price=413.0,
    ),
    "openai/gpt-4.1-nano": ModelSpec(
        id="openai/gpt-4.1-nano",
        title="GPT-4.1 nano",
        input_price=26.0,
        output_price=104.0,
    ),
    "openai/gpt-4o-mini": ModelSpec(
        id="openai/gpt-4o-mini",
        title="GPT-4o mini",
        input_price=39.0,
        output_price=155.0,
    ),
    "deepseek/deepseek-chat": ModelSpec(
        id="deepseek/deepseek-chat",
        title="DeepSeek V3",
        input_price=43.16,
        output_price=126.32,
    ),
    "openai/gpt-4.1": ModelSpec(
        id="openai/gpt-4.1",
        title="GPT-4.1",
        input_price=516.0,
        output_price=2062.0,
    ),
    "openai/gpt-3.5-turbo": ModelSpec(
        id="openai/gpt-3.5-turbo",
        title="GPT-3.5 Turbo",
        input_price=129.0,
        output_price=387.0,
        notes="Модель отключается 23.10.2026",
    ),
    "openai/gpt-5-mini": ModelSpec(
        id="openai/gpt-5-mini",
        title="GPT-5 mini",
        input_price=65.0,
        output_price=516.0,
        supports_temperature=False,
        token_param="max_completion_tokens",
        reasoning_effort="minimal",
        notes="Temperature только 1.0, лимит через max_completion_tokens",
    ),
}


@dataclass(frozen=True)
class Settings:
    bot_token: str
    telegram_proxy: str | None
    openai_api_key: str
    openai_base_url: str
    default_model: str
    fallback_model: str | None
    default_temperature: float
    default_max_tokens: int
    history_limit: int
    request_timeout: float
    system_prompt: str
    log_file: Path
    max_user_message_length: int = 4000


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Не задана переменная окружения {name}. Проверьте файл .env")
    return value


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Переменная {name} должна быть числом") from exc


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Переменная {name} должна быть целым числом") from exc


def load_settings() -> Settings:
    default_model = os.getenv("DEFAULT_MODEL", "openai/gpt-4.1-mini").strip()
    if default_model not in MODELS:
        raise RuntimeError(
            f"Модель {default_model} отсутствует в реестре config.MODELS. "
            f"Доступны: {', '.join(MODELS)}"
        )

    fallback_raw = os.getenv("FALLBACK_MODEL", "deepseek/deepseek-chat").strip()
    fallback_model = fallback_raw or None
    if fallback_model is not None and fallback_model not in MODELS:
        raise RuntimeError(f"Модель {fallback_model} отсутствует в реестре config.MODELS")
    if fallback_model == default_model:
        fallback_model = None

    temperature = _float("DEFAULT_TEMPERATURE", 0.7)
    if not 0.0 <= temperature <= 2.0:
        raise RuntimeError("DEFAULT_TEMPERATURE должна быть в диапазоне 0..2")

    max_tokens = _int("DEFAULT_MAX_TOKENS", 600)
    if max_tokens < 16:
        raise RuntimeError("DEFAULT_MAX_TOKENS должна быть не меньше 16")

    history_limit = _int("HISTORY_LIMIT", 20)
    history_limit -= history_limit % 2
    if history_limit < 2:
        history_limit = 2

    request_timeout = _float("REQUEST_TIMEOUT", 60.0)
    if request_timeout <= 0:
        raise RuntimeError("REQUEST_TIMEOUT должен быть больше нуля")

    system_prompt = os.getenv(
        "SYSTEM_PROMPT",
        "Ты дружелюбный и полезный ассистент. Отвечай на языке пользователя, кратко и по делу.",
    ).strip()

    return Settings(
        bot_token=_required("BOT_TOKEN"),
        telegram_proxy=os.getenv("TELEGRAM_PROXY", "").strip() or None,
        openai_api_key=_required("OPENAI_API_KEY"),
        openai_base_url=os.getenv(
            "OPENAI_BASE_URL", "https://api.proxyapi.ru/v1"
        ).strip(),
        default_model=default_model,
        fallback_model=fallback_model,
        default_temperature=temperature,
        default_max_tokens=max_tokens,
        history_limit=history_limit,
        request_timeout=request_timeout,
        system_prompt=system_prompt,
        log_file=BASE_DIR / os.getenv("LOG_FILE", "bot.log"),
    )


settings = load_settings()
