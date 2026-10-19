from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from config import MODELS, Settings

logger = logging.getLogger(__name__)


class OpenAIServiceError(Exception):
    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class ChatResult:
    text: str
    requested_model: str
    model: str
    temperature: float
    max_tokens: int
    token_param: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_rub: float
    request_id: str | None
    finish_reason: str | None
    elapsed: float
    fallback_used: bool = False


def _extract_error_message(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if body.get("message"):
            return str(body["message"])
    return str(exc) or exc.__class__.__name__


class OpenAIClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout,
            max_retries=2,
        )

    async def ask(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        try:
            return await self._request(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                fallback=False,
            )
        except OpenAIServiceError as exc:
            fallback = self._settings.fallback_model
            if not exc.retryable or not fallback or fallback == model:
                raise
            logger.warning(
                "Основная модель %s недоступна (%s). Пробую резервную %s",
                model,
                exc,
                fallback,
            )
            try:
                return await self._request(
                    model=fallback,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    fallback=True,
                )
            except OpenAIServiceError as fallback_exc:
                raise OpenAIServiceError(
                    f"Основная модель {model} и резервная {fallback} недоступны. "
                    f"Последняя ошибка: {fallback_exc}"
                ) from fallback_exc

    async def _request(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        fallback: bool,
    ) -> ChatResult:
        spec = MODELS[model]
        actual_temperature = temperature if spec.supports_temperature else 1.0
        params: dict[str, object] = {
            "model": spec.id,
            "messages": messages,
            spec.token_param: max_tokens,
        }
        if spec.supports_temperature:
            params["temperature"] = temperature
        if spec.reasoning_effort:
            params["reasoning_effort"] = spec.reasoning_effort

        logger.info(
            "Запрос к API: model=%s temperature=%s %s=%s messages=%d fallback=%s",
            spec.id,
            actual_temperature,
            spec.token_param,
            max_tokens,
            len(messages),
            fallback,
        )
        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(**params)
        except AuthenticationError as exc:
            logger.exception("Ошибка авторизации API")
            raise OpenAIServiceError(
                "Ошибка авторизации API. Проверьте OPENAI_API_KEY в .env",
                retryable=False,
            ) from exc
        except RateLimitError as exc:
            logger.exception("Превышен лимит запросов API")
            raise OpenAIServiceError(
                "Превышен лимит запросов API. Попробуйте позже"
            ) from exc
        except APITimeoutError as exc:
            logger.exception("Таймаут API")
            raise OpenAIServiceError("API не ответил вовремя. Попробуйте ещё раз") from exc
        except APIConnectionError as exc:
            logger.exception("Нет соединения с API")
            raise OpenAIServiceError(
                "Не удалось соединиться с API. Проверьте интернет"
            ) from exc
        except BadRequestError as exc:
            logger.exception("API отклонил параметры запроса")
            detail = _extract_error_message(exc)
            raise OpenAIServiceError(
                f"API отклонил параметры запроса: {detail}"
            ) from exc
        except APIError as exc:
            logger.exception("Ошибка API")
            raise OpenAIServiceError(f"Ошибка API: {_extract_error_message(exc)}") from exc

        elapsed = time.perf_counter() - started
        choice = response.choices[0] if response.choices else None
        text = (choice.message.content or "").strip() if choice else ""
        usage = response.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        cost_rub = (
            prompt_tokens / 1_000_000 * spec.input_price
            + completion_tokens / 1_000_000 * spec.output_price
        )
        request_id = getattr(response, "id", None)
        finish_reason = getattr(choice, "finish_reason", None) if choice else None

        logger.info(
            "Ответ API: model=%s request_id=%s tokens=%d/%d/%d cost=%.4f RUB "
            "finish=%s elapsed=%.2fs fallback=%s",
            spec.id,
            request_id,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost_rub,
            finish_reason,
            elapsed,
            fallback,
        )

        return ChatResult(
            text=text,
            requested_model=model,
            model=spec.id,
            temperature=actual_temperature,
            max_tokens=max_tokens,
            token_param=spec.token_param,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_rub=cost_rub,
            request_id=request_id,
            finish_reason=finish_reason,
            elapsed=elapsed,
            fallback_used=fallback,
        )

    async def close(self) -> None:
        await self._client.close()
