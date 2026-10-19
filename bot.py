from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import ErrorEvent, Message

from config import MODELS, settings
from context_manager import ContextManager
from openai_client import ChatResult, OpenAIClient, OpenAIServiceError

logger = logging.getLogger(__name__)

telegram_session = (
    AiohttpSession(proxy=settings.telegram_proxy)
    if settings.telegram_proxy
    else None
)
bot = Bot(
    token=settings.bot_token,
    default=DefaultBotProperties(parse_mode=None),
    session=telegram_session,
)
dp = Dispatcher()
context = ContextManager(
    history_limit=settings.history_limit,
    system_prompt=settings.system_prompt,
    default_model=settings.default_model,
    default_temperature=settings.default_temperature,
    default_max_tokens=settings.default_max_tokens,
)
client = OpenAIClient(settings)

RESET_PHRASES = {
    "очистить контекст",
    "очисти контекст",
    "сбросить контекст",
    "сброс контекста",
    "очистить историю",
    "очисти историю",
}


def setup_logging() -> None:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logging.getLogger("aiogram.event").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def resolve_model(value: str) -> str | None:
    value = value.strip()
    items = list(MODELS.items())
    if value.isdigit():
        index = int(value)
        if 1 <= index <= len(items):
            return items[index - 1][0]
        return None

    lowered = value.casefold()
    if lowered in MODELS:
        return lowered

    for model_id, spec in items:
        if model_id.split("/", 1)[-1].casefold() == lowered:
            return model_id
        if spec.title.casefold() == lowered:
            return model_id

    matches = [
        model_id
        for model_id, spec in items
        if lowered in model_id.casefold() or lowered in spec.title.casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def models_text(current_model: str) -> str:
    lines = ["Доступные модели:", ""]
    for index, spec in enumerate(MODELS.values(), start=1):
        marker = "  <-- текущая" if spec.id == current_model else ""
        temperature = (
            "temperature: да" if spec.supports_temperature else "temperature: только 1.0"
        )
        lines.append(
            f"{index}. {spec.id} — {spec.input_price:g}/{spec.output_price:g} ₽ "
            f"за 1M токенов, {temperature}, {spec.token_param}{marker}"
        )
        if spec.notes:
            lines.append(f"    {spec.notes}")
    lines.append("")
    lines.append("Переключение: /model 2 или /model deepseek/deepseek-chat")
    return "\n".join(lines)


def format_footer(result: ChatResult) -> str:
    parts = [
        f"Модель: {result.model}",
        f"temperature: {result.temperature:g}",
        f"{result.token_param}: {result.max_tokens}",
        (
            f"токены: {result.prompt_tokens} + {result.completion_tokens} "
            f"= {result.total_tokens}"
        ),
        f"~{result.cost_rub:.5f} ₽",
    ]
    if result.fallback_used:
        parts.append(f"fallback: {result.requested_model} -> {result.model}")
    return "\n\n— " + " · ".join(parts)


def split_text(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit)
        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at < limit // 2:
            split_at = limit
        chunk = remaining[:split_at].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].strip()
    return chunks


async def send_long(message: Message, text: str) -> None:
    for chunk in split_text(text):
        await message.answer(chunk)


async def reset_context(message: Message) -> None:
    cleared = context.clear(message.from_user.id)
    if cleared:
        logger.info("Контекст очищен: user_id=%s", message.from_user.id)
    else:
        logger.info("Попытка очистить пустой контекст: user_id=%s", message.from_user.id)
    await message.answer("Контекст очищен." if cleared else "Контекст уже пуст.")


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    user = context.get_settings(message.from_user.id)
    await message.answer(
        "Привет! Я Telegram-бот с памятью диалога и доступом к моделям через ProxyAPI.\n\n"
        f"Текущая модель: {user.model}\n"
        f"temperature: {user.temperature:g}\n"
        f"{MODELS[user.model].token_param}: {user.max_tokens}\n\n"
        "Команды:\n"
        "/reset — очистить контекст\n"
        "/model — список моделей и переключение\n"
        "/temp 0.7 — задать temperature (0..2)\n"
        "/tokens 600 — задать лимит ответа\n"
        "/status — текущие настройки\n"
        "/help — подробная помощь\n\n"
        "Просто напиши сообщение — я отвечу с учётом последних реплик."
    )


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "Как пользоваться ботом:\n\n"
        "1. Отправь обычный текст — бот ответит и запомнит пару сообщений.\n"
        "2. /reset или фраза «очистить контекст» — удалить историю диалога.\n"
        "3. /model — посмотреть модели; /model 1 — выбрать по номеру; "
        "/model deepseek/deepseek-chat — выбрать по имени.\n"
        "4. /temp 0.2 — низкая температура: ответы стабильнее и суше; "
        "/temp 1.3 — выше креативность и разнообразие.\n"
        "5. /tokens 300 — ограничить длину ответа.\n"
        "6. /status — показать модель, температуру, лимит и размер контекста.\n\n"
        f"История хранится в памяти и ограничена {context.history_limit} сообщениями "
        "(последние пары user/assistant).\n"
        f"Основная модель: {settings.default_model}\n"
        f"Резервная модель: {settings.fallback_model or 'не задана'}"
    )


@dp.message(Command("reset"))
async def cmd_reset(message: Message) -> None:
    await reset_context(message)


@dp.message(Command("model"))
async def cmd_model(message: Message, command: CommandObject) -> None:
    user_id = message.from_user.id
    user = context.get_settings(user_id)
    argument = (command.args or "").strip()

    if not argument:
        await message.answer(models_text(user.model))
        return

    resolved = resolve_model(argument)
    if resolved is None:
        await message.answer(
            f"Модель «{argument}» не найдена.\n\n" + models_text(user.model)
        )
        return

    user = context.set_model(user_id, resolved)
    spec = MODELS[resolved]
    logger.info("Смена модели: user_id=%s model=%s", user_id, spec.id)
    lines = [f"Модель переключена: {spec.id}"]
    if spec.notes:
        lines.append(spec.notes)
    if not spec.supports_temperature:
        lines.append(
            "Для этой модели temperature фиксирована на 1.0, "
            "а лимит передаётся как max_completion_tokens."
        )
    lines.append(f"Текущая temperature: {user.temperature:g}")
    await message.answer("\n".join(lines))


@dp.message(Command("temp"))
async def cmd_temp(message: Message, command: CommandObject) -> None:
    user_id = message.from_user.id
    user = context.get_settings(user_id)
    spec = MODELS[user.model]
    argument = (command.args or "").strip().replace(",", ".")

    if not argument:
        suffix = "" if spec.supports_temperature else " (API использует 1.0)"
        await message.answer(
            f"Текущая temperature: {user.temperature:g}{suffix}\n"
            "Задать: /temp 0.2 (диапазон 0..2)"
        )
        return

    try:
        value = float(argument)
    except ValueError:
        await message.answer("Temperature должна быть числом, например /temp 0.7")
        return

    if not 0.0 <= value <= 2.0:
        await message.answer("Temperature должна быть в диапазоне от 0 до 2.")
        return

    if not spec.supports_temperature:
        await message.answer(
            f"Модель {spec.id} не поддерживает temperature (только 1.0). "
            "Переключи модель через /model."
        )
        return

    context.set_temperature(user_id, value)
    logger.info("Смена temperature: user_id=%s value=%s", user_id, value)
    await message.answer(f"Temperature установлена: {value:g}")


@dp.message(Command("tokens"))
async def cmd_tokens(message: Message, command: CommandObject) -> None:
    user_id = message.from_user.id
    user = context.get_settings(user_id)
    spec = MODELS[user.model]
    argument = (command.args or "").strip()

    if not argument:
        await message.answer(
            f"Текущий лимит ({spec.token_param}): {user.max_tokens}\n"
            "Задать: /tokens 600 (диапазон 16..8000)"
        )
        return

    try:
        value = int(argument)
    except ValueError:
        await message.answer("Лимит должен быть целым числом, например /tokens 600")
        return

    if not 16 <= value <= 8000:
        await message.answer("Лимит должен быть в диапазоне от 16 до 8000.")
        return

    context.set_max_tokens(user_id, value)
    logger.info("Смена лимита: user_id=%s value=%s", user_id, value)
    suffix = ""
    if spec.token_param == "max_completion_tokens":
        suffix = " Лимит включает reasoning-токены модели."
    await message.answer(f"Лимит установлен: {value} ({spec.token_param}).{suffix}")


@dp.message(Command("status"))
async def cmd_status(message: Message) -> None:
    user_id = message.from_user.id
    user = context.get_settings(user_id)
    spec = MODELS[user.model]
    await message.answer(
        "Текущие настройки:\n"
        f"Модель: {spec.id}\n"
        f"Резервная: {settings.fallback_model or 'не задана'}\n"
        f"Temperature: {user.temperature:g}"
        + (" (API использует 1.0)" if not spec.supports_temperature else "")
        + "\n"
        f"Лимит ответа: {user.max_tokens} ({spec.token_param})\n"
        f"Контекст: {context.history_size(user_id)} из {context.history_limit} сообщений\n"
        f"API: {settings.openai_base_url}"
    )


@dp.message(F.text)
async def handle_text(message: Message) -> None:
    user_id = message.from_user.id
    text = (message.text or "").strip()
    if not text:
        return

    if text.casefold() in RESET_PHRASES:
        await reset_context(message)
        return

    if text.startswith("/"):
        await message.answer("Неизвестная команда. Список команд: /help")
        return

    if len(text) > settings.max_user_message_length:
        await message.answer(
            "Сообщение слишком длинное. Максимум: "
            f"{settings.max_user_message_length} символов."
        )
        return

    user = context.get_settings(user_id)
    messages = context.build_messages(user_id, text)
    logger.info(
        "Сообщение пользователя: user_id=%s model=%s temperature=%s %s=%s "
        "history=%d chars=%d",
        user_id,
        user.model,
        user.temperature,
        MODELS[user.model].token_param,
        user.max_tokens,
        context.history_size(user_id),
        len(text),
    )

    try:
        await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
        result = await client.ask(
            model=user.model,
            messages=messages,
            temperature=user.temperature,
            max_tokens=user.max_tokens,
        )
    except OpenAIServiceError as exc:
        logger.error("Ошибка API: user_id=%s error=%s", user_id, exc)
        await message.answer(f"Не удалось получить ответ: {exc}")
        return

    if not result.text:
        logger.warning(
            "Пустой ответ API: user_id=%s model=%s finish=%s tokens=%d",
            user_id,
            result.model,
            result.finish_reason,
            result.completion_tokens,
        )
        await message.answer(
            "Модель вернула пустой ответ. "
            f"finish_reason: {result.finish_reason or 'unknown'}. "
            "Попробуй увеличить лимит через /tokens или сменить модель через /model."
        )
        return

    context.add_exchange(user_id, text, result.text)
    await send_long(message, result.text + format_footer(result))


@dp.errors()
async def errors_handler(event: ErrorEvent) -> bool:
    logger.error("Необработанная ошибка: %s", event.exception, exc_info=event.exception)
    update = event.update
    message = update.message or update.edited_message
    if message is not None:
        try:
            await message.answer("Внутренняя ошибка бота. Попробуй ещё раз.")
        except TelegramBadRequest:
            logger.exception("Не удалось отправить сообщение об ошибке")
    return True


async def main() -> None:
    setup_logging()
    logger.info(
        "Запуск бота: model=%s fallback=%s base_url=%s",
        settings.default_model,
        settings.fallback_model or "нет",
        settings.openai_base_url,
    )
    try:
        await dp.start_polling(bot)
    finally:
        await client.close()
        await bot.session.close()
        logger.info("Бот остановлен")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.getLogger(__name__).info("Работа бота прервана пользователем")

