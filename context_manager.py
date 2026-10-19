from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UserSettings:
    model: str
    temperature: float
    max_tokens: int


class ContextManager:
    def __init__(
        self,
        history_limit: int,
        system_prompt: str,
        default_model: str,
        default_temperature: float,
        default_max_tokens: int,
    ) -> None:
        self.history_limit = max(2, history_limit - history_limit % 2)
        self.system_prompt = system_prompt
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self._history: dict[int, list[dict[str, str]]] = {}
        self._settings: dict[int, UserSettings] = {}

    def get_settings(self, user_id: int) -> UserSettings:
        settings = self._settings.get(user_id)
        if settings is None:
            settings = UserSettings(
                model=self.default_model,
                temperature=self.default_temperature,
                max_tokens=self.default_max_tokens,
            )
            self._settings[user_id] = settings
        return settings

    def set_model(self, user_id: int, model: str) -> UserSettings:
        settings = self.get_settings(user_id)
        settings.model = model
        return settings

    def set_temperature(self, user_id: int, temperature: float) -> UserSettings:
        settings = self.get_settings(user_id)
        settings.temperature = temperature
        return settings

    def set_max_tokens(self, user_id: int, max_tokens: int) -> UserSettings:
        settings = self.get_settings(user_id)
        settings.max_tokens = max_tokens
        return settings

    def get_history(self, user_id: int) -> list[dict[str, str]]:
        return list(self._history.get(user_id, []))

    def build_messages(self, user_id: int, user_text: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        messages.extend(self._history.get(user_id, []))
        messages.append({"role": "user", "content": user_text})
        return messages

    def add_exchange(self, user_id: int, user_text: str, assistant_text: str) -> None:
        history = self._history.setdefault(user_id, [])
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})
        if len(history) > self.history_limit:
            self._history[user_id] = history[-self.history_limit :]

    def clear(self, user_id: int) -> bool:
        return self._history.pop(user_id, None) is not None

    def history_size(self, user_id: int) -> int:
        return len(self._history.get(user_id, []))
