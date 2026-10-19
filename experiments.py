from __future__ import annotations

import asyncio
import csv
import logging
from dataclasses import dataclass

from config import BASE_DIR, MODELS, settings
from openai_client import OpenAIClient, OpenAIServiceError

PROMPT = "Придумай короткий слоган для кофейни «Утро» и объясни его в одном предложении."
MAX_TOKENS_PROMPT = (
    "Придумай 10 коротких слоганов для кофейни «Утро», каждый с новой строки, "
    "без нумерации."
)
RESULT_FILE = BASE_DIR / "experiment_results.csv"


@dataclass(frozen=True)
class Experiment:
    model: str
    temperatures: tuple[float, ...]
    runs: int
    max_tokens: int
    prompt: str = PROMPT
    label: str = "temperature"


EXPERIMENTS = (
    Experiment("openai/gpt-4.1-mini", (0.0, 0.7, 1.3), 2, 120),
    Experiment("openai/gpt-4.1-nano", (0.0, 0.7, 1.3), 2, 120),
    Experiment("openai/gpt-4o-mini", (0.0, 0.7, 1.3), 2, 120),
    Experiment("deepseek/deepseek-chat", (0.0, 0.7, 1.3), 2, 120),
    Experiment("openai/gpt-4.1", (0.7,), 1, 120),
    Experiment("openai/gpt-3.5-turbo", (0.0, 0.7, 1.3), 2, 120),
    Experiment("openai/gpt-5-mini", (1.0,), 2, 400),
)

MAX_TOKENS_EXPERIMENTS = (
    Experiment("openai/gpt-4.1-mini", (0.7,), 1, 20, MAX_TOKENS_PROMPT, "max_tokens"),
    Experiment("openai/gpt-4.1-mini", (0.7,), 1, 60, MAX_TOKENS_PROMPT, "max_tokens"),
    Experiment("openai/gpt-4.1-mini", (0.7,), 1, 200, MAX_TOKENS_PROMPT, "max_tokens"),
)

ALL_EXPERIMENTS = EXPERIMENTS + MAX_TOKENS_EXPERIMENTS

FIELDS = [
    "experiment",
    "model",
    "temperature",
    "max_tokens",
    "token_param",
    "run",
    "status",
    "finish_reason",
    "chars",
    "words",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_rub",
    "request_id",
    "elapsed",
    "prompt",
    "text",
    "error",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


async def run_experiments() -> list[dict[str, object]]:
    client = OpenAIClient(settings)
    rows: list[dict[str, object]] = []
    try:
        for experiment in ALL_EXPERIMENTS:
            spec = MODELS[experiment.model]
            for temperature in experiment.temperatures:
                for run in range(1, experiment.runs + 1):
                    messages = [
                        {"role": "system", "content": settings.system_prompt},
                        {"role": "user", "content": experiment.prompt},
                    ]
                    print(
                        f"\n>>> [{experiment.label}] {spec.id} | "
                        f"temperature={temperature:g} | "
                        f"{spec.token_param}={experiment.max_tokens} | run={run}"
                    )
                    try:
                        result = await client.ask(
                            model=spec.id,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=experiment.max_tokens,
                        )
                    except OpenAIServiceError as exc:
                        print(f"FAILED: {exc}")
                        rows.append(
                            {
                                "experiment": experiment.label,
                                "model": spec.id,
                                "temperature": temperature,
                                "max_tokens": experiment.max_tokens,
                                "token_param": spec.token_param,
                                "run": run,
                                "status": "error",
                                "prompt": experiment.prompt,
                                "error": str(exc),
                            }
                        )
                        continue

                    words = len(result.text.split())
                    chars = len(result.text)
                    rows.append(
                        {
                            "experiment": experiment.label,
                            "model": spec.id,
                            "temperature": temperature,
                            "max_tokens": experiment.max_tokens,
                            "token_param": spec.token_param,
                            "run": run,
                            "status": "ok",
                            "finish_reason": result.finish_reason,
                            "chars": chars,
                            "words": words,
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "total_tokens": result.total_tokens,
                            "cost_rub": f"{result.cost_rub:.6f}",
                            "request_id": result.request_id,
                            "elapsed": f"{result.elapsed:.2f}",
                            "prompt": experiment.prompt,
                            "text": result.text,
                        }
                    )
                    print(
                        f"OK: words={words} chars={chars} "
                        f"tokens={result.prompt_tokens}+{result.completion_tokens}"
                        f"={result.total_tokens} cost={result.cost_rub:.6f} RUB "
                        f"finish={result.finish_reason} id={result.request_id}"
                    )
                    print(f"   {result.text}")
                    await asyncio.sleep(0.3)
    finally:
        await client.close()
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    with RESULT_FILE.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


async def main() -> None:
    setup_logging()
    rows = await run_experiments()
    write_csv(rows)
    ok = sum(1 for row in rows if row.get("status") == "ok")
    total_cost = sum(float(row.get("cost_rub", 0) or 0) for row in rows)
    print(f"\nГотово: {ok}/{len(rows)} успешных запросов")
    print(f"Суммарная стоимость: {total_cost:.6f} ₽")
    print(f"Результаты: {RESULT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
