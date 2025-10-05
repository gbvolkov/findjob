import logging
logging.basicConfig(level=logging.INFO)

import config

import pandas as pd


from langchain_core.messages import HumanMessage
import time, uuid, json, os, base64
from collections import defaultdict

from agents.agent import initialize_agent

from agents.utils import _print_response, summarise_image, image_to_uri

def _get_response(event: dict, _printed: set, max_length=1500):
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            if message.type == "ai" and message.content.strip() != "":
                msg_repr = message.content.strip()
                if len(msg_repr) > max_length:
                    msg_repr = msg_repr[:max_length] + " ... (truncated)"
                return msg_repr
            _printed.add(message.id)
    return ""


def factory():
    cfg = {
            "configurable": {
                # The user_id is used in our tools to
                # fetch the user's information
                "student_id": 1111111,
                # Checkpoints are accessed by thread_id
                "thread_id": 1111111,
            }
        }
    assistant = initialize_agent()

    def get_answer(question):
        _printed = set()
        events = assistant.stream({"messages": [HumanMessage(content=[{"type": "text", "text": question}])]}, cfg, stream_mode="values")
        answer = ""
        for event in events:
            #_print_event(event, _printed)
            answer = answer + _get_response(event, _printed, 100000)
        return answer


    def reset_context():
        assistant.invoke(
            {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]}, cfg, stream_mode="values"
        )

    return get_answer, reset_context

if __name__ == '__main__':
    (get_answer, reset_context) = factory()

    answer = get_answer("Как получить грант или скидку 50 % на обучение в Университете Зерокодер?")
    print(answer)
    
    answer = get_answer("Чем Нейрокот 4-mini  отличается от того, с которым мы сейчас работаем 4-o?")
    print(answer)

    # ── 1. Загрузка вопросов ────────────────────────────────────────────────
    df = pd.read_csv("data/questions.csv")          # question, theme

    # ── 2. Сортировка по теме (чтобы минимизировать лишние reset_context) ──
    df = df.sort_values("theme").reset_index(drop=True)

    # ── 3. Итерация, сброс контекста при смене темы и получение ответов ────
    answers = []
    current_theme = None

    for _, row in df.iterrows():
        theme   = row["theme"]
        question = row["question"]

        if theme != current_theme:     # тема изменилась → начинаем «чистый» диалог
            reset_context()
            current_theme = theme

        answers.append(get_answer(question))

    # ── 4. Сохранение результатов ──────────────────────────────────────────
    df["answer"] = answers
    df.to_csv("data/answers.csv", index=False)

    print("Готово! Ответы сохранены в файл answers.csv")