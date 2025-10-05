"""
Aiogram-based Telegram bot focused on job-search assistance.
"""

import asyncio
import base64
import contextlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

# --- webhook imports (added) ---
from aiohttp import web
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

import config

if getattr(config, "NO_CUDA", "False") == "True":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from thread_settings import ThreadSettings
from agents.utils import ModelType, image_to_uri  # noqa: F401
from langchain_core.messages import HumanMessage

# Voice recognition
from vrecog.vrecog import recognise_text
from alioty import alioty

import telegramify_markdown  # noqa: F401 (used indirectly via helpers)

from bot_helpers import (
    collect_final_text_from_stream,
    determine_upload_action,
    finalize_placeholder_or_fallback,
    send_text_element,
    start_show_typing,
    vision_part_from_uri,
)

BOT_MODE = getattr(config, "BOT_MODE", "polling").lower()
WEBAPP_HOST = getattr(config, "WEBAPP_HOST", "0.0.0.0")
WEBAPP_PORT = int(getattr(config, "WEBAPP_PORT", "8080"))
WEBHOOK_BASE = getattr(config, "WEBHOOK_BASE", "https://0.0.0.0:88")
WEBHOOK_PATH = getattr(config, "WEBHOOK_PATH", "/tg-webhook")
WEBHOOK_URL = (WEBHOOK_BASE or "").rstrip("/") + WEBHOOK_PATH if WEBHOOK_BASE else None
WEBHOOK_SECRET = getattr(config, "WEBHOOK_SECRET", None)



def _truncate_label(text: str, limit: int = 64) -> str:
    return text if len(text) <= limit else text[: limit - 1] + '…'


def build_vacancy_keyboard(vacancies: List[Dict[str, Any]]) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    for idx, job in enumerate(vacancies):
        title = (job.get('title') or 'Вакансия').strip()
        company = (job.get('company') or '').strip()
        label = f"{idx + 1}. {title}" if title else f"Вакансия #{idx + 1}"
        if company:
            label += f" — {company}"
        rows.append([InlineKeyboardButton(text=_truncate_label(label), callback_data=f'job:{idx}')])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_vacancy_actions_keyboard(index: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text='Сформировать отклик', callback_data=f'vac:respond:{index}')],
            [InlineKeyboardButton(text='Скрыть вакансию', callback_data=f'vac:reject:{index}')],
            [InlineKeyboardButton(text='Сохранить', callback_data=f'vac:save:{index}')],
        ]
    )


def format_salary(salary: Optional[Dict[str, Any]]) -> str:
    if not isinstance(salary, dict):
        return 'не указана'
    minimum = salary.get('min')
    maximum = salary.get('max')
    currency = salary.get('currency') or 'RUB'
    if minimum is None and maximum is None:
        return 'не указана'
    if minimum is None:
        return f"до {maximum} {currency}"
    if maximum is None:
        return f"от {minimum} {currency}"
    if minimum == maximum:
        return f"{minimum} {currency}"
    return f"{minimum}-{maximum} {currency}"


def format_vacancy_details(job: Dict[str, Any]) -> str:
    title = job.get('title') or 'Без названия'
    company = job.get('company')
    location = job.get('location')
    salary_str = format_salary(job.get('salary'))
    published = job.get('published_at')
    source = job.get('source')
    experience = job.get('experience')
    match_score = job.get('match_score') or job.get('rank_score')
    description = job.get('description')
    skills = [skill for skill in job.get('skills') or [] if isinstance(skill, str)]

    parts: List[str] = [f'Вакансия: {title}']
    if company:
        parts.append(f'Компания: {company}')
    if location:
        parts.append(f'Локация: {location}')
    parts.append(f'Зарплата: {salary_str}')
    if experience:
        parts.append(f'Опыт: {experience}')
    if match_score is not None:
        parts.append(f'Оценка соответствия: {match_score}')
    if published:
        parts.append(f'Опубликовано: {published}')
    if source:
        parts.append(f'Источник: {source}')
    if skills:
        parts.append('Навыки: ' + ', '.join(skills[:12]))
    if description:
        parts.append('Описание: ' + description)
    url = job.get('url')
    if url:
        parts.append(f'Ссылка: {url}')
    return '\n'.join(parts)


def store_vacancy(thread: ThreadSettings, job: Dict[str, Any]) -> None:
    job_id = job.get('id')
    if job_id is not None:
        for saved in thread.saved_jobs:
            if isinstance(saved, dict) and saved.get('id') == job_id:
                return
    thread.saved_jobs.append(dict(job))


def generate_response(job: Dict[str, Any]) -> str:
    title = job.get('title') or 'Без названия'
    company = job.get('company')
    location = job.get('location')
    experience = job.get('experience')
    salary_str = format_salary(job.get('salary'))
    skills = [skill for skill in job.get('skills') or [] if isinstance(skill, str)]
    url = job.get('url')

    lines = ['Черновик отклика для проверки:']
    lines.append(f'Вакансия: {title}')
    if company:
        lines.append(f'Компания: {company}')
    if location:
        lines.append(f'Локация: {location}')
    lines.append(f'Указанная зарплата: {salary_str}')
    if experience:
        lines.append(f'Требуемый опыт: {experience}')
    if skills:
        lines.append('Основные навыки: ' + ', '.join(skills[:10]))
    if url:
        lines.append(f'Ссылка: {url}')
    lines.append('⚙️ Ответ собран заглушкой generate_response().')
    return '\n'.join(lines)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    bot = Bot(
        config.TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode="MarkdownV2"),
    )
    dp = Dispatcher()

    chats: Dict[int, ThreadSettings] = {}

    class ThrottlingMiddleware:
        def __init__(self, rate: float = 3.0):
            self.rate = rate
            self.last_called: Dict[tuple[int, int], float] = {}

        async def __call__(self, handler, event, data):
            from_user = getattr(event, "from_user", None)
            chat = getattr(event, "chat", None)
            if from_user and chat:
                key = (from_user.id, chat.id)
                now = time.monotonic()
                last = self.last_called.get(key, 0.0)
                elapsed = now - last
                if elapsed < self.rate:
                    await asyncio.sleep(self.rate - elapsed)
                self.last_called[key] = time.monotonic()
            return await handler(event, data)

    dp.message.middleware(ThrottlingMiddleware(rate=3.0))

    @dp.errors()
    async def global_error_handler(event: Any):
        exc = getattr(event, 'exception', None)
        if exc:
            logging.exception('Unhandled exception occured', exc_info=(type(exc), exc, exc.__traceback__))
        else:
            logging.exception('Unhandled exception occured')
        return True

    def ensure_thread(message: types.Message) -> ThreadSettings:
        chat_id = message.chat.id
        user_id = message.from_user.username
        thread = chats.get(chat_id)
        if thread is None:
            thread = ThreadSettings(user_id=user_id, chat_id=chat_id)
            chats[chat_id] = thread
        return thread

    # /start — resets AI agent's memory, greet
    @dp.message(Command("start"))
    async def cmd_start(message: types.Message) -> None:
        chat_id = message.chat.id
        user_id = message.from_user.username
        chats[chat_id] = ThreadSettings(user_id=user_id, chat_id=chat_id)

        assistant = chats[chat_id].assistant
        assistant.invoke(
            {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]},
            chats[chat_id].get_config(),
            stream_mode="values",
        )

        greeting = (
            "Привет! Я помогу подобрать вакансии. "
            "Отправьте резюме текстом, голосом или файлом — постараюсь найти подходящие предложения."
        )
        payload = HumanMessage(content=[{"type": "text", "text": greeting}])

        typing_task = await start_show_typing(bot, chat_id, ChatAction.TYPING)
        try:
            final_text = await collect_final_text_from_stream(
                assistant, payload, chats[chat_id].get_config()
            )
            await send_text_element(bot, chat_id, final_text, usr_msg=message)
        finally:
            typing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing_task

    def set_model(message: types.Message, model: ModelType) -> None:
        thread = ensure_thread(message)
        thread.model = model
        thread.assistant = None

    @dp.message(Command("sber"))
    async def cmd_sber(message: types.Message) -> None:
        set_model(message, ModelType.SBER)
        await bot.send_message(message.chat.id, "Использую GigaChat для дальнейшей работы.", parse_mode=None)

    @dp.message(Command("gpt"))
    async def cmd_gpt(message: types.Message) -> None:
        set_model(message, ModelType.GPT)
        await bot.send_message(message.chat.id, "Использую OpenAI GPT для дальнейшей работы.", parse_mode=None)

    @dp.message(Command("jobsearch"))
    async def cmd_jobsearch(message: types.Message) -> None:
        ensure_thread(message)
        await bot.send_message(message.chat.id, "Режим подбора вакансий активен. Пришлите резюме для анализа.", parse_mode=None)

    @dp.message(Command("reset"))
    async def cmd_reset(message: types.Message) -> None:
        chat_id = message.chat.id
        user_id = message.from_user.username
        chats[chat_id] = ThreadSettings(user_id=user_id, chat_id=chat_id)

        assistant = chats[chat_id].assistant
        assistant.invoke(
            {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]},
            chats[chat_id].get_config(),
            stream_mode="values",
        )
        await bot.send_message(chat_id, "Память бота очищена.", parse_mode=None)

    @dp.message(Command("help"))
    async def cmd_help(message: types.Message) -> None:
        chat_id = message.chat.id
        help_path = Path("./help/help.md")
        if help_path.exists():
            help_str = help_path.read_text(encoding="utf-8")
            await bot.send_message(chat_id, help_str, parse_mode="MarkdownV2")
        else:
            await bot.send_message(chat_id, "Файл справки не найден.", parse_mode=None)

    @dp.message(Command("users"))
    async def cmd_users(message: types.Message) -> None:
        thread = ensure_thread(message)
        if thread.reload_users():
            await bot.send_message(message.chat.id, "База пользователей обновлена.", parse_mode=None)

    @dp.message(lambda m: m.content_type in {"text", "voice", "photo", "document"})
    async def handle_message(message: types.Message) -> None:
        chat_id = message.chat.id
        thread = ensure_thread(message)
        thread.ranked_jobs = []
        placeholder: Optional[types.Message] = None
        typing_task: Optional[asyncio.Task] = None

        if not thread.is_allowed():
            await bot.send_message(
                chat_id,
                "К сожалению, мне не разрешено помогать вам. Пожалуйста, обратитесь к администратору бота.",
                parse_mode=None,
            )
            return

        image_payload: List[dict] = []
        query = message.text or getattr(message, "any_text", None) or (message.caption or "")

        upload_action = determine_upload_action(message)
        upload_task: Optional[asyncio.Task] = None

        try:
            if upload_action is not None:
                upload_task = await start_show_typing(bot, chat_id, upload_action)
                try:
                    if message.content_type == "voice":
                        file_id = message.voice.file_id
                        file_info = await bot.get_file(file_id)
                        voice_io = await bot.download_file(file_info.file_path)
                        raw = voice_io.getvalue() if hasattr(voice_io, "getvalue") else voice_io

                        tmp_path = f"voice_{message.from_user.username}_{int(time.time() * 1000)}.ogg"
                        with open(tmp_path, "wb") as tmp_file:
                            tmp_file.write(raw)

                        try:
                            query = await asyncio.to_thread(recognise_text, tmp_path)
                        finally:
                            with contextlib.suppress(Exception):
                                os.remove(tmp_path)

                        if not query:
                            await bot.send_message(
                                chat_id,
                                "Не удалось распознать голосовое сообщение. "
                                "Пожалуйста, отправьте текст вручную или попробуйте снова.",
                                parse_mode=None,
                            )
                            return

                    if message.content_type in {"photo", "document"} and not message.voice:
                        if message.photo:
                            file_id = message.photo[-1].file_id
                            file_name = f"photo_{message.from_user.username}_{int(time.time()*1000)}.jpg"
                            mime_type = "image/jpeg"
                        else:
                            file_id = message.document.file_id
                            file_name = message.document.file_name or ""
                            mime_type = (message.document.mime_type or "").lower()

                        file_info = await bot.get_file(file_id)
                        file_io = await bot.download_file(file_info.file_path)
                        raw_bytes = file_io.getvalue() if hasattr(file_io, "getvalue") else file_io

                        suffix = Path(file_name).suffix.lower()
                        if message.photo and not suffix:
                            suffix = ".jpg"

                        tmp_suffix = suffix or (".jpg" if mime_type.startswith("image/") else ".bin")
                        tmp_path = None
                        with tempfile.NamedTemporaryFile(delete=False, suffix=tmp_suffix) as tmp_file:
                            tmp_file.write(raw_bytes)
                            tmp_path = tmp_file.name

                        try:
                            extracted = await asyncio.to_thread(alioty, tmp_path, mime_type=mime_type or None)
                            if extracted:
                                query = (query + "\n\n" + extracted).strip()
                            elif not query.strip():
                                await bot.send_message(
                                    chat_id,
                                    "Не удалось извлечь текст из файла. "
                                    "Пожалуйста, отправьте текст вручную или попробуйте снова.",
                                    parse_mode=None,
                                )
                                return
                        finally:
                            if tmp_path:
                                with contextlib.suppress(Exception):
                                    os.remove(tmp_path)

                        is_image = bool(message.photo)
                        if not is_image and message.content_type == "document":
                            is_image = mime_type.startswith("image/") or suffix in {
                                ".png",
                                ".jpg",
                                ".jpeg",
                                ".gif",
                                ".bmp",
                                ".webp",
                                ".tiff",
                            }
                        if is_image:
                            uri = image_to_uri(base64.b64encode(raw_bytes).decode())
                            image_payload = [vision_part_from_uri(uri)]
                finally:
                    if upload_task:
                        upload_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await upload_task

            assistant = thread.assistant

            if not message.reply_to_message:
                assistant.invoke(
                    {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]},
                    thread.get_config(),
                    stream_mode="values",
                )

            placeholder = await message.reply("⌛ Обрабатываю запрос...", parse_mode=None)
            typing_task = await start_show_typing(bot, chat_id, ChatAction.TYPING)

            payload_msg = HumanMessage(content=[{"type": "text", "text": query}] + image_payload)

            try:
                final_answer = await collect_final_text_from_stream(assistant, payload_msg, thread.get_config())

                parsed_payload = None
                if final_answer:
                    try:
                        candidate = json.loads(final_answer)
                        if isinstance(candidate, dict) and "vacancies" in candidate:
                            parsed_payload = candidate
                    except json.JSONDecodeError:
                        parsed_payload = None

                if parsed_payload:
                    summary_text = parsed_payload.get("summary") or "Подбор вакансий завершён."
                    vacancies = parsed_payload.get("vacancies") or []
                    thread.ranked_jobs = [vac for vac in vacancies if isinstance(vac, dict)]
                    await finalize_placeholder_or_fallback(bot, placeholder, chat_id, summary_text)
                    if thread.ranked_jobs:
                        keyboard = build_vacancy_keyboard(thread.ranked_jobs)
                        if thread.vacancy_menu_message_id:
                            try:
                                await bot.edit_message_reply_markup(
                                    chat_id=chat_id,
                                    message_id=thread.vacancy_menu_message_id,
                                    reply_markup=keyboard,
                                )
                            except Exception:
                                msg = await bot.send_message(
                                    chat_id,
                                    "Выберите вакансию для подробностей:",
                                    reply_markup=keyboard,
                                    disable_web_page_preview=True,
                                    parse_mode=None,
                                )
                                thread.vacancy_menu_message_id = msg.message_id
                        else:
                            msg = await bot.send_message(
                                chat_id,
                                "Выберите вакансию для подробностей:",
                                reply_markup=keyboard,
                                disable_web_page_preview=True,
                                parse_mode=None,
                            )
                            thread.vacancy_menu_message_id = msg.message_id
                    else:
                        if thread.vacancy_menu_message_id:
                            with contextlib.suppress(Exception):
                                await bot.edit_message_text(
                                    "Подходящих вакансий больше нет.",
                                    chat_id=chat_id,
                                    message_id=thread.vacancy_menu_message_id,
                                    parse_mode=None,
                                )
                        thread.vacancy_menu_message_id = None
                else:
                    if thread.vacancy_menu_message_id:
                        with contextlib.suppress(Exception):
                            await bot.edit_message_text(
                                "Подходящих вакансий больше нет.",
                                chat_id=chat_id,
                                message_id=thread.vacancy_menu_message_id,
                                parse_mode=None,
                            )
                        thread.vacancy_menu_message_id = None
                    await finalize_placeholder_or_fallback(bot, placeholder, chat_id, final_answer or "Ответ не получен.")

            except Exception:
                logging.exception("Error while streaming/answering")
                with contextlib.suppress(Exception):
                    if placeholder:
                        await bot.edit_message_text(
                            chat_id=placeholder.chat.id,
                            message_id=placeholder.message_id,
                            text="❌ Ошибка при обработке ответа",
                        )
                    else:
                        await bot.send_message(chat_id, "❌ Ошибка при обработке ответа", parse_mode=None)
            finally:
                if typing_task:
                    typing_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await typing_task
        except Exception:
            logging.exception("Unexpected error in handle_message")
            with contextlib.suppress(Exception):
                if placeholder:
                    await bot.edit_message_text(
                        chat_id=placeholder.chat.id,
                        message_id=placeholder.message_id,
                        text="❌ Ошибка при обработке ответа",
                    )
                else:
                    await bot.send_message(chat_id, "❌ Ошибка при обработке ответа", parse_mode=None)


    @dp.callback_query(lambda c: c.data and c.data.startswith("job:"))
    async def handle_job_callback(call: types.CallbackQuery):
        chat_id = call.message.chat.id
        thread = chats.get(chat_id)
        if not thread or not thread.ranked_jobs:
            await call.answer("Список вакансий недоступен", show_alert=True)
            return

        try:
            index = int(call.data.split(":", 1)[1])
        except (ValueError, IndexError):
            await call.answer()
            return

        if index < 0 or index >= len(thread.ranked_jobs):
            await call.answer("Вакансия не найдена", show_alert=True)
            return

        job = thread.ranked_jobs[index]
        details_text = format_vacancy_details(job)
        actions_keyboard = build_vacancy_actions_keyboard(index)
        await call.message.answer(
            details_text,
            disable_web_page_preview=True,
            parse_mode=None,
            reply_markup=actions_keyboard,
        )
        await call.answer()


    @dp.callback_query(lambda c: c.data and c.data.startswith("vac:respond:"))
    async def handle_vacancy_respond(call: types.CallbackQuery):
        chat_id = call.message.chat.id
        thread = chats.get(chat_id)
        if not thread or not thread.ranked_jobs:
            await call.answer("Список вакансий недоступен", show_alert=True)
            return

        try:
            index = int(call.data.split(":", 2)[2])
        except (ValueError, IndexError):
            await call.answer()
            return

        if index < 0 or index >= len(thread.ranked_jobs):
            await call.answer("Вакансия не найдена", show_alert=True)
            return

        job = thread.ranked_jobs[index]
        response_text = generate_response(job)
        await call.message.answer(response_text, disable_web_page_preview=True, parse_mode=None)
        await call.answer("Черновик отклика сформирован", show_alert=False)


    @dp.callback_query(lambda c: c.data and c.data.startswith("vac:reject:"))
    async def handle_vacancy_reject(call: types.CallbackQuery):
        chat_id = call.message.chat.id
        thread = chats.get(chat_id)
        if not thread or not thread.ranked_jobs:
            await call.answer("Список вакансий недоступен", show_alert=True)
            return

        try:
            index = int(call.data.split(":", 2)[2])
        except (ValueError, IndexError):
            await call.answer()
            return

        if index < 0 or index >= len(thread.ranked_jobs):
            await call.answer("Вакансия не найдена", show_alert=True)
            return

        job = thread.ranked_jobs.pop(index)

        with contextlib.suppress(Exception):
            await call.message.edit_text(
                "Вакансия скрыта и удалена из списка.",
                parse_mode=None,
                reply_markup=None,
            )

        if thread.ranked_jobs:
            keyboard = build_vacancy_keyboard(thread.ranked_jobs)
            if thread.vacancy_menu_message_id:
                try:
                    await bot.edit_message_reply_markup(
                        chat_id=chat_id,
                        message_id=thread.vacancy_menu_message_id,
                        reply_markup=keyboard,
                    )
                except Exception:
                    msg = await bot.send_message(
                        chat_id,
                        "Выберите вакансию для подробностей:",
                        reply_markup=keyboard,
                        disable_web_page_preview=True,
                        parse_mode=None,
                    )
                    thread.vacancy_menu_message_id = msg.message_id
            else:
                msg = await bot.send_message(
                    chat_id,
                    "Выберите вакансию для подробностей:",
                    reply_markup=keyboard,
                    disable_web_page_preview=True,
                    parse_mode=None,
                )
                thread.vacancy_menu_message_id = msg.message_id
        else:
            if thread.vacancy_menu_message_id:
                with contextlib.suppress(Exception):
                    await bot.edit_message_text(
                        "Подходящих вакансий больше нет.",
                        chat_id=chat_id,
                        message_id=thread.vacancy_menu_message_id,
                        parse_mode=None,
                    )
                thread.vacancy_menu_message_id = None

        await call.answer("Вакансия скрыта", show_alert=False)


    @dp.callback_query(lambda c: c.data and c.data.startswith("vac:save:"))
    async def handle_vacancy_save(call: types.CallbackQuery):
        chat_id = call.message.chat.id
        thread = chats.get(chat_id)
        if not thread or not thread.ranked_jobs:
            await call.answer("Список вакансий недоступен", show_alert=True)
            return

        try:
            index = int(call.data.split(":", 2)[2])
        except (ValueError, IndexError):
            await call.answer()
            return

        if index < 0 or index >= len(thread.ranked_jobs):
            await call.answer("Вакансия не найдена", show_alert=True)
            return

        job = thread.ranked_jobs[index]
        store_vacancy(thread, job)
        await call.answer("Вакансия сохранена", show_alert=False)
        await call.message.answer("Вакансия сохранена для дальнейшей работы.", parse_mode=None)

    hook_mode = BOT_MODE in {"hook", "@hook@", "webhook"}
    if hook_mode:
        if not WEBHOOK_URL:
            logging.error("WEBHOOK_BASE is not set. Set config.WEBHOOK_BASE or env WEBHOOK_BASE to run in @hook@ mode.")
            return

        app = web.Application()
        SimpleRequestHandler(dispatcher=dp, bot=bot, secret_token=WEBHOOK_SECRET).register(app, path=WEBHOOK_PATH)
        setup_application(app, dp, bot=bot)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host=WEBAPP_HOST, port=WEBAPP_PORT)

        try:
            await bot.set_webhook(url=WEBHOOK_URL, secret_token=WEBHOOK_SECRET)
            await site.start()
            logging.info("Webhook set: %s (listening on %s:%s)", WEBHOOK_URL, WEBAPP_HOST, WEBAPP_PORT)
            await asyncio.Event().wait()
        finally:
            with contextlib.suppress(Exception):
                await bot.delete_webhook(drop_pending_updates=False)
            with contextlib.suppress(Exception):
                await runner.cleanup()
    else:
        await dp.start_polling(bot)

if __name__ == "__main__":
    pid = os.getpid()
    with open(".process", "w", encoding="utf-8") as proc_file:
        proc_file.write(f"{pid}")
    asyncio.run(main())
