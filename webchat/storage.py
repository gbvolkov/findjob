from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import ChatFile, Dialog, Message, Task

DATA_DIR = Path("data")
STATE_PATH = DATA_DIR / "state.json"
UPLOAD_DIR = DATA_DIR / "uploads"
EXPORT_DIR = DATA_DIR / "dialog_exports"
TASKS_PATH = DATA_DIR / "tasks" / "tasks.json"


class Storage:
    def __init__(self) -> None:
        self.dialogs: Dict[str, Dialog] = {}
        self.tasks: Dict[str, Task] = {}
        self._load_state()
        self._load_tasks()

    def _load_state(self) -> None:
        if STATE_PATH.exists():
            with STATE_PATH.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for item in raw.get("dialogs", []):
                dialog = Dialog(
                    id=item["id"],
                    title=item.get("title", "Диалог"),
                    bot_type=item.get("bot_type", "assistant"),
                )
                created_at = item.get("created_at")
                updated_at = item.get("updated_at")
                if created_at:
                    dialog.created_at = dialog.updated_at = datetime.fromisoformat(created_at.replace("Z", ""))
                if updated_at:
                    dialog.updated_at = datetime.fromisoformat(updated_at.replace("Z", ""))
                for message in item.get("messages", []):
                    msg = Message(
                        id=message["id"],
                        sender=message.get("sender", "user"),
                        content=message.get("content", ""),
                    )
                    created = message.get("created_at")
                    if created:
                        msg.created_at = datetime.fromisoformat(created.replace("Z", ""))
                    msg.attachments = list(message.get("attachments", []))
                    dialog.messages.append(msg)
                for file_item in item.get("files", []):
                    chat_file = ChatFile(
                        id=file_item["id"],
                        filename=file_item.get("filename", "file.txt"),
                        content_type=file_item.get("content_type", "text/plain"),
                        size=file_item.get("size", 0),
                        path=file_item.get("path", ""),
                    )
                    uploaded = file_item.get("uploaded_at")
                    if uploaded:
                        chat_file.uploaded_at = datetime.fromisoformat(uploaded.replace("Z", ""))
                    dialog.files[chat_file.id] = chat_file
                if dialog.messages:
                    dialog.updated_at = dialog.messages[-1].created_at
                self.dialogs[dialog.id] = dialog

    def _load_tasks(self) -> None:
        if TASKS_PATH.exists():
            with TASKS_PATH.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for item in raw:
                task = Task(
                    id=item["id"],
                    title=item["title"],
                    category=item["category"],
                    description=item.get("description", ""),
                )
                self.tasks[task.id] = task
        else:
            self.tasks = {
                "demo-strategy": Task(
                    id="demo-strategy",
                    title="Разработать стратегию поиска вакансий",
                    category="job-search",
                    description="Шаги и ресурсы для эффективного поиска вакансий.",
                ),
                "demo-cv": Task(
                    id="demo-cv",
                    title="Подготовить резюме",
                    category="documents",
                    description="Инструкция по сбору информации и подготовке резюме.",
                ),
            }

    def _persist(self) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with STATE_PATH.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "dialogs": [
                        {
                            "id": dialog.id,
                            "title": dialog.title,
                            "bot_type": dialog.bot_type,
                            "created_at": dialog.created_at.isoformat() + "Z",
                            "updated_at": (dialog.updated_at or dialog.created_at).isoformat() + "Z",
                            "messages": [
                                {
                                    "id": message.id,
                                    "sender": message.sender,
                                    "content": message.content,
                                    "created_at": message.created_at.isoformat() + "Z",
                                    "attachments": message.attachments,
                                }
                                for message in dialog.messages
                            ],
                            "files": [
                                {
                                    "id": chat_file.id,
                                    "filename": chat_file.filename,
                                    "content_type": chat_file.content_type,
                                    "size": chat_file.size,
                                    "path": chat_file.path,
                                    "uploaded_at": chat_file.uploaded_at.isoformat() + "Z",
                                }
                                for chat_file in dialog.files.values()
                            ],
                        }
                        for dialog in self.dialogs.values()
                    ]
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )

    def list_dialogs(self) -> List[Dialog]:
        return sorted(
            self.dialogs.values(),
            key=lambda dlg: dlg.updated_at or dlg.created_at,
            reverse=True,
        )

    def search_dialogs(self, query: str) -> List[Dialog]:
        query_lower = query.lower()
        return [
            dialog
            for dialog in self.dialogs.values()
            if query_lower in dialog.title.lower()
            or any(query_lower in message.content.lower() for message in dialog.messages)
        ]

    def create_dialog(self, title: str, bot_type: str) -> Dialog:
        dialog_id = uuid.uuid4().hex
        dialog = Dialog(id=dialog_id, title=title, bot_type=bot_type)
        self.dialogs[dialog_id] = dialog
        self._persist()
        return dialog

    def delete_dialog(self, dialog_id: str) -> None:
        dialog = self.dialogs.pop(dialog_id, None)
        if dialog:
            for chat_file in dialog.files.values():
                try:
                    Path(chat_file.path).unlink(missing_ok=True)
                except OSError:
                    pass
            self._persist()

    def clear_dialog(self, dialog_id: str) -> None:
        dialog = self.dialogs.get(dialog_id)
        if not dialog:
            return
        for chat_file in dialog.files.values():
            try:
                Path(chat_file.path).unlink(missing_ok=True)
            except OSError:
                pass
        dialog.messages.clear()
        dialog.files.clear()
        dialog.updated_at = dialog.created_at
        self._persist()

    def export_dialog(self, dialog_id: str) -> Optional[Path]:
        dialog = self.dialogs.get(dialog_id)
        if not dialog:
            return None
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        export_path = EXPORT_DIR / f"{dialog_id}.txt"
        with export_path.open("w", encoding="utf-8") as fh:
            fh.write(f"Диалог: {dialog.title}\nТип: {dialog.bot_type}\n\n")
            for message in dialog.messages:
                sender = "Пользователь" if message.sender == "user" else "Бот"
                fh.write(f"[{message.created_at.isoformat()}] {sender}: {message.content}\n")
        return export_path

    def get_dialog(self, dialog_id: str) -> Optional[Dialog]:
        return self.dialogs.get(dialog_id)

    def add_message(
        self,
        dialog_id: str,
        sender: str,
        content: str,
        attachments: Optional[Iterable[str]] = None,
    ) -> Message:
        dialog = self.dialogs[dialog_id]
        message = Message(id=uuid.uuid4().hex, sender=sender, content=content)
        if attachments:
            message.attachments = list(attachments)
        dialog.messages.append(message)
        dialog.updated_at = message.created_at
        self._persist()
        return message

    def store_file(self, dialog_id: str, filename: str, content_type: str, data: bytes) -> ChatFile:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        file_id = uuid.uuid4().hex
        sanitized_name = filename.replace("/", "_")
        path = UPLOAD_DIR / f"{dialog_id}_{file_id}_{sanitized_name}"
        with path.open("wb") as fh:
            fh.write(data)
        chat_file = ChatFile(
            id=file_id,
            filename=filename,
            content_type=content_type,
            size=len(data),
            path=str(path),
        )
        dialog = self.dialogs[dialog_id]
        dialog.files[file_id] = chat_file
        self._persist()
        return chat_file

    def get_file(self, dialog_id: str, file_id: str) -> Optional[ChatFile]:
        dialog = self.dialogs.get(dialog_id)
        if not dialog:
            return None
        return dialog.files.get(file_id)

    def list_tasks(self, query: Optional[str] = None, category: Optional[str] = None) -> List[Task]:
        tasks = list(self.tasks.values())
        if category:
            tasks = [task for task in tasks if task.category == category]
        if query:
            query_lower = query.lower()
            tasks = [task for task in tasks if query_lower in task.title.lower() or query_lower in task.description.lower()]
        return tasks

    def list_categories(self) -> List[str]:
        return sorted({task.category for task in self.tasks.values()})


storage = Storage()
