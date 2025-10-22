from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from webchat.models import Dialog
from webchat.storage import EXPORT_DIR, storage

app = FastAPI(title="Job Search Web Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
WEBCHAT_DIR = BASE_DIR / "webchat"
STATIC_DIR = WEBCHAT_DIR / "static"
TEMPLATES_DIR = WEBCHAT_DIR / "templates"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class UserContext(Dict[str, Any]):
    pass


def get_current_user(request: Request) -> UserContext:
    # In a real system, this would validate a token or session identifier.
    user_id = request.headers.get("X-User-ID", "demo-user")
    return UserContext({"id": user_id, "name": "Demo User", "role": "user"})


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return index_path.read_text(encoding="utf-8")


@app.get("/api/session")
def get_session(user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    return {"user": user}


@app.get("/api/dialogs")
def list_dialogs(user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    dialogs = [dialog.to_card() for dialog in storage.list_dialogs()]
    return {"dialogs": dialogs}


@app.get("/api/dialogs/search")
def search_dialogs(q: str, user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    results = [dialog.to_card() for dialog in storage.search_dialogs(q)]
    return {"dialogs": results}


@app.post("/api/dialogs")
def create_dialog(payload: Dict[str, str], user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    title = payload.get("title") or "Новый диалог"
    bot_type = payload.get("bot_type") or "assistant"
    dialog = storage.create_dialog(title, bot_type)
    return dialog.to_dict()


@app.get("/api/dialogs/{dialog_id}")
def get_dialog(dialog_id: str, user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    dialog = storage.get_dialog(dialog_id)
    if not dialog:
        raise HTTPException(status_code=404, detail="Диалог не найден")
    return dialog.to_dict()


@app.delete("/api/dialogs/{dialog_id}")
def delete_dialog(dialog_id: str, user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    storage.delete_dialog(dialog_id)
    return {"status": "deleted"}


@app.get("/api/dialogs/{dialog_id}/export")
def export_dialog(dialog_id: str, user: UserContext = Depends(get_current_user)) -> FileResponse:
    export_path = storage.export_dialog(dialog_id)
    if not export_path:
        raise HTTPException(status_code=404, detail="Диалог не найден")
    return FileResponse(str(export_path), filename=f"dialog-{dialog_id}.txt", media_type="text/plain")


@app.post("/api/dialogs/{dialog_id}/messages")
def add_user_message(dialog_id: str, payload: Dict[str, Any], user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    dialog = storage.get_dialog(dialog_id)
    if not dialog:
        raise HTTPException(status_code=404, detail="Диалог не найден")
    message_text = payload.get("content")
    if not message_text:
        raise HTTPException(status_code=400, detail="Сообщение не может быть пустым")
    attachments = payload.get("attachments") or []
    message = storage.add_message(dialog_id, "user", message_text, attachments)
    return message.to_dict()


@app.post("/api/dialogs/{dialog_id}/files")
async def upload_file(dialog_id: str, uploaded_file: UploadFile = File(...), user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    dialog = storage.get_dialog(dialog_id)
    if not dialog:
        raise HTTPException(status_code=404, detail="Диалог не найден")
    if not uploaded_file.content_type.startswith("text/"):
        raise HTTPException(status_code=400, detail="Поддерживаются только текстовые файлы")
    data = await uploaded_file.read()
    chat_file = storage.store_file(dialog_id, uploaded_file.filename, uploaded_file.content_type, data)
    return chat_file.to_dict()


@app.get("/api/dialogs/{dialog_id}/files/{file_id}")
def download_file(dialog_id: str, file_id: str, user: UserContext = Depends(get_current_user)) -> FileResponse:
    chat_file = storage.get_file(dialog_id, file_id)
    if not chat_file:
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(chat_file.path, filename=chat_file.filename, media_type=chat_file.content_type)


@app.get("/api/tasks")
def list_tasks(q: Optional[str] = None, category: Optional[str] = None, user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    tasks = [task.to_dict() for task in storage.list_tasks(q, category)]
    return {"tasks": tasks}


@app.get("/api/tasks/categories")
def list_categories(user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    return {"categories": storage.list_categories()}


@app.post("/api/dialogs/{dialog_id}/commands")
def run_command(dialog_id: str, payload: Dict[str, Any], user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    command = payload.get("command")
    if not command:
        raise HTTPException(status_code=400, detail="Команда не указана")
    if command == "clear":
        if not storage.get_dialog(dialog_id):
            raise HTTPException(status_code=404, detail="Диалог не найден")
        storage.clear_dialog(dialog_id)
        return {"status": "cleared"}
    if command == "help":
        return {
            "status": "ok",
            "message": "Доступные команды: /help, /clear",
        }
    raise HTTPException(status_code=400, detail="Неизвестная команда")


@app.get("/api/dialogs/{dialog_id}/sources")
def list_sources(dialog_id: str, user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    # Stubbed source service.
    return {
        "sources": [
            {"name": "hh.ru", "description": "HeadHunter вакансии"},
            {"name": "habr career", "description": "Habr Career вакансии"},
        ]
    }


@app.get("/api/dialogs/{dialog_id}/messages")
def get_messages(dialog_id: str, user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    dialog = storage.get_dialog(dialog_id)
    if not dialog:
        raise HTTPException(status_code=404, detail="Диалог не найден")
    return {"messages": [message.to_dict() for message in dialog.messages]}


async def _simulate_bot_response(dialog: Dialog, content: str) -> str:
    await asyncio.sleep(0.5)
    return f"Получил: {content}\n\n(Это демонстрационный ответ бота типа {dialog.bot_type}.)"


@app.websocket("/ws/dialogs/{dialog_id}")
async def chat_websocket(websocket: WebSocket, dialog_id: str) -> None:
    await websocket.accept()
    dialog = storage.get_dialog(dialog_id)
    if not dialog:
        await websocket.close(code=4404)
        return
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            content = payload.get("content", "")
            attachments = payload.get("attachments") or []
            user_message = storage.add_message(dialog_id, "user", content, attachments)
            await websocket.send_json({"event": "ack", "message": user_message.to_dict()})
            bot_reply = await _simulate_bot_response(dialog, content)
            bot_message = storage.add_message(dialog_id, "bot", bot_reply)
            await websocket.send_json({"event": "bot_message", "message": bot_message.to_dict()})
    except WebSocketDisconnect:
        return


@app.get("/api/dialogs/{dialog_id}/export/status")
def export_status(dialog_id: str, user: UserContext = Depends(get_current_user)) -> Dict[str, Any]:
    path = EXPORT_DIR / f"{dialog_id}.txt"
    return {"exists": path.exists()}


def main() -> None:
    """Run the web chat application with Uvicorn."""
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
