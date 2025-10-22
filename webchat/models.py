from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class ChatFile:
    id: str
    filename: str
    content_type: str
    size: int
    path: str
    uploaded_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "filename": self.filename,
            "content_type": self.content_type,
            "size": self.size,
            "uploaded_at": self.uploaded_at.isoformat() + "Z",
        }


@dataclass
class Message:
    id: str
    sender: str
    content: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    attachments: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "sender": self.sender,
            "content": self.content,
            "created_at": self.created_at.isoformat() + "Z",
            "attachments": self.attachments,
        }


@dataclass
class Dialog:
    id: str
    title: str
    bot_type: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    messages: List[Message] = field(default_factory=list)
    files: Dict[str, ChatFile] = field(default_factory=dict)

    def to_card(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "title": self.title,
            "bot_type": self.bot_type,
            "created_at": self.created_at.isoformat() + "Z",
            "updated_at": self.updated_at.isoformat() + "Z",
            "preview": self.messages[-1].content[:160] if self.messages else "",
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "bot_type": self.bot_type,
            "created_at": self.created_at.isoformat() + "Z",
            "updated_at": self.updated_at.isoformat() + "Z",
            "messages": [message.to_dict() for message in self.messages],
            "files": [chat_file.to_dict() for chat_file in self.files.values()],
        }


@dataclass
class Task:
    id: str
    title: str
    category: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "description": self.description,
        }
