from __future__ import annotations

import asyncio
import logging
from typing import Dict, List

from agents.state.state import ConfigSchema
from agents.utils import ModelType
from job_agent.find_job_agent import initialize_agent #as initialize_job_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


async def _collect_final_text(assistant, payload_msg: HumanMessage, cfg: RunnableConfig) -> str:
    """Execute streaming response collection in a worker thread."""

    def _run() -> str:
        final_parts: List[str] = []
        printed_ids = set()
        events = assistant.stream({"messages": [payload_msg]}, cfg, stream_mode="values")
        for event in events:
            message = event.get("messages") if isinstance(event, dict) else None
            if not message:
                continue
            if isinstance(message, list):
                message = message[-1]
            mid = getattr(message, "id", None)
            if mid in printed_ids:
                continue
            if getattr(message, "type", "") == "ai":
                content = (getattr(message, "content", "") or "").strip()
                if content:
                    final_parts.append(content)
                    printed_ids.add(mid)
        return "\n".join(final_parts).strip()

    return await asyncio.to_thread(_run)


class AgentSession:
    """Maintain a single agent instance per dialog."""

    def __init__(self, dialog_id: str, user_id: str, role: str, model: ModelType = ModelType.GPT) -> None:
        self.dialog_id = dialog_id
        self.user_id = user_id
        self.role = role or "default"
        self.model = model
        self._assistant = None
        self._initialised = False
        self._lock = asyncio.Lock()

    def _ensure_assistant(self):
        if self._assistant is None:
            self._assistant = initialize_agent(provider=self.model, role=self.role)
        return self._assistant

    def _config(self) -> RunnableConfig:
        return RunnableConfig(
            ConfigSchema(
                {
                    "user_id": self.user_id,
                    "user_role": self.role,
                    "model": self.model,
                    "thread_id": self.dialog_id,
                }
            )
        )

    async def run(self, message_parts: List[dict]) -> str:
        async with self._lock:
            assistant = self._ensure_assistant()
            cfg = self._config()
            if not self._initialised:
                reset_msg = HumanMessage(content=[{"type": "reset", "text": "RESET"}])

                def _invoke_reset() -> None:
                    assistant.invoke({"messages": [reset_msg]}, cfg, stream_mode="values")

                await asyncio.to_thread(_invoke_reset)
                self._initialised = True
            payload_msg = HumanMessage(content=message_parts)
            return await _collect_final_text(assistant, payload_msg, cfg)

    def reset(self) -> None:
        self._assistant = None
        self._initialised = False


class AgentManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, AgentSession] = {}
        self._lock = asyncio.Lock()

    async def _get_session(self, dialog_id: str, user_id: str, role: str) -> AgentSession:
        async with self._lock:
            session = self._sessions.get(dialog_id)
            if session is None or session.user_id != user_id or session.role != (role or "default"):
                session = AgentSession(dialog_id=dialog_id, user_id=user_id, role=role or "default")
                self._sessions[dialog_id] = session
            return session

    async def run_dialog(self, dialog_id: str, user_id: str, role: str, message_parts: List[dict]) -> str:
        session = await self._get_session(dialog_id, user_id, role)
        try:
            return await session.run(message_parts)
        except Exception:
            logger.exception("Agent execution failed for dialog %s", dialog_id)
            raise

    async def reset_dialog(self, dialog_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(dialog_id, None)
            if session:
                session.reset()


agent_manager = AgentManager()
