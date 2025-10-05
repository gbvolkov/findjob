from agents.utils import ModelType
from job_agent.find_job_agent import initialize_agent as initialize_job_agent
from agents.state.state import ConfigSchema
from typing import Any, Dict, List, Optional

from user_manager.utils import UserManager
from langchain_core.runnables import RunnableConfig
import config


class ThreadSettings:
    user_man = UserManager()

    def __init__(self, user_id, chat_id, model=ModelType.SBER if config.LLM_PROVIDER == "gigachat" else ModelType.GPT):
        self.model = model
        self._assistant = None
        self.user_id = user_id
        self.chat_id = chat_id
        self.role = ThreadSettings.user_man.get_role(user_id)
        self.ranked_jobs: List[Dict[str, Any]] = []
        self.saved_jobs: List[Dict[str, Any]] = []
        self.last_resume_text: Optional[str] = None
        self.vacancy_menu_message_id: Optional[int] = None

    def is_allowed(self) -> bool:
        return config.CHECK_RIGHTS.strip().lower() != 'true' or self.user_man.is_allowed(self.user_id)

    def is_admin(self) -> bool:
        return self.user_man.is_admin(self.user_id)

    def reload_users(self):
        if self.is_admin():
            self.user_man.load_users()
            return True
        return False

    @property
    def assistant(self):
        if self._assistant is None:
            self._assistant = initialize_job_agent(provider=self.model, role=self.role)
        return self._assistant

    @assistant.setter
    def assistant(self, assistant):
        self._assistant = assistant

    def get_config(self):
        return RunnableConfig(ConfigSchema({
            "user_id": self.user_id,
            "user_role": self.role,
            "model": self.model,
            "thread_id": self.chat_id,
        }))
