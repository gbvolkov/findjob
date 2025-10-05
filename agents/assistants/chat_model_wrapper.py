import wrapt
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages.base import BaseMessage
from typing import Callable, Dict, Any, List, Optional
from pydantic import Field
import functools
from typing import Annotated

def make_anonymized_tool(func, anonymiser, deanonymiser):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        # 1) call the real tool
        raw_output = func(*args, **kwargs)

        # 3) deanonymize its output
        return anonymiser(raw_output)

    return wrapped

class AnonimizedChatModelProxy(wrapt.ObjectProxy):
    _anonymiser: Optional[Callable] = Field(None)
    _deanonymiser: Optional[Callable]  = Field(None)    

    def __init__(self, wrapped, anonymiser, deanonymiser):
        super().__init__(wrapped)
        self._anonymiser = anonymiser
        self._deanonymiser = deanonymiser
    def invoke(self, *args, **kwargs):
        prompt = args
        if not isinstance(prompt, ChatPromptValue):
            if isinstance(prompt, (list, tuple)):
                prompt = args[0]
            else:
                return self.__wrapped__.invoke(*args, **kwargs)
        if isinstance(prompt, ChatPromptValue):
            messages  = prompt.messages
            for message in messages:
                if message.content:
                    if isinstance(message.content, str) and message.content.strip() != "":
                        message.content = self._anonymiser(message.content)
                    elif isinstance(message.content, dict) and message.content.get("text") and message.content["text"].strip() != "":
                        message.content["text"] = self._anonymiser(message.content["text"])
                    elif isinstance(message.content, (list, tuple)):
                        for content in message.content:
                            if isinstance(message.content, dict) and message.content.get("text") and message.content["text"].strip() != "":
                                message.content["text"] = self._anonymiser(message.content["text"])
                            elif isinstance(message.content, str) and message.content.strip() != "":
                                message.content = self._anonymiser(message.content)

        result = self.__wrapped__.invoke(*args, **kwargs)
        if isinstance(result, BaseMessage) and result.content and result.content.strip() != "":
            result.content = self._deanonymiser(result.content)
        return result

