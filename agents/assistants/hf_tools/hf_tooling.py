import re
import json
from typing import Any, List, Optional, Sequence, Union, Dict

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    GenerationConfig,
)
from pydantic import BaseModel, Field
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration, LLMResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate

class ChatHuggingFaceWithTools(ChatHuggingFace):
    """
    Extends ChatHuggingFace to support local text-generation pipelines with function calling.
    Injects tool schemas into the prompt and parses full JSON calls, Yandex-style calls,
    and bare-argument JSON robustly.
    """

    def bind_tools(
        self,
        tools: Sequence[Union[BaseTool, dict, Any]],
        *,
        tool_choice: Optional[Union[dict, str, bool]] = None,
        **kwargs: Any,
    ) -> Any:
        # Convert and store for fallback parsing
        self._pipeline_tools = [convert_to_openai_tool(t) for t in tools]
        return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Use parent for remote chat endpoints
        from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
        try:
            from langchain_community.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference  # type: ignore
        except ImportError:
            HuggingFaceTextGenInference = ()

        if isinstance(self.llm, (HuggingFaceTextGenInference, HuggingFaceEndpoint)):
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        # Local pipeline branch: inject tools
        tools = kwargs.pop("tools", None)
        chatml = []
        for m in messages:
            if isinstance(m, SystemMessage):
                chatml.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                chatml.append({"role": "user", "content": m.content})
            else:
                raise ValueError("Local pipeline supports only system and human messages")

        prompt = self.tokenizer.apply_chat_template(
            conversation=chatml,
            tools=tools,
            add_generation_prompt=True,
        )

        llm_result: LLMResult = self.llm._generate(
            prompts=[prompt], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    def _to_chat_result(self, llm_result: LLMResult) -> ChatResult:
        generations = []
        for g in llm_result.generations[0]:
            text = g.text
            calls: List[Dict[str, Any]] = []

            # 1) Full JSON function calls
            for m in re.findall(r"\{.*?\"name\".*?\}", text, flags=re.DOTALL):
                try:
                    obj = json.loads(m)
                    calls.append({"name": obj["name"], "args": obj.get("arguments", {}), "id": "0"})
                except Exception:
                    continue

            # 2) Yandex-style [TOOL_CALL_START]name\n{args}
            if not calls:
                y = re.search(
                    r"\[TOOL_CALL_START\]([^\n]+)\n(\{.*?\})",
                    text,
                    flags=re.DOTALL,
                )
                if y:
                    name = y.group(1).strip()
                    try:
                        args = json.loads(y.group(2))
                        calls.append({"name": name, "args": args, "id": "0"})
                    except Exception:
                        pass

            # 3) Bare-args fallback: single-tool scenarios
            if not calls and hasattr(self, "_pipeline_tools") and len(self._pipeline_tools) == 1:
                for m in re.findall(r"\{.*?\}", text, flags=re.DOTALL):
                    try:
                        args = json.loads(m)
                        if isinstance(args, dict) and "name" not in args:
                            tool_name = self._pipeline_tools[0]["function"]["name"]
                            calls.append({"name": tool_name, "args": args, "id": "0"})
                            break
                    except Exception:
                        continue

            ak = {"tool_calls": calls} if calls else {}
            msg = AIMessage(content=text, additional_kwargs=ak)
            generations.append(ChatGeneration(message=msg, generation_info=g.generation_info))

        return ChatResult(generations=generations, llm_output=llm_result.llm_output)