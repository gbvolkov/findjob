# chat_local_tools.py  ── final version, adopting `conversation=` style
from __future__ import annotations
import copy, json, logging, re, uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch
from pydantic import PrivateAttr
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool



# ─────────────────────── regex for tool‑calls ───────────────────── #
_HERMES_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
#_YANDEX_RE = re.compile(r"\[TOOL_CALL_START\](\w+)\s*\n\s*(\{.*?\})", re.DOTALL)
_YANDEX_RE = re.compile(r"\[TOOL_CALL_START\]\s*(\w+)\s*\r?\n\s*(\{.*?\})",  re.DOTALL)

# ───────────────────── helpers ───────────────────── #
def _coerce_content(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return " ".join(
            (p["text"] if isinstance(p, dict) and "text" in p else str(p))
            for p in raw
        )
    return str(raw)


def _msgs_to_hf(msgs: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in msgs:
        if isinstance(m, SystemMessage):
            #out.append({"role": "system", "content": _coerce_content(m.content)})
            out.append({"role": "user", "content": _coerce_content(m.content)})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": _coerce_content(m.content)})
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": _coerce_content(m.content), "name": m.name})
        elif isinstance(m, AIMessage):
            entry: Dict[str, Any] = {
                "role": "assistant",
                "content": _coerce_content(m.content or ""),
            }
            entry.update(m.additional_kwargs)          # keep function_call / tool_calls
            out.append(entry)
        else:
            raise ValueError(f"Unknown message type: {type(m)}")
    return out


# ─────────────────────── main class ─────────────────────── #
class ChatLocalTools(BaseChatModel):
    """Local Causal‑LM chat wrapper with .bind_tools and automatic tool‑calling."""

    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _tool_registry: Dict[str, Union[BaseTool, callable]] = PrivateAttr(default_factory=dict)
    _tools_schema: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _gen_cfg: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _max_calls: int = PrivateAttr()
    _verbose: bool = PrivateAttr()

    # ───────────────── init ───────────────── #
    def __init__(
        self,
        model_id: str = "yandex/YandexGPT-5-Lite-8B-instruct",
        tools: Optional[Sequence[BaseTool | callable]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        device_map: str | Mapping[str, int] = "auto",
        torch_dtype: str | torch.dtype | None = "auto",
        trust_remote_code: bool = True,
        max_tool_calls: int = 3,
        verbose: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id

        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        #generation_config = GenerationConfig.from_pretrained(model_id)
        #generation_config.max_new_tokens = 24*1024
        #generation_config.temperature = 1
        #generation_config.top_p = 0.9
        #generation_config.do_sample = True
        #generation_config.repetition_penalty = 1.2
        #generation_config.eos_token_id=tok.eos_token_id,
        #generation_config.pad_token_id=tok.eos_token_id
        
        generation_config = {"max_new_tokens": 4*1024, "temperature": 1}


        object.__setattr__(self, "_tokenizer", tok)
        object.__setattr__(self, "_model", mdl)
        object.__setattr__(self, "_gen_cfg", generation_kwargs or generation_config) #{"max_new_tokens": 24*1024})
        object.__setattr__(self, "_max_calls", max_tool_calls)
        object.__setattr__(self, "_verbose", verbose)

        self._register_tools(tools or [])

    # ───────── tool registry / bind_tools ───────── #
    def _register_tools(self, tools: Sequence[BaseTool | callable]) -> None:
        reg = dict(getattr(self, "_tool_registry", {}))
        for t in tools:
            reg[t.name if isinstance(t, BaseTool) else t.__name__] = t
        schema = [convert_to_openai_tool(t) for t in reg.values()]
        object.__setattr__(self, "_tool_registry", reg)
        object.__setattr__(self, "_tools_schema", schema)

    def bind_tools(self, tools: Sequence[BaseTool | callable]) -> "ChatLocalTools":
        clone: "ChatLocalTools" = copy.copy(self)
        clone._register_tools(tools)
        return clone

    # ───────── BaseChatModel plumbing ───────── #
    def _get_token_ids(self, text: str) -> List[int]:
        return self._tokenizer.encode(text)

    @property
    def _llm_type(self) -> str:
        return "chat_local_tools"

    async def _agenerate(self, messages: List[BaseMessage], stop=None, **kw):
        return self._generate(messages, stop=stop, **kw)

    # ───────── the main generate loop ───────── #
    def _generate(self, messages: List[BaseMessage], stop=None, **kw) -> ChatResult:
        #messages = [HumanMessage(content=msg.content) if isinstance(msg, SystemMessage) else msg for msg in messages]
        history = list(messages)
        remaining = self._max_calls

        while True:
            # ❶ Build input dict the HF way (your template)
            enc = self._tokenizer.apply_chat_template(
                conversation=_msgs_to_hf(history),
                tools=self._tools_schema or None,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            if self._verbose:
                prompt_txt = self._tokenizer.decode(enc[0].ids, skip_special_tokens=True)
                logging.debug(f"prompt raw → {prompt_txt}")
            enc = {k: v.to(self._model.device) for k, v in enc.items()}

            # ❷ Generate
            with torch.no_grad():
                out_ids = self._model.generate(
                    **enc,
                    eos_token_id=self._tokenizer.eos_token_id,
                    pad_token_id=self._tokenizer.pad_token_id,
                    **self._gen_cfg,
                )[0]

            gen_ids = out_ids[enc["input_ids"].shape[-1] :]
            text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
            if self._verbose:
                logging.debug("LLM raw → %s", text)

            # ❸ Detect tool‑call
            tool_data = None
            m = _HERMES_RE.search(text)
            if m:
                tool_data = json.loads(m.group(1))
            else:
                m = _YANDEX_RE.search(text)
                if m:
                    tool_data = {
                        "name": m.group(1).strip(),
                        "arguments": json.loads(m.group(2).replace("'", '"')),
                    }

            # ❹ Execute if present
            if tool_data and remaining > 0:
                remaining -= 1
                name, args = tool_data["name"], tool_data.get("arguments", {})
                tool_fn = self._tool_registry.get(name)
                if tool_fn is None:
                    logging.warning("Unknown tool: %s", name)
                    break
                try:
                    if isinstance(tool_fn, BaseTool):
                        result = tool_fn.invoke(args)          # ⬅️ ключевая строка
                    else:
                        result = tool_fn(**args)
                except Exception as e:
                    logging.exception("Tool %s failed: %s", name, e)
                    result = f"ERROR: {e}"

                call_id = str(uuid.uuid4())
                history.extend(
                    [
                        AIMessage(
                            content="",
                            additional_kwargs={
                                "function_call": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                                "tool_calls": [
                                    {
                                        "id": call_id,
                                        "type": "function",
                                        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                                    }
                                ],
                            },
                        ),
                        ToolMessage(name=name, content=str(result), tool_call_id=call_id),
                    ]
                )
                if self._verbose:
                    logging.debug("Tool %s → %s", name, result)
                continue  # regenerate final answer

            # ❺ Return to caller
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
