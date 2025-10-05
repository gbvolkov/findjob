from datetime import date, datetime
from typing import List, Any
import config

from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
#from langchain_community.llms.yandex import YandexGPT
#from langchain_community.chat_models import ChatYandexGPT
from agents.assistants.yandex_tools.yandex_tooling import ChatYandexGPTWithTools as ChatYandexGPT
from langchain_gigachat import GigaChat

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_community.tools import DuckDuckGoSearchRun

from langchain.agents import initialize_agent, AgentType

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from agents.retrievers.retriever import get_search_tool
from agents.utils import ModelType
from agents.state.state import State

from agents.assistants.hf_tools.chat_local import ChatLocalTools

from palimpsest import Palimpsest
import logging
logger = logging.getLogger(__name__)

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            try:
                result = self.runnable.invoke(state)
            except Exception as e:
                logging.error(f"HTTP error: {e}")
                raise e
            if (
                result.tool_calls
                or result.content
                and (
                    not isinstance(result.content, list)
                    or result.content[0].get("text")
                )
            ):
                break
            messages = state["messages"] + [("user", "Respond with a real output.")]
            state = {**state, "messages": messages}
        return {"messages": result}


def assistant_factory(model: ModelType, role: str = "default", tools=None):

    if tools is None:
        tools = []
    # Haiku is faster and cheaper, but less accurate
    # llm = ChatAnthropic(model="claude-3-haiku-20240307")

    #processor = Palimpsest(verbose=False)

    #def anonymize(text):
    #    return processor.anonimize(text)

    #def deanonymize(text):
    #    return processor.deanonimize(text)

    if role == "sales_manager":
        with open("prompts/working_prompt_sales.txt", encoding="utf-8") as f:
            prompt = f.read()
    elif role == "service_desk":
        with open("prompts/working_prompt.txt", encoding="utf-8") as f:
            prompt = f.read()
    elif role == "supervisor":
        with open("prompts/supervisor_prompt.txt", encoding="utf-8") as f:
            prompt = f.read()
    else:
        with open("prompts/working_prompt_employee.txt", encoding="utf-8") as f:
            prompt = f.read()


    if model == ModelType.MISTRAL:
        llm = ChatMistralAI(model="mistral-large-latest", temperature=1, frequency_penalty=0.3)
    elif model == ModelType.YA:
        #model_name=f'gpt://{config.YA_FOLDER_ID}/yandexgpt/rc'
        model_name=f'gpt://{config.YA_FOLDER_ID}/yandexgpt-32k/rc'

        llm = ChatYandexGPT(
            #iam_token = None,
            api_key = config.YA_API_KEY, 
            folder_id=config.YA_FOLDER_ID, 
            model_uri=model_name,
            temperature=1
            )
    elif model == ModelType.LOCAL:
        with open("prompts/working_prompt_ru_short.txt", encoding="utf-8") as f:
            prompt_txt = f.read()
        prompt = eval(f"f'''{prompt_txt}'''")

        prompt = "Ты бот, который отвечает на вопросы пользователей. Перед ответом извлеки информацию из базы знаний, при помощи инструментов."
        llm = ChatLocalTools(model_id="yandex/YandexGPT-5-Lite-8B-instruct")
    elif model == ModelType.SBER:
        llm = GigaChat(
            credentials=config.GIGA_CHAT_AUTH, 
            model="GigaChat-Pro",
            verify_ssl_certs=False,
            temperature=1,
            scope = config.GIGA_CHAT_SCOPE)
    else:
        llm = ChatOpenAI(model="gpt-5-mini", temperature=1, frequency_penalty=0.3)

    primary_assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=datetime.now)


    #assistant_chain = primary_assistant_prompt | llm.bind_tools(assistant_tools)
    #from agents.assistants.chat_model_wrapper import AnonimizedChatModelProxy, make_anonymized_tool

    #search_kb = get_search_tool(processor)
    if not tools or len(tools) == 0:
        search_kb = get_search_tool()
        assistant_tools = [
            search_kb
        ]
    else:
        assistant_tools = tools

    #assistant_chain = {"messages": lambda txt: anonymize(txt, language="en")} | primary_assistant_prompt | llm.bind_tools(assistant_tools) | (lambda ai_message: deanonymize(ai_message))
    #anon_llm = ChatModelInterceptor(llm, anonymize, deanonymize)
    tooled_llm = llm.bind_tools(assistant_tools)
    #anon_llm = AnonimizedChatModelProxy(tooled_llm, anonymize, deanonymize)

    assistant_chain = primary_assistant_prompt | tooled_llm

    return assistant_chain, assistant_tools