import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from agents.assistants.hf_tools.chat_local import ChatLocalTools
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.messages import HumanMessage

from langchain_core.tools import tool
from datetime import date, datetime
from agents.retrievers.retriever import search_kb

import os
import config


os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


@tool
def get_current_temperature(location: str, unit: str) -> float:
    """
    Get the current temperature at a location.
    
    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    Returns:
        The current temperature at the specified location in the specified units, as a float.
    """
    return 22.

llm = ChatLocalTools(model_id="yandex/YandexGPT-5-Lite-8B-instruct", verbose=True)

assistant_tools = [
        search_kb,
        get_current_temperature,
    ]

#prompt = "You are a bot that responds to user's queries. Please try to retrieve information form knowledgebase before answering."

with open("prompts/working_prompt_ru_short.txt", encoding="utf-8") as f:
    prompt = f.read()
#prompt = "Ты бот, который отвечает на вопросы пользователей. Разбивай вопрос пользователя на отдельные смысловые куски и используй каждый смысловой кусок для запроса соответственного инструмента. Если первые запросы не дали результата, расширяйте поисковый запрос (синонимы, другие формулировки)."

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

assistant_chain = primary_assistant_prompt | llm.bind_tools(assistant_tools)
#assistant_chain = llm.bind_tools(assistant_tools)

#messages = [
#  {"role": "system", "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location."},
#  {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
#]


msg = [HumanMessage(content="Мне надо обратиться к кей юзерам за поддержкой, а я не знаю, кто это. Поможешь, бро? И подскажи, какая там у них в городах температура сейчас?")]
#msg = [HumanMessage(content="Кто такие кей юзеры?")]
#msg = [HumanMessage(content="Ты бот, который отвечает на вопросы пользователей. Перед ответом извлеки информацию из базы знаний, при помощи инструментов. Ты так же умеешь получать температуру в определённой локации.\n\nКто такие кей юзеры?")]
answer = assistant_chain.invoke({"messages": msg})
#answer = assistant_chain.invoke(msg)

print(answer.content)