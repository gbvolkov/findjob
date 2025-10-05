from agents.assistants.hf_tools.chat_local import ChatLocalTools
from langchain_core.messages import HumanMessage
import os
from langchain_core.tools import tool
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

@tool
def get_wheather_obserwation(location: str) -> str:
    """
    Get obserwation of the current weather at a location.
    
    Args:
        location: The location to get the weather for, in the format "City, Country"
    Returns:
        The obserwation of current weather at the specified location as string.
    """
    return "It's sunny an windy."

llm = (
    ChatLocalTools(model_id="yandex/YandexGPT-5-Lite-8B-instruct")
    .bind_tools([get_current_temperature, get_wheather_obserwation])
)

messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries. On questions on temperature you should reply with the unit used in the queried location."},
  {"role": "user", "content": "Hey, what's the weather in Paris right now? And what is the temperature?"}
]

answer = llm.invoke(messages)
print(answer.content)