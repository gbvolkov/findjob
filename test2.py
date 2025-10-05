from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def get_current_temperature(location: str, unit: str) -> float:
    """
    Get the current temperature at a location.
    
    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    Returns:
        The current temperature at the specified location in the specified units, as a float.
    """
    return 22.  # A real function should probably actually get the temperature!

def get_current_wind_speed(location: str) -> float:
    """
    Get the current wind speed in km/h at a given location.
    
    Args:
        location: The location to get the temperature for, in the format "City, Country"
    Returns:
        The current wind speed at the given location in km/h, as a float.
    """
    return 6.  # A real function should probably actually get the wind speed!

tools = [get_current_temperature, get_current_wind_speed]

model_id="yandex/YandexGPT-5-Lite-8B-instruct"
#model_id="NousResearch/Hermes-2-Pro-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print(tokenizer.chat_template)


model     = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")

messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location."},
  {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
]

inputs = tokenizer.apply_chat_template(
    conversation=messages,        # note the keyword is `conversation`
    tools=tools,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = {k: v for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):]))

messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location."},
  {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
]
tool_call = {"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]})

inputs = tokenizer.apply_chat_template(
    conversation=messages,        # note the keyword is `conversation`
    tools=tools,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = {k: v for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):]))
