with open("prompts/working_prompt_sales.txt", encoding="utf-8") as f:
    sm_prompt = f.read()
with open("prompts/working_prompt.txt", encoding="utf-8") as f:
    sd_prompt = f.read()
with open("prompts/working_prompt_employee.txt", encoding="utf-8") as f:
    default_prompt = f.read()
with open("prompts/supervisor_prompt.txt", encoding="utf-8") as f:
    sv_prompt = f.read()

with open("prompts/search_web_prompt.txt", encoding="utf-8") as f:
    sd_agent_web_prompt = f.read()
with open("prompts/search_web_prompt_sales.txt", encoding="utf-8") as f:
    sm_agent_web_prompt = f.read()
with open("prompts/search_web_prompt_employee.txt", encoding="utf-8") as f:
    default_search_web_prompt = f.read()
