import time
from agents.classifier import summarise_request, classify_request
import os
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


user_queries =  [
    #"Какой порядок выхода в отпуск?",
    #"Какие виды субсидий МПТ существуют?",
    #"Как настроить калькулятор для коммерческого менеджера?",
    #"Какая погода в Бомбее?",
    "как изменить источник привлечения?",
    "",
    "как изменить источник привлечения в системе у заявки?\nРоль пользователя: техподдержка.",
    "",
    "как оформить отпуск?\nРоль пользователя: сотрудник.",
    "",
    "как оформить отпуск?\nРоль пользователя: техподдержка.",
    "",
    "как оформить отпуск?\nРоль пользователя: продажник.",
]

query_str = ""
for query in user_queries:
    if query == "":
        query_str = ""
        continue
    query_str = query_str + ("" if query_str=="" else ";") + query
    final_query = summarise_request(query_str, maxlen=8)
    query_class = classify_request(final_query)
    print(f"{query}:{query_str}:{final_query}:{query_class}")
    time.sleep(1)