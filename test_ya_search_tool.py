from agents.tools.yandex_search import YandexSearchTool

import config


yandex_tool = YandexSearchTool(
    api_key=config.YA_API_KEY,
    folder_id=config.YA_FOLDER_ID,
    max_results=3,
    max_size = 2048
)

print(yandex_tool.run("Как отформатировать диск в OS360?"))