import time
import requests
import base64
from xml.etree import ElementTree as ET

import config

API_KEY = config.YA_API_KEY
FOLDER_ID = config.YA_FOLDER_ID
QUERY = "кофемашина"

endpoint = "https://searchapi.api.cloud.yandex.net/v2/web/search"
headers = {
    "Authorization": f"Api-Key {API_KEY}",
    "Content-Type": "application/json",
}
payload = {
    "query": {
        "searchType": "SEARCH_TYPE_RU",
        "queryText": QUERY,
        "page": 0,
        "fixTypoMode": "FIX_TYPO_MODE_ON"
    },
    "folderId": FOLDER_ID,
    "responseFormat": "FORMAT_XML",  # или FORMAT_HTML
    "userAgent": "Mozilla/5.0"
}

resp = requests.post(endpoint, json=payload, headers=headers)
resp.raise_for_status()
data = resp.json()

# Декодируем Base64 с результатом
raw = data["rawData"]  # замените в зависимости от формы ответа
decoded = base64.b64decode(raw).decode("utf-8")

print("RAW decoded response:")
print(decoded)

# Если XML → можно распарсить:
root = ET.fromstring(decoded)
for doc in root.findall(".//doc"):
    title = doc.findtext("title")
    url = doc.findtext("url")
    snippet = doc.findtext("passages/passage")
    print(f"- {title}\n  {snippet}\n  {url}\n")
