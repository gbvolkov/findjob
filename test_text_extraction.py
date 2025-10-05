import trafilatura

html = trafilatura.fetch_url("https://habr.com/ru/companies/otus/articles/542144/")
text = trafilatura.extract(html)

print(text)