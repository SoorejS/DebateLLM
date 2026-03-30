import requests
url = "https://huggingface.co/api/models/RowdyI7er/DebateLLM"
r = requests.get(url)
print(r.status_code)
print(r.text)
