import requests
url = "https://huggingface.co/RowdyI7er/DebateLLM/resolve/main/config.json"
try:
    response = requests.get(url)
    print(response.json())
except Exception as e:
    print(e)
