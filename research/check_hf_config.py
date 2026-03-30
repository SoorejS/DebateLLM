import requests

url = "https://huggingface.co/RowdyI7er/DebateLLM/raw/main/config.json"
r = requests.get(url)
print("STATUS:", r.status_code)
if r.status_code == 200:
    print(r.text)
