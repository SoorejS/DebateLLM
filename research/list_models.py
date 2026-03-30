from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(author="RowdyI7er")
for m in models:
    print(m.modelId)
