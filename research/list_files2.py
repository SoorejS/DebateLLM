from huggingface_hub import list_repo_files
files = list_repo_files("RowdyI7er/DebateLLM")
print("FILES:", [f for f in files if f.startswith("fallacy_model")])
