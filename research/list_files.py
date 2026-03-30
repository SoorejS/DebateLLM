from huggingface_hub import list_repo_files
files = list_repo_files("RowdyI7er/DebateLLM")
print("FILES:", files)
