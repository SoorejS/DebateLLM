from huggingface_hub import list_repo_files
files = list_repo_files("RowdyI7er/DebateLLM")
with open("repo_files.txt", "w") as f:
    for file in files:
        f.write(f"{file}\n")
