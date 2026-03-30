import traceback

try:
    from transformers import AutoConfig, AutoModelForSequenceClassification
    model_name = "RowdyI7er/DebateLLM"
    config = AutoConfig.from_pretrained(model_name)
    print("SUCCESS")
except Exception as e:
    with open("error.txt", "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())
