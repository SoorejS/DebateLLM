import traceback

try:
    from transformers import AutoConfig, AutoModelForSequenceClassification
    model_name = "RowdyI7er/DebateLLM"
    config = AutoConfig.from_pretrained(model_name)
    print("LABELS_COUNT:", len(config.id2label))
    print("LABELS:", config.id2label)
except Exception as e:
    print("ERROR:", traceback.format_exc())
