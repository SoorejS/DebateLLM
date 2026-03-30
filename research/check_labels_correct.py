from transformers import AutoConfig

model_name = "RowdyI7er/DebateLLM"
config = AutoConfig.from_pretrained(model_name, subfolder="debate_fallacy_model")
print("LABELS:", config.id2label)
