from transformers import AutoConfig

model_name = "RowdyI7er/DebateLLM"
config = AutoConfig.from_pretrained(model_name, subfolder="debate_fallacy_model")
with open("labels.txt", "w") as f:
    f.write(str(config.id2label))
