import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

test_data = [
    # Ad Hominem
    {"text": "You can't trust John's argument on tax reform; he's just a greedy billionaire who doesn't care about the poor.", "true_label": "ad_hominem"},
    {"text": "Why should we listen to a high school dropout about climate change policy?", "true_label": "ad_hominem"},
    {"text": "Of course you think teachers should be paid more, you're a teacher yourself. Your opinion is completely biased.", "true_label": "ad_hominem"},
    {"text": "The senator's plan for healthcare is ridiculous, considering he was involved in a scandal ten years ago.", "true_label": "ad_hominem"},
    
    # Appeal to Authority
    {"text": "My favorite actor buys this brand of car, so it must be the best vehicle on the market.", "true_label": "appeal_to_authority"},
    {"text": "A famous pop star said that drinking lemon water cures all illnesses, so I'm throwing away my medicine.", "true_label": "appeal_to_authority"},
    {"text": "The CEO of a tech company said we shouldn't worry about the environment, so global warming is clearly a myth.", "true_label": "appeal_to_authority"},
    {"text": "Because Dr. Smith, a dentist, says this new stock is going to double, I should invest all my money in it.", "true_label": "appeal_to_authority"},
    
    # Bandwagon
    {"text": "Everyone I know is voting for Smith, so he must be the best candidate for the job.", "true_label": "bandwagon"},
    {"text": "Millions of people have downloaded this app, so it must be incredibly useful and safe.", "true_label": "bandwagon"},
    {"text": "You should definitely start eating a paleo diet; literally everyone at the gym is doing it right now.", "true_label": "bandwagon"},
    {"text": "Nobody wears those kinds of jeans anymore, you need to buy the new style if you want to fit in.", "true_label": "bandwagon"},
    
    # False Dilemma
    {"text": "We either completely ban all video games, or we let our children become violent criminals.", "true_label": "false_dilemma"},
    {"text": "If you don't support my proposed tax cut, then you clearly hate hard-working Americans.", "true_label": "false_dilemma"},
    {"text": "You're either with us and support our actions, or you're against us and side with the enemy.", "true_label": "false_dilemma"},
    {"text": "We must cut funding to the arts entirely, or the city will go completely bankrupt by next year.", "true_label": "false_dilemma"},
    
    # Hasty Generalization
    {"text": "I met two people from New York and they were both rude. Everyone from New York must be terrible.", "true_label": "hasty_generalization"},
    {"text": "My grandfather smoked a pack a day and lived to be 90, so smoking isn't actually bad for your health.", "true_label": "hasty_generalization"},
    {"text": "The first book I read by this author was boring, so all of her books must be completely unreadable.", "true_label": "hasty_generalization"},
    {"text": "A self-driving car crashed last week. Self-driving technology is completely unsafe and will never work.", "true_label": "hasty_generalization"},
    
    # Slippery Slope
    {"text": "If we allow students to use phones in the hallway, soon they'll be using them during exams, and eventually nobody will learn anything.", "true_label": "slippery_slope"},
    {"text": "If we pass this mild gun control law, next they will confiscate all our hunting rifles, and then we will live in a dictatorship.", "true_label": "slippery_slope"},
    {"text": "If I let you borrow my pen, you'll lose it. Then you'll need to borrow my paper, and soon I'll be doing all your homework for you.", "true_label": "slippery_slope"},
    {"text": "If we approve this small property tax increase, soon taxes will be so high that everyone will be forced to sell their homes.", "true_label": "slippery_slope"},
    
    # Strawman
    {"text": "You say we should increase funding for education, but I don't think throwing unlimited money at schools while ignoring the economy is a good idea.", "true_label": "strawman"},
    {"text": "My opponent wants to expand public transit. It's ridiculous that he wants to force everyone to give up their cars and ride buses.", "true_label": "strawman"},
    {"text": "You think we should be more relaxed about the dress code, so you basically want students to show up in their pajamas.", "true_label": "strawman"},
    {"text": "Since you don't support the new military budget, you clearly want our country to be defenseless against foreign attacks.", "true_label": "strawman"},
    
    # No Fallacy
    {"text": "The study measured the heart rates of 500 participants before and after exercise, showing a statistically significant increase.", "true_label": "no_fallacy"},
    {"text": "Based on the quarterly financial report, the company's revenue has decreased by 5% compared to last year.", "true_label": "no_fallacy"},
    {"text": "The new law requires all drivers to renew their licenses every ten years to ensure they are still fit to drive safely.", "true_label": "no_fallacy"},
    {"text": "According to meteorological data collected from satellites, there is an 80% chance of heavy rainfall tomorrow evening.", "true_label": "no_fallacy"}
]

print("Loading model and tokenizer...")
model_name = "RowdyI7er/DebateLLM"
tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="debate_fallacy_model")
model = AutoModelForSequenceClassification.from_pretrained(model_name, subfolder="debate_fallacy_model")

# If CUDA available, use it
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

results = []

print(f"Running inference on {len(test_data)} examples...")
start_time = time.time()

for item in test_data:
    inputs = tokenizer(item["text"], return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    predicted_label = model.config.id2label[predicted_class_id]
    
    results.append({
        "text": item["text"],
        "true_label": item["true_label"],
        "predicted_label": predicted_label,
        "is_correct": predicted_label == item["true_label"]
    })

end_time = time.time()

print("Evaluation complete. Saving results...")
with open("evaluation_results.json", "w") as f:
    json.dump({"results": results, "time_taken": end_time - start_time}, f, indent=4)
print("Done!")
