# 📊 DebateLLM 300-Statement Evaluation Report

## 🎯 Executive Summary

The `RowdyI7er/DebateLLM` model was subjected to a rigorous test of **300 unique statements** across 8 logical fallacy categories.

- **Overall Accuracy:** 82.00% (246/300 correct)
- **Total Inference Time:** 6.36 seconds
- **Evaluation Date:** 2026-03-30
- **Test Set Diversity:** 8 categories, ~37-38 statements per category.

---

## 📈 Performance by Category

| Fallacy Category | Accuracy | Correct | Total | Top Confusion |
| :--- | :---: | :---: | :---: | :--- |
| Ad Hominem | 86.8% | 33 | 38 | Strawman (5) |
| Appeal To Authority | 97.4% | 37 | 38 | Hasty Generalization (1) |
| Bandwagon | 92.1% | 35 | 38 | No Fallacy (2) |
| False Dilemma | 100.0% | 37 | 37 | None |
| Hasty Generalization | 86.5% | 32 | 37 | Ad Hominem (4) |
| No Fallacy | 15.8% | 6 | 38 | Ad Hominem (12) |
| Slippery Slope | 78.4% | 29 | 37 | False Dilemma (8) |
| Strawman | 100.0% | 37 | 37 | None |

---

## 🔍 Detailed Error Analysis

The following table highlights specific examples where the model's prediction differed from the ground truth. These insights help identify potential over-generalization or subtle nuances the model may be missing.

| Statement | True Label | Predicted Label |
| :--- | :--- | :--- |
| Of course you want a raise; you're just greed personified. | Ad Hominem | Strawman |
| She's just trying to get attention with these radical ideas. | Ad Hominem | Strawman |
| He's a puppet for the corporate elites. | Ad Hominem | Strawman |
| You're only arguing this point to make yourself look smart. | Ad Hominem | Strawman |
| He's just a shill for the pharmaceutical companies. | Ad Hominem | Strawman |
| My grandfather, who lived to 100, said to always eat raw onions. | Appeal To Authority | Hasty Generalization |
| Jump on the bandwagon and support the winning team! | Bandwagon | False Dilemma |
| The most-watched show on Netflix is always the highest quality. | Bandwagon | No Fallacy |
| The popular vote shows that people want this law. | Bandwagon | No Fallacy |
| He wore a red shirt today and he's mean, so people in red shirts are mean. | Hasty Generalization | Ad Hominem |
| I met a quiet student, so all students must be introverts. | Hasty Generalization | Ad Hominem |
| I met a shy actor, so all actors are actually introverts. | Hasty Generalization | Ad Hominem |
| I saw a person with tattoos acting rudely, so people with tattoos are all rude. | Hasty Generalization | Ad Hominem |
| I met a person who likes pineapple on pizza, so clearly everyone loves it. | Hasty Generalization | Bandwagon |
| I need to go to the store to buy milk. | No Fallacy | False Dilemma |

---

## 🛠 Model Metadata

- **Hugging Face Path:** `RowdyI7er/DebateLLM` (subfolder: `debate_fallacy_model`)
- **Architecture:** `DeBERTa-v3` (Base)
- **Batch Size:** 16
- **Device Used:** cpu
