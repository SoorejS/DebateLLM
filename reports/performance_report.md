# 📊 DebateLLM Fallacy Detector Performance Report

**Model Evaluated:** `RowdyI7er/DebateLLM`
**Base Model:** `deberta-base`
**Task:** Logical Fallacy Detection
**Date:** 2026-03-29
**Time Taken for Inference (32 items):** 2.24 seconds

## 🎯 Overall Performance

**Test Examples:** 32
**Correct Predictions:** 29
**Overall Accuracy:** 90.62%

## 📊 Class Breakdown

| Fallacy Category | Accuracy | Correct | Total |
| :--- | :---: | :---: | :---: |
| Ad Hominem | 100.0% | 4 | 4 |
| Appeal To Authority | 75.0% | 3 | 4 |
| Bandwagon | 100.0% | 4 | 4 |
| False Dilemma | 75.0% | 3 | 4 |
| Hasty Generalization | 100.0% | 4 | 4 |
| No Fallacy | 75.0% | 3 | 4 |
| Slippery Slope | 100.0% | 4 | 4 |
| Strawman | 100.0% | 4 | 4 |

## 📝 Detailed Predictions

| Statement | True Label | Predicted Label | Result |
| :--- | :--- | :--- | :---: |
| You can't trust John's argument on tax reform; he's just a greedy billionaire who doesn't care about the poor. | Ad Hominem | Ad Hominem | ✅ |
| Why should we listen to a high school dropout about climate change policy? | Ad Hominem | Ad Hominem | ✅ |
| Of course you think teachers should be paid more, you're a teacher yourself. Your opinion is completely biased. | Ad Hominem | Ad Hominem | ✅ |
| The senator's plan for healthcare is ridiculous, considering he was involved in a scandal ten years ago. | Ad Hominem | Ad Hominem | ✅ |
| My favorite actor buys this brand of car, so it must be the best vehicle on the market. | Appeal To Authority | Bandwagon | ❌ |
| A famous pop star said that drinking lemon water cures all illnesses, so I'm throwing away my medicine. | Appeal To Authority | Appeal To Authority | ✅ |
| The CEO of a tech company said we shouldn't worry about the environment, so global warming is clearly a myth. | Appeal To Authority | Appeal To Authority | ✅ |
| Because Dr. Smith, a dentist, says this new stock is going to double, I should invest all my money in it. | Appeal To Authority | Appeal To Authority | ✅ |
| Everyone I know is voting for Smith, so he must be the best candidate for the job. | Bandwagon | Bandwagon | ✅ |
| Millions of people have downloaded this app, so it must be incredibly useful and safe. | Bandwagon | Bandwagon | ✅ |
| You should definitely start eating a paleo diet; literally everyone at the gym is doing it right now. | Bandwagon | Bandwagon | ✅ |
| Nobody wears those kinds of jeans anymore, you need to buy the new style if you want to fit in. | Bandwagon | Bandwagon | ✅ |
| We either completely ban all video games, or we let our children become violent criminals. | False Dilemma | False Dilemma | ✅ |
| If you don't support my proposed tax cut, then you clearly hate hard-working Americans. | False Dilemma | Ad Hominem | ❌ |
| You're either with us and support our actions, or you're against us and side with the enemy. | False Dilemma | False Dilemma | ✅ |
| We must cut funding to the arts entirely, or the city will go completely bankrupt by next year. | False Dilemma | False Dilemma | ✅ |
| I met two people from New York and they were both rude. Everyone from New York must be terrible. | Hasty Generalization | Hasty Generalization | ✅ |
| My grandfather smoked a pack a day and lived to be 90, so smoking isn't actually bad for your health. | Hasty Generalization | Hasty Generalization | ✅ |
| The first book I read by this author was boring, so all of her books must be completely unreadable. | Hasty Generalization | Hasty Generalization | ✅ |
| A self-driving car crashed last week. Self-driving technology is completely unsafe and will never work. | Hasty Generalization | Hasty Generalization | ✅ |
| If we allow students to use phones in the hallway, soon they'll be using them during exams, and eventually nobody will learn anything. | Slippery Slope | Slippery Slope | ✅ |
| If we pass this mild gun control law, next they will confiscate all our hunting rifles, and then we will live in a dictatorship. | Slippery Slope | Slippery Slope | ✅ |
| If I let you borrow my pen, you'll lose it. Then you'll need to borrow my paper, and soon I'll be doing all your homework for you. | Slippery Slope | Slippery Slope | ✅ |
| If we approve this small property tax increase, soon taxes will be so high that everyone will be forced to sell their homes. | Slippery Slope | Slippery Slope | ✅ |
| You say we should increase funding for education, but I don't think throwing unlimited money at schools while ignoring the economy is a good idea. | Strawman | Strawman | ✅ |
| My opponent wants to expand public transit. It's ridiculous that he wants to force everyone to give up their cars and ride buses. | Strawman | Strawman | ✅ |
| You think we should be more relaxed about the dress code, so you basically want students to show up in their pajamas. | Strawman | Strawman | ✅ |
| Since you don't support the new military budget, you clearly want our country to be defenseless against foreign attacks. | Strawman | Strawman | ✅ |
| The study measured the heart rates of 500 participants before and after exercise, showing a statistically significant increase. | No Fallacy | No Fallacy | ✅ |
| Based on the quarterly financial report, the company's revenue has decreased by 5% compared to last year. | No Fallacy | No Fallacy | ✅ |
| The new law requires all drivers to renew their licenses every ten years to ensure they are still fit to drive safely. | No Fallacy | Bandwagon | ❌ |
| According to meteorological data collected from satellites, there is an 80% chance of heavy rainfall tomorrow evening. | No Fallacy | No Fallacy | ✅ |
