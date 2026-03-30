import json

with open("evaluation_results.json", "r") as f:
    data = json.load(f)

results = data["results"]
time_taken = data["time_taken"]

y_true = [item["true_label"] for item in results]
y_pred = [item["predicted_label"] for item in results]

correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
total = len(y_true)
accuracy = correct / total * 100

from collections import Counter
true_counts = Counter(y_true)
correct_counts = Counter()
for yt, yp in zip(y_true, y_pred):
    if yt == yp:
        correct_counts[yt] += 1

markdown_content = f"""# 📊 DebateLLM Fallacy Detector Performance Report

**Model Evaluated:** `RowdyI7er/DebateLLM`
**Base Model:** `deberta-base`
**Task:** Logical Fallacy Detection
**Date:** 2026-03-29
**Time Taken for Inference ({total} items):** {time_taken:.2f} seconds

## 🎯 Overall Performance

**Test Examples:** {total}
**Correct Predictions:** {correct}
**Overall Accuracy:** {accuracy:.2f}%

## 📊 Class Breakdown

| Fallacy Category | Accuracy | Correct | Total |
| :--- | :---: | :---: | :---: |
"""

for label in sorted(true_counts.keys()):
    t_cnt = true_counts[label]
    c_cnt = correct_counts[label]
    acc = c_cnt / t_cnt * 100
    markdown_content += f"| {label.replace('_', ' ').title()} | {acc:.1f}% | {c_cnt} | {t_cnt} |\n"

markdown_content += "\n## 📝 Detailed Predictions\n\n"
markdown_content += "| Statement | True Label | Predicted Label | Result |\n"
markdown_content += "| :--- | :--- | :--- | :---: |\n"

for item in results:
    text = item['text'].replace('|', '&#124;') # escape markdown tables
    true_label = item['true_label'].replace('_', ' ').title()
    pred_label = item['predicted_label'].replace('_', ' ').title()
    status = "✅" if item['is_correct'] else "❌"
    markdown_content += f"| {text} | {true_label} | {pred_label} | {status} |\n"

with open("performance_report.md", "w", encoding="utf-8") as f:
    f.write(markdown_content)

print("Report generated: performance_report.md")
