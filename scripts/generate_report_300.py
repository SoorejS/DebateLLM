import json
from collections import Counter, defaultdict

# Load results
try:
    with open("evaluation_results_300.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: evaluation_results_300.json not found.")
    exit(1)

results = data["results"]
time_taken = data["time_taken"]
total = len(results)

# Metrics
correct = sum(1 for item in results if item["is_correct"])
accuracy = (correct / total) * 100

# Per-class metrics
class_stats = defaultdict(lambda: {"correct": 0, "total": 0, "confusions": Counter()})
for item in results:
    true_label = item["true_label"]
    pred_label = item["predicted_label"]
    
    class_stats[true_label]["total"] += 1
    if item["is_correct"]:
        class_stats[true_label]["correct"] += 1
    else:
        class_stats[true_label]["confusions"][pred_label] += 1

# Generate Report
report = f"""# 📊 DebateLLM 300-Statement Evaluation Report

## 🎯 Executive Summary

The `RowdyI7er/DebateLLM` model was subjected to a rigorous test of **300 unique statements** across 8 logical fallacy categories.

- **Overall Accuracy:** {accuracy:.2f}% ({correct}/{total} correct)
- **Total Inference Time:** {time_taken:.2f} seconds
- **Evaluation Date:** 2026-03-30
- **Test Set Diversity:** 8 categories, ~37-38 statements per category.

---

## 📈 Performance by Category

| Fallacy Category | Accuracy | Correct | Total | Top Confusion |
| :--- | :---: | :---: | :---: | :--- |
"""

sorted_labels = sorted(class_stats.keys())
for label in sorted_labels:
    stats = class_stats[label]
    acc = (stats["correct"] / stats["total"]) * 100
    top_confusion = "None"
    if stats["confusions"]:
        conf, count = stats["confusions"].most_common(1)[0]
        top_confusion = f"{conf.replace('_', ' ').title()} ({count})"
    
    report += f"| {label.replace('_', ' ').title()} | {acc:.1f}% | {stats['correct']} | {stats['total']} | {top_confusion} |\n"

report += """
---

## 🔍 Detailed Error Analysis

The following table highlights specific examples where the model's prediction differed from the ground truth. These insights help identify potential over-generalization or subtle nuances the model may be missing.

| Statement | True Label | Predicted Label |
| :--- | :--- | :--- |
"""

# Show up to 15 misclassifications
misclassified = [item for item in results if not item["is_correct"]]
for item in misclassified[:15]:
    text = item["text"].replace("|", "&#124;")
    true_l = item["true_label"].replace("_", " ").title()
    pred_l = item["predicted_label"].replace("_", " ").title()
    report += f"| {text} | {true_l} | {pred_l} |\n"

report += f"""
---

## 🛠 Model Metadata

- **Hugging Face Path:** `RowdyI7er/DebateLLM` (subfolder: `debate_fallacy_model`)
- **Architecture:** `DeBERTa-v3` (Base)
- **Batch Size:** {data.get('batch_size', 'N/A')}
- **Device Used:** {data.get('device', 'cpu')}
"""

with open("performance_report_300.md", "w", encoding="utf-8") as f:
    f.write(report)

print("Report successfully generated: performance_report_300.md")
