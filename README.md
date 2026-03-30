# DebateLLM Fallacy Detector - Evaluation Suite

This is the evaluation framework for the **DebateLLM Fallacy Detector**, a DeBERTa-base model specialized in identifying logical fallacies in text. Developed as part of my **GSoC (Google Summer of Code)** project.

## 🚀 Project Overview

The primary aim of this suite is to rigorously evaluate the performance of [RowdyI7er/DebateLLM](https://huggingface.co/RowdyI7er/DebateLLM). It includes tools for:
1.  **Small-Scale Verification**: Initial assessment of the model with 32 statements.
2.  **Large-Scale Performance Benchmarking**: A comprehensive test using 300 unique logical fallacy statements.
3.  **Detailed Reporting**: Automated generation of performance metrics, class breakdowns, and confusion analysis.

## 📋 Detected Fallacies

The model covers the following 8 categories:
- Ad Hominem
- Appeal to Authority
- Bandwagon
- False Dilemma
- Hasty Generalization
- No Fallacy (Standard Factual Statement)
- Slippery Slope
- Strawman

## 📁 Repository Structure

```text
├── scripts/             # Main evaluation and report generation logic.
│   ├── evaluate_300.py   # Large-scale (300 statements) inferencing script.
│   └── ...
├── reports/             # Detailed Markdown performance reports.
│   └── performance_report_300.md
├── data/                # Results and label mappings.
│   └── evaluation_results_300.json
└── research/            # Investigation and debug history.
```

## 🛠 Quick Start

### Installation

```bash
pip install torch transformers tqdm
```

### Running the 300-Statement Evaluation

To replicate the performance test and generate a fresh report:

1.  **Execute the Evaluation**:
    ```bash
    python scripts/evaluate_300.py
    ```
2.  **Generate the Report**:
    ```bash
    python scripts/generate_report_300.py
    ```

Check the `reports/` folder for the result.
