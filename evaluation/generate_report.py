
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "evaluation"

FINAL_FILE = DATA_DIR / "final_evaluation.csv"
REPORT_FILE = DATA_DIR / "evaluation_report.md"

# Load existing evaluation results
df = pd.read_csv(FINAL_FILE)

# Calculate evaluation metrics
total_questions = len(df)

avg_correctness = df["Correctness"].mean()
avg_faithfulness = df["Faithfulness"].mean()

correctness_distribution = df["Correctness"].value_counts().sort_index()
faithfulness_distribution = df["Faithfulness"].value_counts().sort_index()

# Previously calculated retrieval results
recall_results = {
    "Recall@1": 26.83,
    "Recall@3": 49.75,
    "Recall@5": 54.75,
    "Recall@10": 61.58
}

# Generate score distribution chart
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

correctness_distribution.plot(
    kind="bar", ax=axes[0], title="Correctness Distribution"
)
faithfulness_distribution.plot(
    kind="bar", ax=axes[1], title="Faithfulness Distribution"
)

for ax in axes:
    ax.set_xlabel("Score")
    ax.set_ylabel("Number of Questions")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.tight_layout()

chart_file = DATA_DIR / "score_distribution.png"
plt.savefig(chart_file, dpi=150)
plt.close()

# Build report
report = f"""
# RAG System Evaluation Report

## 1. Evaluation Overview

- Total evaluation questions: {total_questions}
- Answer correctness evaluation: Qwen LLM-as-a-Judge
- Answer faithfulness evaluation: Qwen LLM-as-a-Judge

## 2. Retrieval Evaluation

| Metric | Score |
|---|---:|
"""

for metric, score in recall_results.items():
    report += f"| {metric} | {score:.2f}% |\n"

report += f"""
## 3. Answer Evaluation

| Metric | Average Score |
|---|---:|
| Correctness | {avg_correctness:.2f}/5 |
| Faithfulness | {avg_faithfulness:.2f}/5 |

## 4. Score Distribution

### Correctness
"""

for score, count in correctness_distribution.items():
    report += f"- Score {score}: {count} questions\n"

report += "\n### Faithfulness\n"

for score, count in faithfulness_distribution.items():
    report += f"- Score {score}: {count} questions\n"

report += """
## 5. Limitations

- Evaluation was conducted using an automated LLM judge.
- The judge scores are estimates and have not been independently human-verified.
- Retrieval quality and answer quality are evaluated separately.
- Results are based on the selected evaluation dataset and questions.

## 6. Conclusion

The RAG system completed retrieval and answer evaluation
for the selected evaluation questions. The results provide
a baseline for assessing retrieval quality, answer correctness,
and faithfulness to retrieved document context.
"""

REPORT_FILE.write_text(report, encoding="utf-8")

print("Evaluation report generated successfully!")
print("Total questions:", total_questions)
print("Average correctness:", round(avg_correctness, 2))
print("Average faithfulness:", round(avg_faithfulness, 2))
print("Report:", REPORT_FILE)
print("Chart:", chart_file)
