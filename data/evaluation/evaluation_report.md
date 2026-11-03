
# RAG System Evaluation Report

## 1. Evaluation Overview

- Total evaluation questions: 20
- Answer correctness evaluation: Qwen LLM-as-a-Judge
- Answer faithfulness evaluation: Qwen LLM-as-a-Judge

## 2. Retrieval Evaluation

| Metric | Score |
|---|---:|
| Recall@1 | 26.83% |
| Recall@3 | 49.75% |
| Recall@5 | 54.75% |
| Recall@10 | 61.58% |

## 3. Answer Evaluation

| Metric | Average Score |
|---|---:|
| Correctness | 3.20/5 |
| Faithfulness | 3.45/5 |

## 4. Score Distribution

### Correctness
- Score 3: 16 questions
- Score 4: 4 questions

### Faithfulness
- Score 3: 11 questions
- Score 4: 9 questions

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
