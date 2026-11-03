
from pathlib import Path
import pandas as pd

from document_qa_chatbot.evaluation.answer_evaluation import (
    evaluate_answer
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

ANSWERS_FILE = EVALUATION_DIR / "evaluation_results.csv"
JUDGE_FILE = EVALUATION_DIR / "judge_results.csv"
FINAL_FILE = EVALUATION_DIR / "final_evaluation.csv"


def load_evaluation_data():
    """Load previously generated answers and judge results."""

    answers_df = pd.read_csv(ANSWERS_FILE)

    if JUDGE_FILE.exists():
        judge_df = pd.read_csv(JUDGE_FILE)
    else:
        judge_df = pd.DataFrame(
            columns=[
                "Question ID",
                "Correctness",
                "Faithfulness",
                "Evaluation Reason"
            ]
        )

    return answers_df, judge_df


def run_answer_evaluation(model, tokenizer):
    """Evaluate only questions without completed judge scores."""

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    answers_df, judge_df = load_evaluation_data()

    completed_ids = set(
        judge_df.dropna(subset=["Correctness", "Faithfulness"])[
            "Question ID"
        ].astype(str)
    )

    for _, row in answers_df.iterrows():

        q_id = str(row["Question ID"])

        if q_id in completed_ids:
            continue

        print(f"Evaluating {q_id}...")

        result = evaluate_answer(
            question=row["Question"],
            expected_answer=row["Expected Answer"],
            generated_answer=row["Generated Answer"],
            retrieved_context=row["Retrieved Context"],
            model=model,
            tokenizer=tokenizer
        )

        new_result = {
            "Question ID": q_id,
            **result
        }

        judge_df = pd.concat(
            [judge_df, pd.DataFrame([new_result])],
            ignore_index=True
        )

        # Save immediately to support recovery after interruption
        judge_df.to_csv(JUDGE_FILE, index=False)

        completed_ids.add(q_id)

        print(f"Saved evaluation: {len(completed_ids)}/20")

    return build_final_evaluation()


def build_final_evaluation():
    """Merge generated answers and judge scores."""

    answers_df, judge_df = load_evaluation_data()

    final_df = answers_df.merge(
        judge_df,
        on="Question ID",
        how="left"
    )

    final_df.to_csv(FINAL_FILE, index=False)

    return final_df
