
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from document_qa_chatbot.evaluation_controller import run_answer_evaluation


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main():

    print("Loading Qwen evaluation model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )

    model.eval()

    print("Model loaded successfully.")

    final_df = run_answer_evaluation(
        model=model,
        tokenizer=tokenizer
    )

    print("\nEvaluation completed.")

    print("Total questions:", len(final_df))

    print(
        "Average correctness:",
        round(final_df["Correctness"].mean(), 2)
    )

    print(
        "Average faithfulness:",
        round(final_df["Faithfulness"].mean(), 2)
    )

    del model
    del tokenizer

    print("\nFinal evaluation saved successfully.")


if __name__ == "__main__":
    main()
