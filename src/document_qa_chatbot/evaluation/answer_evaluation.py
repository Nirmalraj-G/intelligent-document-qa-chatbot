
import json
import re
import torch


def evaluate_answer(
    question,
    expected_answer,
    generated_answer,
    retrieved_context,
    model,
    tokenizer
):
    """
    Evaluate answer correctness and faithfulness using an LLM judge.
    """

    prompt = f"""
You are a strict evaluator for a Retrieval-Augmented Generation system.

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

GENERATED ANSWER:
{generated_answer}

RETRIEVED DOCUMENT CONTEXT:
{retrieved_context[:9000]}

Evaluate these metrics independently.

1. Correctness (1-5):
Does the generated answer correctly answer the question
and agree with the expected answer in meaning?

2. Faithfulness (1-5):
Are the generated answer's claims supported by the
retrieved document context?

Scoring:
1 = Very poor
2 = Poor
3 = Partially correct
4 = Good
5 = Excellent

Return ONLY valid JSON:

{{
    "correctness": 1,
    "faithfulness": 1,
    "reason": "Brief explanation"
}}
"""

    messages = [
        {
            "role": "system",
            "content": "You are a strict and objective RAG evaluator."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs["input_ids"].shape[1]

    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    ).strip()

    match = re.search(r"\{.*\}", response, re.DOTALL)

    if not match:
        raise ValueError(f"Invalid judge response: {response}")

    result = json.loads(match.group())

    correctness = int(result["correctness"])
    faithfulness = int(result["faithfulness"])

    if not (1 <= correctness <= 5):
        raise ValueError("Correctness must be between 1 and 5")

    if not (1 <= faithfulness <= 5):
        raise ValueError("Faithfulness must be between 1 and 5")

    return {
        "Correctness": correctness,
        "Faithfulness": faithfulness,
        "Evaluation Reason": result.get("reason", "")
    }
