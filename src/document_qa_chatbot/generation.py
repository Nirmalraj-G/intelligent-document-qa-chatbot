
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)


def rewrite_query(question):

    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the user's question into a concise search query "
                "using important keywords related to the question. "
                "Remove conversational phrases and unnecessary words. "
                "Preserve the original intent and meaning. "
                "Do not answer the question. "
                "Do not introduce new information or assumptions. "
                "Return only the rewritten search query, preferably "
                "as one short sentence."
            )
        },
        {
            "role": "user",
            "content": question
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
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs["input_ids"].shape[1]

    rewritten_query = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )

    return rewritten_query.strip()


def create_prompt(question, retrieved_chunks):
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"Source: {chunk['document_name']}, "
            f"Page: {chunk['page_number']}\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "Answer the user's question using only the provided document context.\n\n"

                "Instructions:\n"
                "1. Provide a clear, detailed, and well-structured answer.\n"
                "2. Include all relevant information available in the context.\n"
                "3. Explain important points instead of giving only short summaries.\n"
                "4. Use bullet points or numbered steps when appropriate.\n"
                "5. Explain technical terms when they appear in the context.\n"
                "6. Do not unnecessarily repeat information.\n"
                "7. Do not introduce external knowledge or unsupported facts.\n"
                "8. If the context contains only partial information, clearly explain "
                "what information is available and what is missing.\n"
                "9. If the answer is not supported by the context, state: "
                "'No relevant information was found in the documents.'\n"
                "10. Preserve important names, numbers, dates, and other factual details.\n\n"
                "Your goal is to provide a complete and informative answer based "
                "on the retrieved document context."
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )
        }
    ]

    return messages


def generate_answer(question, retrieved_chunks):
    if not retrieved_chunks:
        return "No relevant information was found in the documents."

    messages = create_prompt(question, retrieved_chunks)

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
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs["input_ids"].shape[1]

    answer = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )

    return answer.strip()


if __name__ == "__main__":

    question = "Does my company cover hospital expenses?"

    rewritten_query = rewrite_query(question)

    print("Original Question:", question)
    print("Rewritten Query:", rewritten_query)