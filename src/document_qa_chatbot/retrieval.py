
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from document_qa_chatbot.generation import rewrite_query
from rank_bm25 import BM25Okapi

model = SentenceTransformer("all-MiniLM-L6-v2")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def create_bm25_index(chunks):
    tokenized_chunks = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    bm25_index = BM25Okapi(tokenized_chunks)

    return bm25_index

def retrieve_bm25_chunks(question, chunks, bm25_index, top_k=5):
    tokenized_question = question.lower().split()

    scores = bm25_index.get_scores(tokenized_question)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:
        if scores[idx] > 0:
            result = chunks[idx].copy()
            result["bm25_score"] = float(scores[idx])
            results.append(result)

    return results

def combine_results_rrf(
    faiss_results,
    bm25_results,
    k=60,
    top_k=5
):
    rrf_scores = {}
    chunk_lookup = {}

    retrieval_results = [
        faiss_results,
        bm25_results
    ]

    for results in retrieval_results:

        for rank, chunk in enumerate(results, start=1):

            chunk_id = (
                chunk["document_name"],
                chunk["page_number"],
                chunk["text"]
            )

            score = 1 / (k + rank)

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0) + score
            )

            chunk_lookup[chunk_id] = chunk

    ranked_ids = sorted(
        rrf_scores,
        key=lambda chunk_id: rrf_scores[chunk_id],
        reverse=True
    )

    combined_results = []

    for chunk_id in ranked_ids[:top_k]:

        result = chunk_lookup[chunk_id].copy()
        result["rrf_score"] = rrf_scores[chunk_id]

        combined_results.append(result)

    return combined_results

def generate_embeddings(chunks):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.asarray(embeddings, dtype="float32")


def generate_question_embedding(question):
    embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.asarray(embedding, dtype="float32")


def create_faiss_index(chunk_embeddings):
    dimension = chunk_embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(chunk_embeddings)

    return index


def rerank_chunks(question, retrieved_chunks, top_k=3):

    # Step 1: Handle empty results
    if not retrieved_chunks:
        return []

    # Step 2: Create question and chunk pairs
    pairs = [
        (question, chunk["text"])
        for chunk in retrieved_chunks
    ]

    # Step 3: Calculate relevance scores
    scores = reranker.predict(pairs)

    # Step 4: Attach reranking scores to chunks
    for chunk, score in zip(retrieved_chunks, scores):
        chunk["rerank_score"] = float(score)

    # Step 5: Sort chunks by reranking score
    reranked_chunks = sorted(
        retrieved_chunks,
        key=lambda chunk: chunk["rerank_score"],
        reverse=True
    )

    # Step 6: Return the top-ranked chunks
    return reranked_chunks[:top_k]


def get_user_question(question):
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    return question



def retrieve_relevant_chunks(
    question,
    index,
    chunks,
    chunk_embeddings,
    bm25_index,
    top_k=5,
    min_score=0.35,
    document_filter=None,
    rerank_top_k=3,
    bm25_top_k=5,
    rrf_top_k=5
):
    # Step 1: Validate question
    question = get_user_question(question)

    # Step 2: Rewrite question
    rewritten_question = rewrite_query(question)

    print("\n--- QUERY REWRITING ---")
    print("Original Question:", question)
    print("Rewritten Question:", rewritten_question)

    # Step 3: Apply metadata filtering
    if document_filter:
        filtered_indices = [
            i for i, chunk in enumerate(chunks)
            if chunk["document_name"] == document_filter
        ]
    else:
        filtered_indices = list(range(len(chunks)))

    filtered_chunks = [
        chunks[i] for i in filtered_indices
    ]

    filtered_embeddings = chunk_embeddings[filtered_indices]

    if not filtered_chunks:
        return []

    # Step 4: Prepare BM25 index
    if document_filter:
        filtered_bm25_index = create_bm25_index(filtered_chunks)
    else:
        filtered_bm25_index = bm25_index

    # Step 5: Generate question embedding
    question_embedding = generate_question_embedding(
        rewritten_question
    )

    # Step 6: FAISS retrieval
    filtered_faiss_index = create_faiss_index(
        filtered_embeddings
    )

    faiss_top_k = min(top_k, filtered_faiss_index.ntotal)

    faiss_scores, faiss_indices = filtered_faiss_index.search(
        question_embedding,
        faiss_top_k
    )

    faiss_results = []

    for score, idx in zip(faiss_scores[0], faiss_indices[0]):

        if idx == -1:
            continue

        if float(score) >= min_score:

            original_idx = filtered_indices[int(idx)]

            result = chunks[original_idx].copy()
            result["score"] = float(score)

            faiss_results.append(result)

    print("\n--- FAISS RETRIEVAL ---")
    print("Retrieved chunks:", len(faiss_results))

    # Step 7: BM25 retrieval
    bm25_results = retrieve_bm25_chunks(
        question=rewritten_question,
        chunks=filtered_chunks,
        bm25_index=filtered_bm25_index,
        top_k=bm25_top_k
    )

    print("\n--- BM25 RETRIEVAL ---")
    print("Retrieved chunks:", len(bm25_results))

    # Step 8: Combine results using RRF
    combined_results = combine_results_rrf(
        faiss_results=faiss_results,
        bm25_results=bm25_results,
        k=60,
        top_k=rrf_top_k
    )

    print("\n--- RRF FUSION ---")

    for rank, chunk in enumerate(combined_results, start=1):
        print(
            f"Rank: {rank} | "
            f"RRF Score: {chunk['rrf_score']:.6f} | "
            f"Page: {chunk['page_number']}"
        )

    if not combined_results:
        return []

    # Step 9: CrossEncoder reranking
    reranked_results = rerank_chunks(
        question=question,
        retrieved_chunks=combined_results,
        top_k=rerank_top_k
    )
    
    # Step 10: Final relevance threshold

    min_rerank_score = 0.0

    if not reranked_results:
        return []

    best_score = reranked_results[0]["rerank_score"]

    if best_score < min_rerank_score:
        print("\nNo sufficiently relevant chunks found.")
        return []

    print("\n--- CROSSENCODER RERANKING ---")

    for rank, chunk in enumerate(reranked_results, start=1):
        print(f"\nRank: {rank}")
        print(f"FAISS Score: {chunk.get('score', 'N/A')}")
        print(f"RRF Score: {chunk['rrf_score']:.6f}")
        print(f"CrossEncoder Score: {chunk['rerank_score']:.4f}")
        print(f"Document: {chunk['document_name']}")
        print(f"Page: {chunk['page_number']}")

    return reranked_results




#TESTING
'''if __name__ == "__main__":

    question = "How to choose my mutual fund?"

    test_chunks = [
        {
            "text": "Start with your investment goal and time horizon.",
            "document_name": "Mutual_Fund.pdf",
            "page_number": 104
        },
        {
            "text": "A mutual fund collects money from many investors.",
            "document_name": "Mutual_Fund.pdf",
            "page_number": 97
        },
        {
            "text": "The annual report explains the company's financial performance.",
            "document_name": "Mutual_Fund.pdf",
            "page_number": 151
        },
        {
            "text": "Consider the asset class that suits your investment horizon.",
            "document_name": "Mutual_Fund.pdf",
            "page_number": 104
        }
    ]

    reranked_chunks = rerank_chunks(
        question,
        test_chunks,
        top_k=4
    )

    print("\n--- CROSSENCODER RERANKING RESULTS ---")

    for i, chunk in enumerate(reranked_chunks, start=1):
        print(f"\nRank: {i}")
        print(f"Rerank Score: {chunk['rerank_score']:.4f}")
        print(f"Document: {chunk['document_name']}")
        print(f"Text: {chunk['text']}")'''

