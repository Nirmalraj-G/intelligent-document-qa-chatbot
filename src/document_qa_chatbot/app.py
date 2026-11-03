
import hashlib
import streamlit as st

from document_qa_chatbot.ingestion import (
    extract_text_from_pdfs,
    chunk_documents
)

from document_qa_chatbot.retrieval import (
    generate_embeddings,
    create_faiss_index,
    create_bm25_index,
    retrieve_relevant_chunks
)

from document_qa_chatbot.generation import generate_answer


st.set_page_config(
    page_title="Intelligent Document Q&A Chatbot",
    page_icon="📄"
)

st.title("Intelligent Document Q&A Chatbot")


# Initialize Streamlit session state

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_signature" not in st.session_state:
    st.session_state.document_signature = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = None

if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = None


# PDF upload

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


# Process uploaded PDFs

if uploaded_files:

    signature = hashlib.sha256(
        b"".join(
            file.name.encode() + file.getvalue()
            for file in uploaded_files
        )
    ).hexdigest()

    # Rebuild indexes only when uploaded documents change

    if signature != st.session_state.document_signature:

        try:

            with st.spinner("Processing PDF documents..."):

                # Step 1: Extract text from PDFs
                documents = extract_text_from_pdfs(uploaded_files)

                if not documents:
                    raise ValueError(
                        "No extractable text found in the uploaded PDFs."
                    )

                # Step 2: Create chunks
                chunks = chunk_documents(documents)

                if not chunks:
                    raise ValueError(
                        "No text chunks were created."
                    )

                print("\nTotal chunks:", len(chunks))

                # Step 3: Generate embeddings
                embeddings = generate_embeddings(chunks)

                # Step 4: Create FAISS index
                index = create_faiss_index(embeddings)

                print("FAISS index created successfully!")

                # Step 5: Create BM25 index
                bm25_index = create_bm25_index(chunks)

                print("BM25 index created successfully!")

                # Step 6: Store indexes and document data
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.chunk_embeddings = embeddings
                st.session_state.bm25_index = bm25_index
                st.session_state.document_signature = signature

                # Clear previous conversation for new documents
                st.session_state.messages = []

            st.success(
                f"Processed {len(uploaded_files)} PDF(s) "
                f"into {len(chunks)} chunks."
            )

        except Exception as error:

            st.session_state.chunks = None
            st.session_state.index = None
            st.session_state.chunk_embeddings = None
            st.session_state.bm25_index = None
            st.session_state.document_signature = None

            st.error(f"Document processing failed: {error}")

else:

    # Clear document state when no PDFs are uploaded
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.chunk_embeddings = None
    st.session_state.bm25_index = None
    st.session_state.document_signature = None


# Document selection for metadata filtering

document_names = sorted({
    chunk["document_name"]
    for chunk in st.session_state.chunks
}) if st.session_state.chunks else []

selected_document = st.selectbox(
    "Filter by document",
    options=["All Documents"] + document_names
)


# Display previous conversation

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):

            with st.expander("View sources"):

                for source in message["sources"]:

                    st.write(
                        f"**Document:** {source['document_name']} | "
                        f"**Page:** {source['page_number']}"
                    )

                    if source.get("score") is not None:
                        st.write(
                            f"**FAISS similarity score:** "
                            f"{source['score']:.4f}"
                        )

                    if source.get("rrf_score") is not None:
                        st.write(
                            f"**RRF score:** "
                            f"{source['rrf_score']:.6f}"
                        )

                    st.write(source["text"])
                    st.divider()


# Chat input

if (
    st.session_state.index is not None
    and st.session_state.bm25_index is not None
):

    question = st.chat_input(
        "Ask a question about your documents..."
    )

    if question:

        # Display and save the user's question

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Hybrid retrieval and answer generation

        with st.chat_message("assistant"):

            try:

                with st.spinner("Finding relevant information..."):

                    # Step 1: Retrieve using FAISS + BM25 + RRF
                    retrieved_chunks = retrieve_relevant_chunks(
                        question=question,
                        index=st.session_state.index,
                        chunks=st.session_state.chunks,
                        chunk_embeddings=st.session_state.chunk_embeddings,
                        bm25_index=st.session_state.bm25_index,
                        top_k=5,
                        min_score=0.35,
                        document_filter=(
                            None
                            if selected_document == "All Documents"
                            else selected_document
                        ),
                        bm25_top_k=5,
                        rrf_top_k=5,
                        rerank_top_k=3
                    )

                    # Step 2: Generate answer
                    answer = generate_answer(
                        question,
                        retrieved_chunks
                    )

                st.markdown(answer)

                # Prepare source information
                sources = [
                    {
                        "document_name": chunk["document_name"],
                        "page_number": chunk["page_number"],
                        "text": chunk["text"],
                        "score": chunk.get("score"),
                        "rrf_score": chunk.get("rrf_score"),
                        "rerank_score": chunk.get("rerank_score")
                    }
                    for chunk in retrieved_chunks
                ]

                # Display sources
                if sources:

                    with st.expander("View sources"):

                        for source in sources:

                            st.write(
                                f"**Document:** {source['document_name']} | "
                                f"**Page:** {source['page_number']}"
                            )

                            if source["score"] is not None:
                                st.write(
                                    f"**FAISS similarity score:** "
                                    f"{source['score']:.4f}"
                                )

                            if source["rrf_score"] is not None:
                                st.write(
                                    f"**RRF score:** "
                                    f"{source['rrf_score']:.6f}"
                                )

                            if source["rerank_score"] is not None:
                                st.write(
                                    f"**CrossEncoder score:** "
                                    f"{source['rerank_score']:.4f}"
                                )

                            st.write(source["text"])
                            st.divider()

                # Save assistant response and sources
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:

                error_message = (
                    f"Could not generate an answer: {error}"
                )

                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": []
                })

else:

    st.info("Upload one or more PDFs to start chatting.")
