import hashlib
import streamlit as st

from document_qa_chatbot.ingestion import (
    extract_text_from_pdfs,
    chunk_documents,
)

from document_qa_chatbot.retrieval import (
    generate_embeddings,
    create_faiss_index,
    create_bm25_index,
    retrieve_relevant_chunks,
)

from document_qa_chatbot.generation import generate_answer


# ---------------------------------------------------------
# Page configuration and visual styling
# ---------------------------------------------------------

st.set_page_config(
    page_title="DocuMind | Intelligent Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --bg: #0b1020;
        --panel: #111a2e;
        --panel-soft: #162239;
        --line: rgba(148, 163, 184, 0.16);
        --muted: #94a3b8;
        --text: #e8eef8;
        --accent: #7c9cff;
        --accent-2: #43d6c5;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 50% -20%, rgba(72, 104, 190, 0.20), transparent 48%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, 0.72);
    }

    [data-testid="stSidebar"] {
        background: #0e1729;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
        margin: 0 0 8px 0;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 13px;
        background: linear-gradient(135deg, #668cff, #42cfc2);
        color: #07111f;
        font-size: 21px;
        font-weight: 800;
        box-shadow: 0 8px 24px rgba(77, 127, 255, 0.22);
    }

    .brand-name {
        color: #f4f7ff;
        font-family: 'Manrope', sans-serif;
        font-size: 17px;
        font-weight: 800;
        letter-spacing: -0.4px;
        line-height: 1.2;
    }

    .brand-caption {
        color: var(--muted);
        font-size: 11px;
        margin-top: 3px;
    }

    .sidebar-label {
        color: #dce6f7;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 25px 0 10px 0;
    }

    .main-wrap {
        max-width: 980px;
        margin: 0 auto;
        padding: 28px 24px 110px 24px;
    }

    .topline {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 28px;
    }

    .eyebrow {
        color: #8da8ff;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .main-title {
        color: #f5f7ff;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(27px, 4vw, 39px);
        font-weight: 800;
        letter-spacing: -1.3px;
        line-height: 1.15;
        margin: 0;
    }

    .subtitle {
        color: #9baac1;
        font-size: 14px;
        margin-top: 10px;
        line-height: 1.6;
    }

    .status-pill {
        white-space: nowrap;
        border: 1px solid rgba(67, 214, 197, 0.25);
        background: rgba(67, 214, 197, 0.08);
        color: #75e4d5;
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 11px;
        font-weight: 700;
    }

    .welcome-card {
        border: 1px solid var(--line);
        background: linear-gradient(135deg, rgba(24, 39, 67, 0.96), rgba(17, 26, 46, 0.96));
        border-radius: 20px;
        padding: 25px 27px;
        margin: 18px 0 24px 0;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.12);
    }

    .welcome-title {
        color: #f4f7ff;
        font-family: 'Manrope', sans-serif;
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .welcome-copy {
        color: #aab8ce;
        font-size: 13px;
        line-height: 1.65;
    }

    .feature-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 17px;
    }

    .feature-chip {
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: rgba(9, 16, 31, 0.35);
        color: #c9d5e8;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 11px;
    }

    .section-heading {
        color: #e7edf8;
        font-size: 13px;
        font-weight: 700;
        margin: 24px 0 12px 0;
    }

    .empty-state {
        border: 1px dashed rgba(148, 163, 184, 0.28);
        background: rgba(17, 26, 46, 0.55);
        border-radius: 18px;
        padding: 35px 22px;
        text-align: center;
        margin-top: 20px;
    }

    .empty-icon {
        font-size: 32px;
        margin-bottom: 10px;
    }

    .empty-title {
        color: #e9effb;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .empty-copy {
        color: #96a6bf;
        font-size: 12px;
        line-height: 1.6;
    }

    [data-testid="stChatMessage"] {
        border: 1px solid rgba(148, 163, 184, 0.12);
        background: rgba(17, 26, 46, 0.72);
        border-radius: 16px;
        padding: 15px 17px;
        margin-bottom: 13px;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: rgba(35, 53, 86, 0.52);
    }

    [data-testid="stChatInput"] {
        border: 1px solid rgba(124, 156, 255, 0.35);
        border-radius: 17px;
        background: #111a2e;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.22);
    }

    [data-testid="stChatInput"] textarea {
        color: #eef3ff;
    }

    [data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 12px;
        background: rgba(11, 18, 33, 0.55);
    }

    [data-testid="stFileUploader"] {
        background: rgba(22, 34, 57, 0.55);
        border: 1px dashed rgba(124, 156, 255, 0.35);
        border-radius: 14px;
        padding: 9px;
    }

    div[data-testid="stMetric"] {
        background: rgba(22, 34, 57, 0.65);
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: 12px 14px;
    }

    .doc-item {
        color: #c8d5e8;
        font-size: 12px;
        line-height: 1.45;
        padding: 8px 10px;
        border: 1px solid rgba(148, 163, 184, 0.12);
        background: rgba(22, 34, 57, 0.45);
        border-radius: 9px;
        margin: 6px 0;
        overflow-wrap: anywhere;
    }

    .footer-note {
        color: #71819a;
        font-size: 11px;
        text-align: center;
        margin-top: 35px;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(124, 156, 255, 0.32);
        background: rgba(124, 156, 255, 0.10);
        color: #dce6ff;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #7c9cff;
        color: #ffffff;
        background: rgba(124, 156, 255, 0.18);
    }

    @media (max-width: 700px) {
        .main-wrap { padding: 18px 10px 100px 10px; }
        .welcome-card { padding: 19px; }
        .status-pill { display: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Sidebar: document workspace
# ---------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">✦</div>
            <div>
                <div class="brand-name">DocuMind</div>
                <div class="brand-caption">Intelligent document assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">Your documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Add PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to start asking questions.",
    )

    if uploaded_files:
        st.markdown('<div class="sidebar-label">Uploaded files</div>', unsafe_allow_html=True)
        for uploaded_file in uploaded_files:
            st.markdown(
                f'<div class="doc-item">📄 {uploaded_file.name}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Workspace</div>', unsafe_allow_html=True)

    if st.button("＋  New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Processing status and document filter are populated after indexing.
    if st.session_state.chunks:
        st.markdown('<div class="sidebar-label">Document filter</div>', unsafe_allow_html=True)
        document_names = sorted({
            chunk["document_name"] for chunk in st.session_state.chunks
        })
        selected_document = st.selectbox(
            "Search within",
            options=["All Documents"] + document_names,
            label_visibility="collapsed",
        )
        st.markdown('<div class="sidebar-label">Knowledge base</div>', unsafe_allow_html=True)
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("PDFs", len(uploaded_files) if uploaded_files else len(document_names))
        metric_col2.metric("Chunks", len(st.session_state.chunks))
    else:
        selected_document = "All Documents"
        st.caption("Upload PDFs to build your searchable knowledge base.")

    st.markdown("---")
    st.caption("Powered by hybrid retrieval · FAISS · BM25 · RAG")


# ---------------------------------------------------------
# Process uploaded PDFs (original RAG logic retained)
# ---------------------------------------------------------

if uploaded_files:
    signature = hashlib.sha256(
        b"".join(
            file.name.encode() + file.getvalue()
            for file in uploaded_files
        )
    ).hexdigest()

    if signature != st.session_state.document_signature:
        try:
            with st.spinner("Reading PDFs and building your knowledge base..."):
                # Step 1: Extract text from PDFs
                documents = extract_text_from_pdfs(uploaded_files)

                if not documents:
                    raise ValueError(
                        "No extractable text found in the uploaded PDFs."
                    )

                # Step 2: Create chunks
                chunks = chunk_documents(documents)

                if not chunks:
                    raise ValueError("No text chunks were created.")

                # Step 3: Generate embeddings
                embeddings = generate_embeddings(chunks)

                # Step 4: Create FAISS index
                index = create_faiss_index(embeddings)

                # Step 5: Create BM25 index
                bm25_index = create_bm25_index(chunks)

                # Step 6: Store indexes and document data
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.chunk_embeddings = embeddings
                st.session_state.bm25_index = bm25_index
                st.session_state.document_signature = signature

                # Clear previous conversation when the document set changes
                st.session_state.messages = []

            st.toast(
                f"Knowledge base ready: {len(uploaded_files)} PDF(s), "
                f"{len(chunks)} chunks.",
                icon="✅",
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


# ---------------------------------------------------------
# Main chat workspace
# ---------------------------------------------------------

ready = (
    st.session_state.index is not None
    and st.session_state.bm25_index is not None
)

st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="topline">
        <div>
            <div class="eyebrow">AI-powered knowledge workspace</div>
            <h1 class="main-title">Ask your documents.</h1>
            <div class="subtitle">
                Get clear answers grounded in your PDFs, with source references
                you can verify.
            </div>
        </div>
        <div class="status-pill">{'● KNOWLEDGE BASE READY' if ready else '○ WAITING FOR DOCUMENTS'}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not ready and not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
            <div class="welcome-title">Welcome to your document assistant</div>
            <div class="welcome-copy">
                Upload your PDF files in the left panel. DocuMind will process
                the content and make it searchable, so you can ask questions
                and explore information across your documents.
            </div>
            <div class="feature-row">
                <span class="feature-chip">📚 Multiple PDFs</span>
                <span class="feature-chip">🔎 Hybrid search</span>
                <span class="feature-chip">🧠 Context-grounded answers</span>
                <span class="feature-chip">📌 Page-level sources</span>
            </div>
        </div>
        <div class="empty-state">
            <div class="empty-icon">📄</div>
            <div class="empty-title">Your workspace is ready</div>
            <div class="empty-copy">
                Add one or more PDFs from the sidebar to begin your conversation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif ready and not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
            <div class="welcome-title">Your documents are ready to explore</div>
            <div class="welcome-copy">
                Ask a question below. Answers are generated using retrieved
                document passages, and you can expand the sources to inspect
                the supporting pages and retrieval scores.
            </div>
            <div class="feature-row">
                <span class="feature-chip">✓ PDFs processed</span>
                <span class="feature-chip">✓ FAISS + BM25 search</span>
                <span class="feature-chip">✓ Source transparency</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources):
    """Render retrieved passages and their source metadata."""
    if not sources:
        return

    with st.expander(f"📚 View sources ({len(sources)})", expanded=False):
        for source_number, source in enumerate(sources, start=1):
            st.markdown(f"**Source {source_number} · {source['document_name']}**")
            st.caption(f"Page {source['page_number']}")

            score_cols = st.columns(3)
            if source.get("score") is not None:
                score_cols[0].caption(
                    f"FAISS similarity: {source['score']:.4f}"
                )
            if source.get("rrf_score") is not None:
                score_cols[1].caption(
                    f"RRF score: {source['rrf_score']:.6f}"
                )
            if source.get("rerank_score") is not None:
                score_cols[2].caption(
                    f"CrossEncoder: {source['rerank_score']:.4f}"
                )

            st.markdown(
                f'<div class="doc-item">{source["text"]}</div>',
                unsafe_allow_html=True,
            )
            if source_number < len(sources):
                st.divider()


# Display previous conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))


# Chat input and RAG response generation
if ready:
    question = st.chat_input("Ask anything about your uploaded documents...")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your documents and preparing an answer..."):
                    # Step 1: Retrieve using FAISS + BM25 + RRF + reranking
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
                        rerank_top_k=3,
                    )

                    # Step 2: Generate answer
                    answer = generate_answer(question, retrieved_chunks)

                st.markdown(answer)

                sources = [
                    {
                        "document_name": chunk["document_name"],
                        "page_number": chunk["page_number"],
                        "text": chunk["text"],
                        "score": chunk.get("score"),
                        "rrf_score": chunk.get("rrf_score"),
                        "rerank_score": chunk.get("rerank_score"),
                    }
                    for chunk in retrieved_chunks
                ]

                render_sources(sources)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as error:
                error_message = f"Could not generate an answer: {error}"
                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                })
else:
    st.caption("Upload and process at least one PDF to enable the chat input.")

st.markdown(
    '<div class="footer-note">DocuMind · Intelligent Document Q&A · '
    'Answers are grounded in your uploaded documents.</div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)
