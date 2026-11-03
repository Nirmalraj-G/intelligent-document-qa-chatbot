
import re
import hashlib
from pypdf import PdfReader


def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def extract_text_from_pdfs(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)

        for page_number, page in enumerate(pdf_reader.pages, start=1):
            text = clean_text(page.extract_text() or "")

            if text:
                documents.append({
                    "document_name": uploaded_file.name,
                    "page_number": page_number,
                    "text": text
                })

    return documents



def recursive_chunk_text(text, chunk_size=500, chunk_overlap=100):

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    if not text:
        return []

    separators = ["\n\n", "\n", " ", ""]

    def split_text(text, separators):

        # Base case: text already fits within chunk size
        if len(text) <= chunk_size:
            return [text]

        if not separators:
            return [text[i:i + chunk_size]
                for i in range(0, len(text), chunk_size - chunk_overlap)
            ]

        # Select the first available separator
        separator = separators[-1]

        for sep in separators:
            if sep == "" or sep in text:
                separator = sep
                break

        # Split while preserving separators
        if separator:
            parts = text.split(separator)

            splits = [part + separator for part in parts[:-1] if part]

            if parts[-1]: splits.append(parts[-1])

        else:
            splits = list(text)

        # Recursively split oversized pieces
        final_splits = []

        for part in splits:
            if len(part) > chunk_size:
                final_splits.extend(split_text(part, separators[separators.index(separator) + 1:]))
            else:
                final_splits.append(part)

        return final_splits

    def merge_splits(splits):

        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:

            split_length = len(split)

            # Check whether the next piece fits
            if current_length + split_length > chunk_size:

                if current_chunk:
                    chunks.append("".join(current_chunk).strip())

                # Remove previous pieces until overlap fits
                while (current_chunk and (current_length > chunk_overlap or current_length + split_length > chunk_size)):
                    removed = current_chunk.pop(0)
                    current_length -= len(removed)

                # Handle an oversized individual split
                if split_length > chunk_size:
                    for i in range(0, split_length, chunk_size - chunk_overlap):
                        chunk = split[i:i + chunk_size].strip()
                        if chunk:
                            chunks.append(chunk)

                    current_chunk = []
                    current_length = 0

                    continue

            current_chunk.append(split)
            current_length += split_length

        if current_chunk:
            chunks.append("".join(current_chunk).strip())

        return [chunk for chunk in chunks if chunk]

    splits = split_text(text, separators)

    chunks = merge_splits(splits)

    return chunks



def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    chunk_records = []

    for document in documents:
        text_chunks = recursive_chunk_text(
            document["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for chunk_index, chunk in enumerate(text_chunks):
            if chunk.strip():

                # Generate a unique ID for each chunk
                unique_text = (
                    f"{document['document_name']}_"
                    f"{document['page_number']}_"
                    f"{chunk_index}_"
                    f"{chunk}"
                )

                chunk_id = hashlib.md5(
                    unique_text.encode("utf-8")
                ).hexdigest()

                chunk_records.append({
                    "chunk_id": chunk_id,
                    "document_name": document["document_name"],
                    "page_number": document["page_number"],
                    "chunk_index": chunk_index,
                    "text": chunk
                })
                

    return chunk_records



