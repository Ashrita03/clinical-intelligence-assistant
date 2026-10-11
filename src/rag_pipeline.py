import faiss
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def chunk_text(text, chunk_size=900, overlap=150):
    """
    Split a medical report into overlapping chunks.

    Larger chunks preserve more clinical context than the original
    500-character chunks while overlap reduces the chance that an
    important finding is split across chunk boundaries.
    """

    if not text or not text.strip():
        return []

    text = text.strip()

    # Short reports do not need to be fragmented.
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def load_embedding_model():
    """
    Load the sentence embedding model used for semantic retrieval.
    """

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def create_embeddings(chunks, model):
    """
    Convert report chunks into normalized embedding vectors.
    """

    if not chunks:
        return []

    return model.encode(
        chunks,
        normalize_embeddings=True,
    )


def create_faiss_index(embeddings):
    """
    Create a cosine-similarity FAISS index.

    Because embeddings are normalized, inner product behaves like
    cosine similarity.
    """

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Cannot create a FAISS index from empty embeddings.")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def retrieve_relevant_chunks(
    query,
    model,
    index,
    chunks,
    top_k=5,
):
    """
    Retrieve the most relevant report chunks for a user question.

    Short reports return all available chunks. Longer reports return
    the highest-ranking semantic matches.
    """

    if not query or not query.strip():
        return []

    if not chunks:
        return []

    top_k = min(max(1, top_k), len(chunks))

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        results.append(
            {
                "chunk": chunks[idx],
                "score": float(score),
                "chunk_index": int(idx),
            }
        )

    return results