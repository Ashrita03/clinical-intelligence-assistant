import faiss
from sentence_transformers import SentenceTransformer


def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks for retrieval.
    """

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def load_embedding_model():
    """
    Load the SentenceTransformer model used to create text embeddings.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def create_embeddings(chunks, model):
    """
    Convert text chunks into numerical embedding vectors.
    """

    if not chunks:
        return []

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    return embeddings


def create_faiss_index(embeddings):
    """
    Create a FAISS vector index from document embeddings.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def retrieve_relevant_chunks(query, model, index, chunks, top_k=2):
    """
    Retrieve the most relevant document chunks for a user query.
    """

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "chunk": chunks[idx],
            "score": float(score)
        })

    return results