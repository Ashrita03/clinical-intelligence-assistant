import json
import sys

from src.rag_pipeline import (
    chunk_text,
    load_embedding_model,
    create_embeddings,
    create_faiss_index,
    retrieve_relevant_chunks,
)


def retrieve_evidence(report_text, question):

    chunks = chunk_text(report_text)

    if not chunks:
        return {
            "success": False,
            "error": "No usable text was found in the report.",
        }

    embedding_model = load_embedding_model()

    embeddings = create_embeddings(
        chunks,
        embedding_model,
    )

    index = create_faiss_index(
        embeddings
    )

    retrieval_results = retrieve_relevant_chunks(
        question,
        embedding_model,
        index,
        chunks,
        top_k=min(2, len(chunks)),
    )

    return {
        "success": True,
        "evidence": retrieval_results,
    }


def main():

    try:

        request_data = json.load(sys.stdin)

        report_text = request_data.get(
            "report_text",
            ""
        )

        question = request_data.get(
            "question",
            ""
        )

        if not report_text.strip():
            raise ValueError(
                "Report text is empty."
            )

        if not question.strip():
            raise ValueError(
                "Question is empty."
            )

        result = retrieve_evidence(
            report_text,
            question,
        )

        print(
            json.dumps(result)
        )

    except Exception as error:

        print(
            json.dumps(
                {
                    "success": False,
                    "error": str(error),
                }
            )
        )


if __name__ == "__main__":
    main()