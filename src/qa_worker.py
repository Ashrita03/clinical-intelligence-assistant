import json
import sys

from src.qa_pipeline import (
    load_qa_model,
    generate_grounded_answer,
)


def answer_question(question, context):
    """
    Generate a grounded answer using FLAN-T5.

    This worker intentionally loads ONLY the QA model.
    The embedding model runs separately in rag_worker.py.
    """

    tokenizer, qa_model = load_qa_model()

    answer = generate_grounded_answer(
        question,
        context,
        tokenizer,
        qa_model,
    )

    return {
        "success": True,
        "answer": answer,
    }


def main():
    try:
        request_data = json.load(sys.stdin)

        question = request_data.get(
            "question",
            "",
        )

        context = request_data.get(
            "context",
            "",
        )

        if not question.strip():
            raise ValueError(
                "Question is empty."
            )

        if not context.strip():
            raise ValueError(
                "Retrieved context is empty."
            )

        result = answer_question(
            question,
            context,
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