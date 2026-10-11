import json
import sys

import platform

if platform.system() == "Darwin":
    from src.qa_pipeline import (
        load_qa_model,
        generate_grounded_answer,
    )
else:
    from src.qa_pipeline_linux import (
        load_qa_model,
        generate_grounded_answer,
    )


def answer_question(
    question,
    context,
    conversation_history=None,
):
    """
    Generate a grounded conversational answer using the
    Qwen2.5 1.5B model.

    This worker intentionally loads only the QA model.
    The embedding model runs separately in rag_worker.py
    to keep the two model workloads isolated.
    """

    tokenizer, qa_model = load_qa_model()

    answer = generate_grounded_answer(
        question=question,
        context=context,
        tokenizer=tokenizer,
        model=qa_model,
        conversation_history=conversation_history,
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

        conversation_history = request_data.get(
            "conversation_history",
            [],
        )

        if not question.strip():
            raise ValueError(
                "Question is empty."
            )

        if not context.strip():
            raise ValueError(
                "Retrieved context is empty."
            )

        if not isinstance(conversation_history, list):
            conversation_history = []

        result = answer_question(
            question=question,
            context=context,
            conversation_history=conversation_history,
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