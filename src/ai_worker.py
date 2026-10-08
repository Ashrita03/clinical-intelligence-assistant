import json
import subprocess
import sys


def run_worker(module_name, input_data):
    """
    Run a worker in a separate Python process
    and return the worker's JSON response.
    """

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            module_name,
        ],
        input=json.dumps(input_data),
        text=True,
        capture_output=True,
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"{module_name} failed: {process.stderr.strip()}"
        )

    output_lines = [
        line.strip()
        for line in process.stdout.splitlines()
        if line.strip()
    ]

    if not output_lines:
        raise RuntimeError(
            f"{module_name} returned no output."
        )

    # Search backward for the final valid JSON line.
    for line in reversed(output_lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue

    raise RuntimeError(
        f"{module_name} did not return valid JSON."
    )


def answer_question(report_text, question):
    """
    Run the complete AI pipeline using isolated processes.

    Process 1:
        SentenceTransformer + FAISS retrieval

    Process 2:
        FLAN-T5 grounded question answering

    Keeping the models in separate processes prevents
    the native-library conflict observed on macOS.
    """

    # =====================================================
    # 1. RETRIEVE REPORT EVIDENCE
    # =====================================================

    rag_result = run_worker(
        "src.rag_worker",
        {
            "report_text": report_text,
            "question": question,
        },
    )

    if not rag_result.get("success"):
        return {
            "success": False,
            "error": rag_result.get(
                "error",
                "RAG retrieval failed.",
            ),
        }

    evidence = rag_result.get(
        "evidence",
        [],
    )

    if not evidence:
        return {
            "success": False,
            "error": "No relevant evidence was retrieved.",
        }


    # =====================================================
    # 2. BUILD CONTEXT FROM RETRIEVED EVIDENCE
    # =====================================================

    context = "\n\n".join(
        item["chunk"]
        for item in evidence
    )


    # =====================================================
    # 3. GENERATE GROUNDED ANSWER
    # =====================================================

    qa_result = run_worker(
        "src.qa_worker",
        {
            "question": question,
            "context": context,
        },
    )

    if not qa_result.get("success"):
        return {
            "success": False,
            "error": qa_result.get(
                "error",
                "Question answering failed.",
            ),
        }


    # =====================================================
    # 4. RETURN FINAL RESULT
    # =====================================================

    return {
        "success": True,
        "answer": qa_result.get(
            "answer",
            "",
        ),
        "evidence": evidence,
    }


def main():
    try:
        request_data = json.load(
            sys.stdin
        )

        report_text = request_data.get(
            "report_text",
            "",
        )

        question = request_data.get(
            "question",
            "",
        )

        if not report_text.strip():
            raise ValueError(
                "Report text is empty."
            )

        if not question.strip():
            raise ValueError(
                "Question is empty."
            )

        result = answer_question(
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