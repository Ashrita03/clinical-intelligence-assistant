import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-small"


STOPWORDS = {
    "what", "was", "were", "is", "are", "the", "a", "an",
    "patient", "patients", "result", "results", "level", "levels",
    "value", "values", "report", "clinical", "show", "tell", "me",
    "of", "for", "in", "from",
    "s"
}


FALLBACK_ANSWER = (
    "The information is not available "
    "in the provided report."
)


def load_qa_model():
    """
    Load the local language model and tokenizer used
    for grounded question answering.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    return tokenizer, model


def normalize_question(question):
    """
    Normalize possessive endings before question analysis.
    """

    return re.sub(
        r"['’]s\b",
        "",
        question.lower()
    )


def question_supported_by_context(question, context):
    """
    Check whether meaningful terms from the question
    appear in the retrieved clinical context.

    This conservative safeguard reduces unsupported
    answers and hallucinations.
    """

    normalized_question = normalize_question(
        question
    )

    question_words = set(
        re.findall(
            r"\b[a-zA-Z]+\b",
            normalized_question
        )
    )

    context_words = set(
        re.findall(
            r"\b[a-zA-Z]+\b",
            context.lower()
        )
    )

    meaningful_words = (
        question_words - STOPWORDS
    )

    if not meaningful_words:
        return True

    return meaningful_words.issubset(
        context_words
    )


def extract_explicit_lab_value(question, context):
    """
    Extract an explicitly stated laboratory value from
    retrieved report context when the question directly
    asks about that laboratory test.

    This deterministic step prevents the language model
    from replacing an explicit report value with unrelated
    text from the document.
    """

    normalized_question = normalize_question(
        question
    )

    lab_pattern = re.compile(
        r"([A-Za-z][A-Za-z ]*?):\s*"
        r"([\d.]+)\s*"
        r"([A-Za-z0-9/%^.\-]+)"
    )

    matches = lab_pattern.findall(
        context
    )

    for test_name, value, unit in matches:
        cleaned_test_name = (
            test_name.strip().lower()
        )

        test_words = set(
            re.findall(
                r"\b[a-zA-Z]+\b",
                cleaned_test_name
            )
        )

        if (
            test_words
            and test_words.issubset(
                set(
                    re.findall(
                        r"\b[a-zA-Z]+\b",
                        normalized_question
                    )
                )
            )
        ):
            return f"{value} {unit}"

    return None


def generate_grounded_answer(
    question,
    context,
    tokenizer,
    model
):
    """
    Generate a grounded answer using retrieved clinical
    report evidence.

    Explicit laboratory values are extracted
    deterministically when possible. The language model
    is used as a fallback for other supported questions.
    """

    if not question_supported_by_context(
        question,
        context
    ):
        return FALLBACK_ANSWER

    explicit_lab_value = (
        extract_explicit_lab_value(
            question,
            context
        )
    )

    if explicit_lab_value is not None:
        return explicit_lab_value

    prompt = f"""
Use only the clinical report context to answer the question.

Return only the information that directly answers the question.

Do not use outside medical knowledge.
Do not guess.
Do not return headings or document markers such as END OF REPORT.

If the answer is not explicitly present, return exactly:
NOT FOUND

Clinical Report Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    invalid_answers = {
        "",
        "NOT FOUND",
        "END OF REPORT",
        "END REPORT",
    }

    if answer.upper() in invalid_answers:
        return FALLBACK_ANSWER

    return answer


def build_context_from_results(
    retrieval_results
):
    """
    Combine retrieved document chunks into one context
    string for grounded question answering.
    """

    if not retrieval_results:
        return ""

    context_parts = [
        result["chunk"]
        for result in retrieval_results
    ]

    return "\n\n".join(
        context_parts
    )