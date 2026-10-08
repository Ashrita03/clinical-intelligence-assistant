import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-small"

STOPWORDS = {
    "what", "was", "were", "is", "are", "the", "a", "an",
    "patient", "patients", "result", "results", "level", "levels",
    "value", "values", "report", "clinical", "show", "tell", "me",
    "of", "for", "in", "from"
}


def load_qa_model():
    """
    Load the local language model and tokenizer used
    for grounded question answering.
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    return tokenizer, model


def question_supported_by_context(question, context):
    """
    Check whether all meaningful terms from the question
    appear in the retrieved clinical context.

    This is a conservative grounding safeguard designed
    to reduce unsupported answers.
    """

    question_words = set(
        re.findall(r"\b[a-zA-Z]+\b", question.lower())
    )

    context_words = set(
        re.findall(r"\b[a-zA-Z]+\b", context.lower())
    )

    meaningful_words = question_words - STOPWORDS

    if not meaningful_words:
        return True

    return meaningful_words.issubset(context_words)


def generate_grounded_answer(question, context, tokenizer, model):
    """
    Generate an answer only when the retrieved context
    contains information relevant to the question.
    """

    if not question_supported_by_context(question, context):
        return "The information is not available in the provided report."

    prompt = f"""
Answer the question using only the clinical report context below.

Rules:
1. Use only information explicitly present in the context.
2. Do not use outside medical knowledge.
3. Do not guess or infer missing information.
4. If the requested information is not explicitly present,
   answer exactly: NOT FOUND

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
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    if answer.upper() == "NOT FOUND":
        return "The information is not available in the provided report."

    return answer


def build_context_from_results(retrieval_results):
    """
    Combine retrieved RAG chunks into context for the language model.
    """

    if not retrieval_results:
        return ""

    context_parts = [
        result["chunk"]
        for result in retrieval_results
    ]

    return "\n\n".join(context_parts)