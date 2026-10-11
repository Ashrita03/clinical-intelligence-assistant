
import re

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors


MODEL_NAME = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


# ============================================================
# MODEL LOADING
# ============================================================

def load_qa_model():
    """Load the quantized Qwen model using MLX."""
    model, tokenizer = load(MODEL_NAME)
    return tokenizer, model


# ============================================================
# DIRECT LAB VALUE EXTRACTION
# ============================================================

def extract_explicit_lab_value(question, context):
    """Extract an explicitly documented lab value."""

    if not question or not context:
        return None

    question_lower = question.lower()

    pattern = (
        r"([A-Za-z][A-Za-z0-9 ()/%+\-]*?):\s*"
        r"(-?[\d.]+)\s*([^\n]+)"
    )

    matches = re.findall(
        pattern,
        context,
        flags=re.IGNORECASE,
    )

    for test_name, value, unit in matches:
        test_name = test_name.strip()

        if test_name.lower() not in question_lower:
            continue

        unit = unit.split("Reference Range:")[0].strip()

        return (
            f"According to the uploaded report, "
            f"{test_name} is {value.strip()} {unit}."
        )

    return None


# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt():
    """Provide concise, topic-focused medical instructions."""

    return """
You are an educational assistant that helps users understand
uploaded medical reports.

Answer the CURRENT USER QUESTION directly.

RULES:

1. Use REPORT CONTEXT as the only source of patient-specific
facts. Never invent findings, values, diagnoses, or treatments.

2. Do not change the meaning of the report. Respect negative
findings and do not misrepresent recommendations.

3. Focus on the specific medical topic in the current question.
Do not borrow advice from unrelated medical conditions.

4. You may provide general medical education, but distinguish
it from findings documented in the report.

5. For questions asking what someone can generally do:
- Give exactly 4 practical, condition-specific suggestions.
- Use a numbered list.
- Keep each suggestion to one short sentence.
- Do not repeat any suggestion.
- Do not add a second list or a long conclusion.
- If fewer than 4 reliable suggestions are available, give fewer.

6. Prioritize established, directly relevant measures.
For gum inflammation, focus on oral hygiene and professional
dental care rather than generic lifestyle advice.

7. Do not diagnose a disease from one abnormal laboratory
result. Do not prescribe personalized medications or doses.

8. Do not present general advice as personalized treatment.

9. Use conversation history only to understand references.
The current question takes priority.

10. Finish every sentence. Avoid repeated words, sentences,
or bullet points.

11. If information is missing, acknowledge it rather than
guessing.

12. Recommend urgent medical evaluation for potentially
dangerous symptoms, but avoid unnecessary warnings.

Keep responses clear, concise, and medically cautious.
""".strip()


# ============================================================
# USER PROMPT
# ============================================================

def build_user_prompt(
    question,
    context,
    conversation_history=None,
):
    """Build the grounded question prompt."""

    history_text = ""

    if conversation_history:
        recent_history = conversation_history[-6:]
        formatted_history = []

        for message in recent_history:
            if not isinstance(message, dict):
                continue

            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()

            if role in ("user", "assistant") and content:
                formatted_history.append(
                    f"{role.upper()}: {content}"
                )

        if formatted_history:
            history_text = (
                "\n\nRECENT CONVERSATION:\n"
                + "\n".join(formatted_history)
            )

    return f"""
REPORT CONTEXT:
----------------
{context}
----------------
{history_text}

CURRENT USER QUESTION:
{question}

Answer only the current question.

If the user asks for general management advice, provide
exactly 4 short, distinct, relevant suggestions in a numbered
list. Do not add extra suggestions or repeat previous points.

Patient-specific facts must come from REPORT CONTEXT.
Do not diagnose the patient.
""".strip()


# ============================================================
# RESPONSE CLEANUP
# ============================================================

def clean_qa_response(answer, max_list_items=4):
    """
    Remove duplicate list items, excessive list items,
    and clearly incomplete trailing text.

    This is formatting cleanup, not medical fact checking.
    """

    if not answer:
        return ""

    lines = answer.strip().splitlines()

    cleaned = []
    seen_items = set()
    list_count = 0

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        numbered = re.match(
            r"^\d+[\.\)]\s+(.+)$",
            stripped,
        )

        bullet = re.match(
            r"^[-*]\s+(.+)$",
            stripped,
        )

        if numbered or bullet:
            content = (
                numbered.group(1)
                if numbered
                else bullet.group(1)
            ).strip()

            normalized = re.sub(
                r"\s+",
                " ",
                content.lower(),
            )

            if normalized in seen_items:
                continue

            if list_count >= max_list_items:
                continue

            if not content.endswith((".", "!", "?")):
                continue

            seen_items.add(normalized)
            list_count += 1

            if numbered:
                cleaned.append(
                    f"{list_count}. {content}"
                )
            else:
                cleaned.append(
                    f"- {content}"
                )

        else:
            cleaned.append(stripped)

    while cleaned and cleaned[-1] == "":
        cleaned.pop()

    # Remove an incomplete final prose sentence.
    if cleaned:
        last = cleaned[-1].strip()

        if (
            not last.endswith((".", "!", "?", ":"))
            and not re.match(r"^\d+[\.\)]", last)
            and not last.startswith(("- ", "* "))
        ):
            cleaned.pop()

    return "\n".join(cleaned).strip()


# ============================================================
# GROUNDED ANSWER GENERATION
# ============================================================

def generate_grounded_answer(
    question,
    context,
    tokenizer,
    model,
    conversation_history=None,
):
    """Generate and clean a report-grounded AI answer."""

    if not question or not question.strip():
        return "Please enter a question."

    if not context or not context.strip():
        return (
            "I could not find usable report context "
            "for this question."
        )

    question_lower = question.lower()

    interpretation_terms = (
        "mean",
        "normal",
        "range",
        "high",
        "low",
        "diabetes",
        "diagnosis",
        "diagnose",
        "why",
        "how",
        "should",
        "improve",
        "manage",
    )

    is_narrow_value_question = not any(
        term in question_lower
        for term in interpretation_terms
    )

    if is_narrow_value_question:
        explicit_value_answer = extract_explicit_lab_value(
            question,
            context,
        )

        if explicit_value_answer:
            return explicit_value_answer

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(),
        },
        {
            "role": "user",
            "content": build_user_prompt(
                question,
                context,
                conversation_history,
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    sampler = make_sampler(
        temp=0.3,
        top_p=0.9,
    )

    repetition_processors = make_logits_processors(
        repetition_penalty=1.15,
        repetition_context_size=128,
    )

    answer = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=300,
        sampler=sampler,
        logits_processors=repetition_processors,
        verbose=False,
    )

    answer = clean_qa_response(answer)

    if not answer:
        return (
            "I could not generate a complete, reliable "
            "response. Please try a more specific question."
        )

    return answer


# ============================================================
# RETRIEVED CONTEXT
# ============================================================

def build_context_from_results(retrieval_results):
    """Combine retrieved report chunks."""

    if not retrieval_results:
        return ""

    chunks = []

    for result in retrieval_results:
        chunk = result.get("chunk", "").strip()

        if chunk and chunk not in chunks:
            chunks.append(chunk)

    return "\n\n".join(chunks)

