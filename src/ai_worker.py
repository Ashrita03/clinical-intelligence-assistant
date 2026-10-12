
import json
import re
import subprocess
import sys


# ============================================================
# SUBPROCESS COMMUNICATION
# ============================================================

def run_worker(module_name, input_data):
    """Run a worker in a separate Python process."""

    process = subprocess.run(
        [sys.executable, "-m", module_name],
        input=json.dumps(input_data),
        text=True,
        capture_output=True,
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"{module_name} failed: {process.stderr.strip()}"
        )

    for line in reversed(process.stdout.splitlines()):
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError:
            continue

    raise RuntimeError(
        f"{module_name} did not return valid JSON."
    )


# ============================================================
# ANSWER CLEANING
# ============================================================

def clean_generated_answer(answer):
    """Remove exact repeated lines and incomplete trailing bullets."""

    if not answer:
        return ""

    cleaned_lines = []
    seen_content = set()

    for line in answer.splitlines():
        stripped = line.strip()

        if not stripped:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        normalized = stripped.lower()

        if normalized.startswith(("- ", "* ")):
            normalized = normalized[2:].strip()

        is_heading = stripped.lower() in {
            "from the report:",
            "general information:",
        }

        if not is_heading:
            if normalized in seen_content:
                continue
            seen_content.add(normalized)

        cleaned_lines.append(stripped)

    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()

    if cleaned_lines:
        last_line = cleaned_lines[-1].strip()

        if (
            last_line.startswith(("- ", "* "))
            and len(last_line) < 35
            and not last_line.endswith((".", "!", "?"))
        ):
            cleaned_lines.pop()

    return "\n".join(cleaned_lines).strip()


# ============================================================
# STRUCTURED LABORATORY EXTRACTION
# ============================================================

def extract_structured_lab_facts(report_text):
    """
    Extract explicitly documented laboratory values and
    reference ranges without language-model interpretation.
    """

    if not report_text:
        return []

    pattern = re.compile(
        r"([A-Za-z][A-Za-z0-9 ()/%+\-]*?):\s*"
        r"(-?[\d.]+)\s*"
        r"([A-Za-z%/\u00b5\u03bc]+(?:/[A-Za-z0-9\u00b5\u03bc]+)?)"
        r".{0,80}?"
        r"Reference Range:\s*"
        r"(-?[\d.]+)\s*-\s*(-?[\d.]+)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    lab_facts = []

    for match in pattern.finditer(report_text):
        test_name, value_text, unit, low_text, high_text = (
            group.strip() for group in match.groups()
        )

        try:
            value = float(value_text)
            low = float(low_text)
            high = float(high_text)
        except ValueError:
            continue

        if value < low:
            status = "Low"
        elif value > high:
            status = "High"
        else:
            status = "Normal"

        lab_facts.append(
            {
                "test_name": test_name,
                "value": value_text,
                "unit": unit,
                "reference_low": low_text,
                "reference_high": high_text,
                "status": status,
            }
        )

    return lab_facts


def question_mentions_lab(question, test_name):
    """Match a question to a laboratory test name."""

    if not question or not test_name:
        return False

    question_lower = question.lower()
    test_lower = test_name.lower()

    if test_lower in question_lower:
        return True

    ignored_words = {
        "fasting",
        "serum",
        "blood",
        "plasma",
        "total",
    }

    test_words = [
        word
        for word in re.findall(r"[a-z0-9]+", test_lower)
        if word not in ignored_words
    ]

    return any(
        len(word) >= 3
        and re.search(
            rf"\b{re.escape(word)}\b",
            question_lower,
        )
        for word in test_words
    )


def get_relevant_lab_facts(report_text, question):
    """Find explicitly documented lab tests relevant to the question."""

    lab_facts = extract_structured_lab_facts(report_text)

    return [
        fact
        for fact in lab_facts
        if question_mentions_lab(question, fact["test_name"])
    ]


def build_deterministic_lab_section(report_text, question):
    """Build laboratory facts directly from report measurements."""

    relevant_facts = get_relevant_lab_facts(
        report_text,
        question,
    )

    if not relevant_facts:
        return None

    lines = ["From the report:"]

    for fact in relevant_facts:
        status = fact["status"]

        if status == "High":
            comparison = "above"
        elif status == "Low":
            comparison = "below"
        else:
            comparison = "within"

        lines.append(
            f"- {fact['test_name']}: "
            f"{fact['value']} {fact['unit']}. "
            f"The listed reference range is "
            f"{fact['reference_low']} - "
            f"{fact['reference_high']} {fact['unit']}, "
            f"so this result is {comparison} that range."
        )

    return "\n".join(lines)


# ============================================================
# SAFE LABORATORY EXPLANATIONS
# ============================================================

def asks_about_diagnosis(question):
    """
    Recognize questions asking whether a laboratory result
    establishes a diagnosis.
    """

    question_lower = question.lower()

    diagnosis_terms = (
        "diagnosis",
        "diagnosed",
        "disease",
        "diabetes",
        "prediabetes",
        "anemia",
        "anaemia",
        "infection",
        "kidney disease",
        "liver disease",
        "thyroid disease",
    )

    question_patterns = (
        r"\bdo(?:es)?\s+(?:this|it|that|my result)\s+mean\b",
        r"\b(?:do|does|did|am|is|have|has)\s+"
        r"(?:i|my|this|it|the result)\b",
        r"\b(?:indicate|confirm|prove|diagnose)\b",
    )

    has_diagnosis_term = any(
        term in question_lower
        for term in diagnosis_terms
    )

    has_question_pattern = any(
        re.search(pattern, question_lower)
        for pattern in question_patterns
    )

    return has_diagnosis_term and has_question_pattern


def asks_for_general_management(question):
    """Recognize requests for general ways to manage a lab finding."""

    question_lower = question.lower()

    phrases = (
        "what can i do",
        "how can i improve",
        "how do i improve",
        "how can i lower",
        "how do i lower",
        "how can i increase",
        "how do i increase",
        "how can i manage",
        "how do i manage",
        "what should i do",
        "how can i reduce",
        "how do i reduce",
    )

    return any(
        phrase in question_lower
        for phrase in phrases
    )


def build_safe_lab_answer(report_text, question):
    """
    Build a conservative answer from documented laboratory facts.

    This function deliberately does not append model-generated
    laboratory measurements, diagnoses, or reference ranges.

    General explanations are limited to information that can
    be stated without interpreting an individual patient's
    full clinical situation.
    """

    relevant_facts = get_relevant_lab_facts(
        report_text,
        question,
    )

    if not relevant_facts:
        return None

    lab_section = build_deterministic_lab_section(
        report_text,
        question,
    )

    general_points = []

    if asks_about_diagnosis(question):
        general_points.append(
            "A single laboratory result does not, by itself, "
            "establish a diagnosis. Interpretation depends on "
            "the specific test, other results, symptoms, and "
            "clinical context."
        )

        if "diabetes" in question.lower():
            general_points.append(
                "Diabetes is assessed using established "
                "diagnostic criteria and appropriate testing. "
                "A healthcare professional can determine "
                "whether further testing is needed."
            )

    if asks_for_general_management(question):
        general_points.append(
            "General steps depend on which test is abnormal "
            "and why. A healthcare professional can help "
            "interpret the result and identify appropriate "
            "follow-up."
        )

        glucose_related = any(
            "glucose" in fact["test_name"].lower()
            for fact in relevant_facts
        )

        if glucose_related:
            general_points.append(
                "For glucose-related concerns, general "
                "healthy habits include balanced meals, "
                "regular physical activity, adequate sleep, "
                "and limiting sugar-sweetened drinks."
            )

    if not general_points:
        general_points.append(
            "A result outside a laboratory's reference range "
            "does not automatically establish a disease. "
            "Its significance depends on the test and "
            "clinical context."
        )

    general_section = (
        "General information:\n"
        + "\n".join(
            f"- {point}"
            for point in general_points
        )
    )

    return f"{lab_section}\n\n{general_section}"


# ============================================================
# DENTAL FACTUAL GROUNDING
# ============================================================

def extract_explicit_dental_location(report_text):
    """
    Recognize an explicitly documented cavity location
    when exactly one matching tooth location is found.

    Return None for ambiguous cases.
    """

    if not report_text:
        return None

    text = re.sub(r"\s+", " ", report_text).lower()

    tooth_pattern = re.compile(
        r"\b(?:upper|lower)[ -]+(?:left|right)[ -]+"
        r"(?:(?:first|second|third)[ -]+)?"
        r"(?:molar|premolar|incisor|canine)\b"
    )

    cavity_pattern = re.compile(
        r"\b(?:cavity|cavities|caries|decay)\b"
    )

    candidates = []

    for match in tooth_pattern.finditer(text):
        nearby = text[
            max(0, match.start() - 100):
            min(len(text), match.end() + 100)
        ]

        if cavity_pattern.search(nearby):
            candidates.append(
                re.sub(r"[ -]+", " ", match.group(0))
            )

    unique_locations = list(dict.fromkeys(candidates))

    if len(unique_locations) == 1:
        return unique_locations[0]

    return None


def build_deterministic_dental_answer(report_text, question):
    """
    Answer a narrow tooth-location question when the
    report contains one explicit cavity location.
    """

    question_lower = question.lower()

    asks_tooth_location = (
        any(
            phrase in question_lower
            for phrase in (
                "which tooth",
                "what tooth",
                "where is the cavity",
                "where is the decay",
                "where are the caries",
                "which molar",
            )
        )
        and any(
            word in question_lower
            for word in (
                "cavity",
                "caries",
                "decay",
                "affected",
                "tooth",
                "molar",
            )
        )
    )

    if not asks_tooth_location:
        return None

    location = extract_explicit_dental_location(
        report_text
    )

    if not location:
        return None

    answer = (
        "From the report:\n"
        f"- The documented cavity is associated with "
        f"the **{location}**."
    )

    asks_about_cleaning = (
        "cleaning" in question_lower
        and any(
            phrase in question_lower
            for phrase in (
                "fix",
                "repair",
                "treat",
                "cure",
                "remove",
            )
        )
    )

    if asks_about_cleaning:
        text = report_text.lower()

        cleaning_recommended = (
            "cleaning" in text
            and "recommend" in text
        )

        evaluation_recommended = (
            "evaluation" in text
            and "recommend" in text
        )

        if cleaning_recommended and evaluation_recommended:
            answer += (
                "\n- The report mentions dental cleaning "
                "and further evaluation. It does not "
                "establish that cleaning repairs the cavity."
            )
        else:
            answer += (
                "\n- The report does not establish that "
                "dental cleaning repairs the cavity."
            )

        answer += (
            "\n\nGeneral information:\n"
            "Routine dental cleaning does not repair "
            "an established cavity. A dentist can assess "
            "the tooth and discuss appropriate options."
        )

    return answer


# ============================================================
# GROUNDED ANSWER COMBINATION
# ============================================================

def combine_grounded_answer(
    report_text,
    question,
    generated_answer,
):
    """
    Prefer deterministic answers when supported.

    Laboratory questions never receive unverified
    model-generated measurements or reference ranges.
    """

    dental_answer = build_deterministic_dental_answer(
        report_text,
        question,
    )

    if dental_answer:
        return dental_answer

    lab_answer = build_safe_lab_answer(
        report_text,
        question,
    )

    if lab_answer:
        return lab_answer

    return generated_answer


# ============================================================
# MAIN AI PIPELINE
# ============================================================

def answer_question(
    report_text,
    question,
    conversation_history=None,
    lab_results=None,
):
    """
    Run FAISS retrieval and Qwen question answering
    in isolated subprocesses.

    Recognized factual lab and dental questions use
    conservative deterministic answers.
    """

    if conversation_history is None:
        conversation_history = []

    if lab_results is None:
        lab_results = []

    abnormal_answer = build_abnormal_lab_answer(
        question,
        lab_results,
    )

    if abnormal_answer is not None:
        return {
            "success": True,
            "answer": abnormal_answer,
            "evidence": [],
        }

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

    evidence = rag_result.get("evidence", [])

    if not evidence:
        return {
            "success": False,
            "error": "No relevant evidence was retrieved.",
        }

    context = "\n\n".join(
        item.get("chunk", "")
        for item in evidence
        if item.get("chunk", "").strip()
    )

    if not context.strip():
        return {
            "success": False,
            "error": "Retrieved report context is empty.",
        }

    # Use verified facts when a deterministic answer applies.
    # This also avoids unnecessary model loading for these cases.

    deterministic_answer = (
        build_deterministic_dental_answer(
            report_text,
            question,
        )
    )

    if not deterministic_answer:
        deterministic_answer = build_safe_lab_answer(
            report_text,
            question,
        )

    if deterministic_answer:
        return {
            "success": True,
            "answer": deterministic_answer,
            "evidence": evidence,
        }

    # Other questions continue through the Qwen model.

    qa_result = run_worker(
        "src.qa_worker",
        {
            "question": question,
            "context": context,
            "conversation_history": conversation_history,
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

    cleaned_answer = clean_generated_answer(
        qa_result.get("answer", "")
    )

    if not cleaned_answer:
        return {
            "success": False,
            "error": "The AI model returned an empty answer.",
        }

    grounded_answer = combine_grounded_answer(
        report_text,
        question,
        cleaned_answer,
    )

    grounded_answer = clean_generated_answer(
        grounded_answer
    )

    if not grounded_answer:
        return {
            "success": False,
            "error": "The grounded answer is empty.",
        }

    return {
        "success": True,
        "answer": grounded_answer,
        "evidence": evidence,
    }


# ============================================================
# JSON ENTRY POINT
# ============================================================

def main():
    try:
        request_data = json.load(sys.stdin)

        report_text = request_data.get(
            "report_text", ""
        )

        question = request_data.get(
            "question", ""
        )

        conversation_history = request_data.get(
            "conversation_history", []
        )

        if not isinstance(report_text, str) or not report_text.strip():
            raise ValueError("Report text is empty.")

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question is empty.")

        if not isinstance(conversation_history, list):
            conversation_history = []

        lab_results = request_data.get("lab_results", [])

        if not isinstance(lab_results, list):
            lab_results = []

        result = answer_question(
            report_text=report_text,
            question=question,
            conversation_history=conversation_history,
            lab_results=lab_results,
        )

        print(json.dumps(result))

    except Exception as error:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": str(error),
                }
            )
        )





def build_abnormal_lab_answer(question, lab_results):
    """Answer out-of-range questions using structured laboratory data."""
    if not isinstance(question, str) or not isinstance(lab_results, list):
        return None

    question_lower = question.lower()

    asks_about_abnormal = any(
        phrase in question_lower
        for phrase in (
            "outside the",
            "out of range",
            "out-of-range",
            "abnormal",
            "which results are high",
            "which results are low",
            "above the reference",
            "below the reference",
        )
    )

    if not asks_about_abnormal:
        return None

    abnormal = [
        lab for lab in lab_results
        if isinstance(lab, dict)
        and str(lab.get("status", "")).lower() in ("high", "low")
    ]

    if not abnormal:
        return "No out-of-range measurements were identified in the extracted laboratory results."

    lines = ["According to the extracted laboratory results:"]

    for lab in abnormal:
        lines.append(
            f"- {lab['test_name']}: {lab['value']} "
            f"{lab.get('unit', '')} — {lab['status']} "
            f"(reference range: {lab.get('reference_low', 'N/A')}–"
            f"{lab.get('reference_high', 'N/A')})"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    main()
