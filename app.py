
import hashlib
import html
import json
import os
import subprocess
import sys
import tempfile

import streamlit as st

from src.document_processor import process_medical_pdf


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Clinical Intelligence Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# SESSION STATE
# =========================================================

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "active_report_id" not in st.session_state:
    st.session_state.active_report_id = None


# =========================================================
# AI SUBPROCESS PIPELINE
# =========================================================

def run_ai_pipeline(
    report_text,
    question,
    conversation_history=None,
    lab_results=None,
):
    """
    Run the RAG + QA pipeline in an isolated process.

    Structured laboratory results are passed to the worker
    so reference-range questions can use extracted values.
    """

    if conversation_history is None:
        conversation_history = []

    if lab_results is None:
        lab_results = []

    request_data = {
        "report_text": report_text,
        "question": question,
        "conversation_history": conversation_history,
        "lab_results": lab_results,
    }

    process = subprocess.run(
        [sys.executable, "-m", "src.ai_worker"],
        input=json.dumps(request_data),
        text=True,
        capture_output=True,
    )

    if process.returncode != 0:
        raise RuntimeError(
            process.stderr.strip()
            or "AI worker failed."
        )

    output_lines = [
        line.strip()
        for line in process.stdout.splitlines()
        if line.strip()
    ]

    if not output_lines:
        raise RuntimeError(
            "AI worker returned no output."
        )

    for line in reversed(output_lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue

    raise RuntimeError(
        "AI worker did not return valid JSON."
    )


# =========================================================
# REPORT DISPLAY HELPERS
# =========================================================

REPORT_ICONS = {
    "Laboratory Report": "🧪",
    "Cardiology Report": "❤️",
    "Dental Report": "🦷",
    "Dermatology / Hair Report": "🧴",
    "Gastroenterology Report": "🩻",
    "Radiology / Imaging Report": "🔬",
    "Discharge Summary": "🏥",
    "Clinical Consultation": "🩺",
    "General Medical Report": "📄",
    "Unknown / Empty Report": "❓",
}


def get_report_icon(report_type):
    return REPORT_ICONS.get(
        report_type,
        "📄",
    )


def get_chat_placeholder(report_type):
    """
    Return a report-aware example while still allowing
    completely free-form questions.
    """

    placeholders = {
        "Laboratory Report": (
            "Ask about a result, what it means, or general ways "
            "to improve it..."
        ),
        "Dental Report": (
            "Ask about dental findings, recommendations, or "
            "general oral-health information..."
        ),
        "Cardiology Report": (
            "Ask about heart-related findings or terminology..."
        ),
        "Dermatology / Hair Report": (
            "Ask about skin, scalp, or hair findings..."
        ),
        "Gastroenterology Report": (
            "Ask about digestive findings or terminology..."
        ),
        "Radiology / Imaging Report": (
            "Ask what an imaging finding means..."
        ),
        "Discharge Summary": (
            "Ask about diagnoses, medications, procedures, "
            "or discharge instructions..."
        ),
        "Clinical Consultation": (
            "Ask about findings, assessment, or recommendations..."
        ),
    }

    return placeholders.get(
        report_type,
        "Ask anything about this medical report...",
    )


def build_conversation_history():
    """
    Convert Streamlit chat history into the compact format
    expected by the QA worker.

    Evidence metadata is intentionally excluded.
    """

    history = []

    for message in st.session_state.chat_messages[-6:]:
        role = message.get("role")
        content = message.get("content", "")

        if role in {"user", "assistant"} and content:
            history.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    return history


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 92% 8%,
            rgba(71, 190, 255, 0.20),
            transparent 28%
        ),
        radial-gradient(
            circle at 4% 92%,
            rgba(118, 100, 255, 0.13),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #f9fcff 0%,
            #eef7ff 50%,
            #f5fcff 100%
        );
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.hero {
    padding: 30px 5px 18px 5px;
}

.hero-title {
    font-size: 3.6rem;
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -2px;
    color: #10245c;
    margin-bottom: 20px;
}

.gradient-word {
    background: linear-gradient(
        90deg,
        #12b8ae,
        #2584ff
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    color: #536780;
    font-size: 1.15rem;
    line-height: 1.7;
    max-width: 900px;
}

.feature-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 28px;
    margin-bottom: 25px;
}

.feature-badge {
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid rgba(71, 128, 210, 0.14);
    padding: 13px 18px;
    border-radius: 16px;
    font-weight: 650;
    color: #18366f;
    box-shadow: 0 6px 20px rgba(38, 81, 150, 0.07);
}

.safety-box {
    background: linear-gradient(
        90deg,
        #e8f3ff,
        #edf9ff
    );
    border: 1px solid #cfe5ff;
    padding: 18px 22px;
    border-radius: 16px;
    color: #1462be;
    font-weight: 600;
    margin: 10px 0 35px 0;
}

.upload-heading {
    text-align: center;
    color: #12265c;
    font-size: 2rem;
    font-weight: 800;
    margin-top: 5px;
    margin-bottom: 7px;
}

.upload-subtitle {
    text-align: center;
    color: #687b94;
    font-size: 1rem;
    margin-bottom: 22px;
}

[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid #d8e8fb;
    border-radius: 22px;
    padding: 25px;
    box-shadow: 0 12px 40px rgba(44, 91, 160, 0.10);
}

[data-testid="stFileUploaderDropzone"] {
    background: #f9fcff;
    border: 2px dashed #9fc9ff;
    border-radius: 18px;
    padding-top: 35px;
    padding-bottom: 35px;
}

[data-testid="stAlert"] {
    border-radius: 14px;
}

.section-heading {
    color: #12265c;
    font-size: 1.65rem;
    font-weight: 800;
    margin-top: 30px;
    margin-bottom: 8px;
}

.section-subtitle {
    color: #687b94;
    margin-bottom: 15px;
}

.report-type-card {
    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.96),
            rgba(237,249,255,0.96)
        );
    border: 1px solid #d4e9ff;
    border-radius: 20px;
    padding: 22px 24px;
    margin-top: 28px;
    margin-bottom: 18px;
    box-shadow:
        0 10px 30px
        rgba(40, 84, 150, 0.08);
}

.report-type-label {
    color: #73859d;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 7px;
}

.report-type-value {
    color: #12265c;
    font-size: 1.55rem;
    font-weight: 800;
}

.ai-section {
    margin-top: 38px;
    padding: 28px;
    border-radius: 22px;
    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.94),
            rgba(240,249,255,0.94)
        );
    border: 1px solid #d5e9ff;
    box-shadow:
        0 12px 35px
        rgba(35, 80, 150, 0.09);
}

.ai-title {
    color: #12265c;
    font-size: 1.75rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.ai-description {
    color: #687b94;
    line-height: 1.6;
}

.chat-note {
    color: #71839b;
    font-size: 0.9rem;
    margin-top: 10px;
    margin-bottom: 18px;
}

.info-card {
    min-height: 175px;
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid rgba(100, 150, 220, 0.14);
    border-radius: 20px;
    padding: 23px;
    box-shadow: 0 8px 30px rgba(37, 79, 145, 0.08);
    margin-top: 28px;
}

.card-icon {
    font-size: 2rem;
    margin-bottom: 11px;
}

.card-title {
    font-size: 1.05rem;
    font-weight: 750;
    color: #142c66;
    margin-bottom: 8px;
}

.card-text {
    color: #667891;
    line-height: 1.55;
    font-size: 0.93rem;
}

.prototype-note {
    text-align: center;
    color: #8291a7;
    font-size: 0.82rem;
    margin-top: 35px;
    padding-bottom: 10px;
}

@media (max-width: 800px) {
    .hero-title {
        font-size: 2.5rem;
    }

    .hero-subtitle {
        font-size: 1rem;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
<div class="hero">

<div class="hero-title">
🩺 Clinical Intelligence
<span class="gradient-word">Assistant</span>
</div>

<div class="hero-subtitle">
Upload a text-based medical report and transform it into structured,
understandable information. Explore detected findings and have a
multi-turn conversation about the report using grounded document
evidence and clearly separated general medical information.
</div>

<div class="feature-row">

<div class="feature-badge">
📄 Medical PDF Processing
</div>

<div class="feature-badge">
🧠 Report Type Detection
</div>

<div class="feature-badge">
📊 Adaptive Dashboard
</div>

<div class="feature-badge">
💬 Conversational AI
</div>

</div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SAFETY
# =========================================================

st.markdown(
    """
<div class="safety-box">
🛡️ <strong>Educational and research prototype only.</strong>
The assistant can help explain information contained in medical
reports and provide clearly identified general educational information.
It does not diagnose medical conditions, prescribe personalized
treatment, or replace care from a qualified healthcare professional.
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# UPLOAD
# =========================================================

st.markdown(
    """
<div class="upload-heading">
📄 Upload Medical Report
</div>

<div class="upload-subtitle">
Upload a text-based PDF such as a laboratory, dental,
cardiology, dermatology, gastroenterology, imaging,
consultation, or discharge report.
</div>
""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload medical report (PDF, JPG, or PNG)",
    type=["pdf", "jpg", "jpeg", "png"],
    label_visibility="collapsed",
)


# =========================================================
# PROCESS REPORT
# =========================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    report_id = hashlib.sha256(
        file_bytes
    ).hexdigest()

    # A different report starts a new conversation.
    if st.session_state.active_report_id != report_id:
        st.session_state.active_report_id = report_id
        st.session_state.chat_messages = []

    st.success(
        f"✓ {uploaded_file.name} uploaded successfully"
    )

    temp_file_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:

            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        # Convert image uploads into PDFs for OCR processing.
        if uploaded_file.name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            import io
            from PIL import Image

            image = Image.open(
                io.BytesIO(file_bytes)
            ).convert("RGB")

            image.save(temp_file_path, format="PDF")

        report = process_medical_pdf(
            temp_file_path
        )

        extracted_text = report["text"]
        report_type = report["report_type"]
        metadata = report["metadata"]
        lab_dataframe = report["lab_dataframe"]

        # Convert the dashboard's structured laboratory data
        # into JSON-compatible records for the AI worker.
        lab_results = json.loads(
            lab_dataframe.to_json(orient="records")
        )

        report_icon = get_report_icon(
            report_type
        )

        # =================================================
        # REPORT TYPE
        # =================================================

        safe_report_type = html.escape(
            report_type
        )

        st.markdown(
            f"""
<div class="report-type-card">

<div class="report-type-label">
Detected Document Type
</div>

<div class="report-type-value">
{report_icon} {safe_report_type}
</div>

</div>
""",
            unsafe_allow_html=True,
        )

        # =================================================
        # GENERAL REPORT OVERVIEW
        # =================================================

        st.markdown(
            """
<div class="section-heading">
📋 Report Overview
</div>

<div class="section-subtitle">
Document-level information extracted from the uploaded report.
</div>
""",
            unsafe_allow_html=True,
        )

        overview1, overview2, overview3 = st.columns(3)

        with overview1:
            st.metric(
                "Report Type",
                report_type,
            )

        with overview2:
            st.metric(
                "Word Count",
                metadata["word_count"],
            )

        with overview3:
            st.metric(
                "Lab Values Detected",
                metadata["lab_results_detected"],
            )

        # =================================================
        # ADAPTIVE LAB DASHBOARD
        # =================================================

        if not lab_dataframe.empty:

            st.markdown(
                """
<div class="section-heading">
🧪 Laboratory Findings
</div>

<div class="section-subtitle">
Structured laboratory measurements detected in this report.
Status is determined only by comparison with the reference
range printed in the document.
</div>
""",
                unsafe_allow_html=True,
            )

            metric1, metric2, metric3, metric4 = st.columns(4)

            with metric1:
                st.metric(
                    "Tests Detected",
                    metadata["lab_results_detected"],
                )

            with metric2:
                st.metric(
                    "Normal",
                    metadata["normal_results"],
                )

            with metric3:
                st.metric(
                    "High",
                    metadata["high_results"],
                )

            with metric4:
                st.metric(
                    "Low",
                    metadata["low_results"],
                )

            st.dataframe(
                lab_dataframe,
                width="stretch",
                hide_index=True,
            )

            abnormal_dataframe = lab_dataframe[
                lab_dataframe["status"].isin(["High", "Low"])
            ]

            if not abnormal_dataframe.empty:

                st.markdown(
                    """
<div class="section-heading">
⚠️ Values Outside the Reported Reference Range
</div>
""",
                    unsafe_allow_html=True,
                )

                st.dataframe(
                    abnormal_dataframe,
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    "A value outside a printed reference range "
                    "does not by itself establish a diagnosis."
                )

        else:

            st.info(
                f"No structured laboratory measurements were detected "
                f"in this {report_type.lower()}. The complete extracted "
                "report text is still available to the conversational "
                "question-answering pipeline."
            )

        # =================================================
        # EXTRACTION QUALITY REVIEW
        # =================================================

        extraction_quality = report.get("extraction_quality", {})

        if not extraction_quality.get("assessment_available", False):
            st.info(
                "Extraction completeness could not be assessed. "
                "Review the original report for potentially missing measurements."
            )

        elif extraction_quality.get("requires_review", False):
            missed_count = extraction_quality.get(
                "potentially_missed_count", 0
            )

            st.warning(
                f"Potentially incomplete extraction: {missed_count} "
                "measurement-like row(s) could not be reliably extracted. "
                "Review these entries against the original report."
            )

            with st.expander("Review potentially missed measurements"):
                for item in extraction_quality.get(
                    "potentially_missed_rows", []
                ):
                    st.markdown(
                        f"**{item.get('test_name', 'Unknown measurement')}**"
                    )
                    st.code(item.get("source_text", ""))
                    st.caption(
                        f"Page: {item.get('page', 'Unknown')}"
                    )

        else:
            st.caption(
                "No additional measurement-like rows were flagged "
                "by the extraction-quality checker. This does not "
                "guarantee that every measurement was extracted."
            )

        # =================================================
        # EXTRACTED TEXT
        # =================================================

        with st.expander(
            "🔍 View Extracted Report Text"
        ):

            if extracted_text.strip():

                st.text(
                    extracted_text
                )

            else:

                st.warning(
                    "No readable text was extracted from this PDF. "
                    "The file may contain scanned images rather than "
                    "machine-readable text."
                )

        # =================================================
        # CONVERSATIONAL AI
        # =================================================

        st.markdown(
            """
<div class="ai-section">

<div class="ai-title">
💬 Chat About This Report
</div>

<div class="ai-description">
Ask free-form questions about the uploaded report. Follow-up questions
can use the recent conversation for context. Patient-specific facts
remain grounded in the report, while broader explanations are presented
as general medical information.
</div>

<div class="chat-note">
Your conversation resets automatically when you upload a different report.
</div>

</div>
""",
            unsafe_allow_html=True,
        )

        # =================================================
        # DISPLAY EXISTING CHAT HISTORY
        # =================================================

        for message in st.session_state.chat_messages:

            role = message.get(
                "role",
                "assistant",
            )

            content = message.get(
                "content",
                "",
            )

            with st.chat_message(role):

                st.markdown(
                    content
                )

                evidence_items = message.get(
                    "evidence",
                    [],
                )

                if (
                    role == "assistant"
                    and evidence_items
                ):

                    with st.expander(
                        "🔎 View Retrieved Evidence"
                    ):

                        for number, evidence in enumerate(
                            evidence_items,
                            start=1,
                        ):

                            score = evidence.get(
                                "score",
                                0.0,
                            )

                            chunk = evidence.get(
                                "chunk",
                                "",
                            )

                            st.markdown(
                                f"**Evidence {number} "
                                f"— Similarity: {score:.3f}**"
                            )

                            st.write(
                                chunk
                            )

                            st.divider()

        # =================================================
        # CHAT INPUT
        # =================================================

        if extracted_text.strip():

            user_question = st.chat_input(
                get_chat_placeholder(
                    report_type
                )
            )

            if user_question:

                # Build history BEFORE adding the new user
                # message so the current question is not duplicated.
                conversation_history = (
                    build_conversation_history()
                )

                st.session_state.chat_messages.append(
                    {
                        "role": "user",
                        "content": user_question,
                    }
                )

                with st.chat_message("user"):
                    st.markdown(
                        user_question
                    )

                with st.chat_message("assistant"):

                    with st.spinner(
                        "Retrieving report evidence and generating answer..."
                    ):

                        try:

                            result = run_ai_pipeline(
                                report_text=extracted_text,
                                question=user_question,
                                conversation_history=conversation_history,
                                lab_results=lab_results,
                            )

                            if not result.get("success"):

                                error_message = result.get(
                                    "error",
                                    "The AI pipeline could not complete "
                                    "the request.",
                                )

                                st.error(
                                    error_message
                                )

                            else:

                                answer = result.get(
                                    "answer",
                                    "",
                                )

                                retrieval_results = result.get(
                                    "evidence",
                                    [],
                                )

                                st.markdown(
                                    answer
                                )

                                if retrieval_results:

                                    with st.expander(
                                        "🔎 View Retrieved Evidence"
                                    ):

                                        for number, evidence in enumerate(
                                            retrieval_results,
                                            start=1,
                                        ):

                                            score = evidence.get(
                                                "score",
                                                0.0,
                                            )

                                            chunk = evidence.get(
                                                "chunk",
                                                "",
                                            )

                                            st.markdown(
                                                f"**Evidence {number} "
                                                f"— Similarity: "
                                                f"{score:.3f}**"
                                            )

                                            st.write(
                                                chunk
                                            )

                                            st.divider()

                                st.session_state.chat_messages.append(
                                    {
                                        "role": "assistant",
                                        "content": answer,
                                        "evidence": retrieval_results,
                                    }
                                )

                        except Exception as ai_error:

                            st.error(
                                "The AI question-answering pipeline "
                                "could not complete the request: "
                                f"{ai_error}"
                            )

        else:

            st.warning(
                "Chat is unavailable because no machine-readable "
                "text was extracted from this PDF."
            )

        # =================================================
        # CHAT CONTROLS
        # =================================================

        if st.session_state.chat_messages:

            if st.button(
                "🗑️ Clear Conversation",
                key="clear_conversation",
            ):

                st.session_state.chat_messages = []
                st.rerun()

    except Exception as error:

        st.error(
            f"Unable to process the uploaded PDF: {error}"
        )

    finally:

        if (
            temp_file_path is not None
            and os.path.exists(temp_file_path)
        ):

            os.remove(
                temp_file_path
            )


# =========================================================
# FEATURE CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        """
<div class="info-card">

<div class="card-icon">📄</div>

<div class="card-title">
Medical Report Processing
</div>

<div class="card-text">
Extract machine-readable text from uploaded medical PDF reports
for downstream analysis.
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        """
<div class="info-card">

<div class="card-icon">🧠</div>

<div class="card-title">
Adaptive Document Intelligence
</div>

<div class="card-text">
Identify the report category and adapt the interface to the
information detected in the document.
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with col3:

    st.markdown(
        """
<div class="info-card">

<div class="card-icon">💬</div>

<div class="card-title">
Conversational AI
</div>

<div class="card-text">
Ask natural-language questions and follow-up questions while retaining
recent conversational context.
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with col4:

    st.markdown(
        """
<div class="info-card">

<div class="card-icon">🔎</div>

<div class="card-title">
Grounded Evidence
</div>

<div class="card-text">
Retrieve relevant report passages so answers can be connected
back to the uploaded document.
</div>

</div>
""",
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="prototype-note">
Clinical Intelligence Assistant • AI & Data Science Research Prototype
</div>
""",
    unsafe_allow_html=True,
)
