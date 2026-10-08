import os
import tempfile

import streamlit as st

from src.document_processor import process_clinical_pdf


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
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

/* MAIN PAGE */

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


/* HIDE DEFAULT STREAMLIT ELEMENTS */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* HERO */

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
    max-width: 850px;
}


/* FEATURE BADGES */

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


/* SAFETY NOTICE */

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


/* UPLOAD */

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


/* ALERTS */

[data-testid="stAlert"] {
    border-radius: 14px;
}


/* RESULTS */

.results-heading {
    color: #12265c;
    font-size: 1.65rem;
    font-weight: 800;
    margin-top: 30px;
    margin-bottom: 8px;
}

.results-subtitle {
    color: #687b94;
    margin-bottom: 15px;
}


/* FEATURE CARDS */

.info-card {
    min-height: 175px;
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid rgba(100, 150, 220, 0.14);
    border-radius: 20px;
    padding: 23px;

    box-shadow:
        0 8px 30px
        rgba(37, 79, 145, 0.08);

    margin-top: 28px;

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}

.info-card:hover {
    transform: translateY(-4px);

    box-shadow:
        0 14px 35px
        rgba(37, 79, 145, 0.13);
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


/* FOOTER */

.prototype-note {
    text-align: center;
    color: #8291a7;
    font-size: 0.82rem;
    margin-top: 35px;
    padding-bottom: 10px;
}


/* MOBILE */

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
# HERO SECTION
# =========================================================

st.markdown(
    """
<div class="hero">

<div class="hero-title">
🩺 Clinical Intelligence
<span class="gradient-word">Assistant</span>
</div>

<div class="hero-subtitle">
Transform clinical laboratory reports into structured, understandable information.
Extract laboratory values, analyze results, and ask questions using an AI-powered
retrieval system grounded in your uploaded document.
</div>

<div class="feature-row">

<div class="feature-badge">
📋 Extract Information
</div>

<div class="feature-badge">
📊 Analyze Lab Results
</div>

<div class="feature-badge">
💬 Ask Questions with AI
</div>

<div class="feature-badge">
🔎 Evidence-Based Retrieval
</div>

</div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SAFETY NOTICE
# =========================================================

st.markdown(
    """
<div class="safety-box">
🛡️ <strong>Educational and research prototype only.</strong>
This application does not provide medical diagnoses,
treatment recommendations, or professional medical advice.
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    """
<div class="upload-heading">
📄 Upload Clinical Report
</div>

<div class="upload-subtitle">
Choose a PDF clinical laboratory report to get started.
</div>
""",
    unsafe_allow_html=True,
)


uploaded_file = st.file_uploader(
    "Upload clinical PDF",
    type=["pdf"],
    label_visibility="collapsed",
)


# =========================================================
# PROCESS UPLOADED PDF
# =========================================================

if uploaded_file is not None:

    st.success(
        f"✓ {uploaded_file.name} uploaded successfully"
    )

    temp_file_path = None

    try:

        # -------------------------------------------------
        # SAVE UPLOADED PDF TEMPORARILY
        # -------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            temp_file_path = temp_file.name


        # -------------------------------------------------
        # PROCESS CLINICAL REPORT
        # -------------------------------------------------

        extracted_text, lab_dataframe = process_clinical_pdf(
            temp_file_path
        )


        # -------------------------------------------------
        # DISPLAY LAB RESULTS
        # -------------------------------------------------

        st.markdown(
            """
<div class="results-heading">
📊 Extracted Laboratory Results
</div>

<div class="results-subtitle">
Structured laboratory values detected from the uploaded report.
</div>
""",
            unsafe_allow_html=True,
        )


        if not lab_dataframe.empty:

            st.dataframe(
                lab_dataframe,
                use_container_width=True,
                hide_index=True
            )


            # ---------------------------------------------
            # QUICK SUMMARY
            # ---------------------------------------------

            total_results = len(
                lab_dataframe
            )

            normal_results = (
                lab_dataframe["status"]
                == "Normal"
            ).sum()

            high_results = (
                lab_dataframe["status"]
                == "High"
            ).sum()

            low_results = (
                lab_dataframe["status"]
                == "Low"
            ).sum()


            metric1, metric2, metric3, metric4 = st.columns(4)


            with metric1:

                st.metric(
                    "Tests Detected",
                    total_results
                )


            with metric2:

                st.metric(
                    "Normal",
                    int(normal_results)
                )


            with metric3:

                st.metric(
                    "High",
                    int(high_results)
                )


            with metric4:

                st.metric(
                    "Low",
                    int(low_results)
                )


            # ---------------------------------------------
            # EXTRACTED TEXT PREVIEW
            # ---------------------------------------------

            with st.expander(
                "🔍 View Extracted Report Text"
            ):

                st.text(
                    extracted_text
                )


        else:

            st.warning(
                "No structured laboratory results were detected "
                "in this PDF."
            )


    except Exception as error:

        st.error(
            f"Unable to process the uploaded PDF: {error}"
        )


    finally:

        # Remove temporary uploaded file
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

<div class="card-icon">
📋
</div>

<div class="card-title">
Automated Extraction
</div>

<div class="card-text">
Extract structured laboratory information directly
from uploaded clinical PDF reports.
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        """
<div class="info-card">

<div class="card-icon">
📊
</div>

<div class="card-title">
Lab Value Analysis
</div>

<div class="card-text">
Review laboratory values, reference ranges,
and automatically detected result status.
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with col3:

    st.markdown(
        """
<div class="info-card">

<div class="card-icon">
💬
</div>

<div class="card-title">
Ask Questions
</div>

<div class="card-text">
Ask natural-language questions about information
contained directly inside your clinical report.
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with col4:

    st.markdown(
        """
<div class="info-card">

<div class="card-icon">
🔎
</div>

<div class="card-title">
Grounded Answers
</div>

<div class="card-text">
Retrieve supporting document evidence before
generating AI-assisted answers.
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