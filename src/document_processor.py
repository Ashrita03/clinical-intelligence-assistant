import re
from pathlib import Path

import pandas as pd
import pymupdf


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF, using Tesseract OCR
    when a page has little or no selectable text.
    """
    import io
    import pytesseract
    from PIL import Image

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page in document:
            page_text = page.get_text("text").strip()

            if len(page_text) < 30:
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False
                )

                image = Image.open(
                    io.BytesIO(pixmap.tobytes("png"))
                )

                ocr_text = pytesseract.image_to_string(
                    image,
                    lang="eng"
                ).strip()

                if len(ocr_text) > len(page_text):
                    page_text = ocr_text

            if page_text:
                pages.append(page_text)

    return "\\n\\n".join(pages).strip()


# =========================================================
# REPORT TYPE DETECTION
# =========================================================

REPORT_TYPE_KEYWORDS = {
    "Laboratory Report": [
        "reference range",
        "laboratory",
        "lab result",
        "blood test",
        "hemoglobin",
        "glucose",
        "creatinine",
        "platelet",
        "sodium",
        "potassium",
    ],

    "Cardiology Report": [
        "cardiology",
        "cardiac",
        "ecg",
        "ekg",
        "echocardiogram",
        "ejection fraction",
        "heart rate",
        "coronary",
        "ventricle",
    ],

    "Dental Report": [
        "dental",
        "dentist",
        "tooth",
        "teeth",
        "gingiva",
        "gingivitis",
        "periodontal",
        "caries",
        "cavity",
        "oral examination",
    ],

    "Dermatology / Hair Report": [
        "dermatology",
        "dermatologist",
        "skin",
        "scalp",
        "hair loss",
        "alopecia",
        "lesion",
        "rash",
        "acne",
        "dermatitis",
    ],

    "Gastroenterology Report": [
        "gastroenterology",
        "gastrointestinal",
        "stomach",
        "abdomen",
        "abdominal",
        "gastric",
        "colon",
        "endoscopy",
        "colonoscopy",
        "esophagus",
        "intestinal",
    ],

    "Radiology / Imaging Report": [
        "radiology",
        "x-ray",
        "xray",
        "ultrasound",
        "ct scan",
        "computed tomography",
        "mri",
        "magnetic resonance",
        "imaging",
        "radiograph",
    ],

    "Discharge Summary": [
        "discharge summary",
        "discharge diagnosis",
        "discharge medications",
        "hospital course",
        "admission date",
        "discharge date",
    ],

    "Clinical Consultation": [
        "chief complaint",
        "history of present illness",
        "assessment",
        "plan",
        "physical examination",
        "medical history",
        "clinical impression",
    ],
}


def detect_report_type(text):
    """
    Estimate the medical report category using transparent
    keyword-based scoring.

    The function returns the best matching category.
    If no category has sufficient evidence, it returns
    'General Medical Report'.
    """

    if not text or not text.strip():
        return "Unknown / Empty Report"

    normalized_text = text.lower()

    scores = {}

    for report_type, keywords in REPORT_TYPE_KEYWORDS.items():

        score = 0

        for keyword in keywords:
            if keyword in normalized_text:
                score += 1

        scores[report_type] = score

    best_report_type = max(
        scores,
        key=scores.get
    )

    if scores[best_report_type] == 0:
        return "General Medical Report"

    return best_report_type


# =========================================================
# LAB RESULT EXTRACTION
# =========================================================

def extract_lab_results(text):
    """
    Extract laboratory values that follow the currently
    supported structured format.

    Example:

    Glucose: 108 mg/dL
    Reference Range: 70 - 99
    """

    if not text:
        return []

    lab_results = []

    pattern = (
        r"([A-Za-z][A-Za-z0-9 ()/%+\-]*?):\s*"
        r"(-?[\d.]+)\s*([^\n]+)\n"
        r"\s*Reference Range:\s*"
        r"(-?[\d.]+)\s*-\s*(-?[\d.]+)"
    )

    matches = re.findall(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    for match in matches:

        (
            test_name,
            value,
            unit,
            reference_low,
            reference_high,
        ) = match

        try:
            value = float(value)
            reference_low = float(reference_low)
            reference_high = float(reference_high)

        except ValueError:
            continue

        if value < reference_low:
            status = "Low"

        elif value > reference_high:
            status = "High"

        else:
            status = "Normal"

        lab_results.append(
            {
                "test_name": test_name.strip(),
                "value": value,
                "unit": unit.strip(),
                "reference_low": reference_low,
                "reference_high": reference_high,
                "status": status,
            }
        )

    return lab_results


def lab_results_to_dataframe(lab_results):
    """
    Convert extracted laboratory results into a DataFrame.
    """

    columns = [
        "test_name",
        "value",
        "unit",
        "reference_low",
        "reference_high",
        "status",
    ]

    return pd.DataFrame(
        lab_results,
        columns=columns,
    )


# =========================================================
# REPORT METADATA
# =========================================================

def build_report_metadata(text, lab_results=None):
    """
    Create lightweight metadata describing the uploaded
    medical report.

    This metadata will later power the adaptive dashboard.
    """

    if lab_results is None:
        lab_results = []

    report_type = detect_report_type(text)

    word_count = len(
        re.findall(r"\b\w+\b", text)
    )

    character_count = len(text)

    high_count = sum(
        result.get("status") == "High"
        for result in lab_results
    )

    low_count = sum(
        result.get("status") == "Low"
        for result in lab_results
    )

    normal_count = sum(
        result.get("status") == "Normal"
        for result in lab_results
    )

    return {
        "report_type": report_type,
        "word_count": word_count,
        "character_count": character_count,
        "lab_results_detected": len(lab_results),
        "high_results": high_count,
        "low_results": low_count,
        "normal_results": normal_count,
    }


# =========================================================
# COMPLETE MEDICAL PDF PROCESSING
# =========================================================

def process_medical_pdf(pdf_path):
    """
    Process a medical PDF and return general document
    information.

    Returns
    -------
    dict
        {
            "text": ...,
            "report_type": ...,
            "metadata": ...,
            "lab_results": ...,
            "lab_dataframe": ...
        }
    """

    text = extract_text_from_pdf(pdf_path)

    lab_results = extract_lab_results(text)

    lab_dataframe = lab_results_to_dataframe(
        lab_results
    )

    metadata = build_report_metadata(
        text,
        lab_results
    )

    return {
        "text": text,
        "report_type": metadata["report_type"],
        "metadata": metadata,
        "lab_results": lab_results,
        "lab_dataframe": lab_dataframe,
    }


# =========================================================
# BACKWARD COMPATIBILITY
# =========================================================

def process_clinical_pdf(pdf_path):
    """
    Backward-compatible wrapper used by the current
    Streamlit application.

    This keeps the existing app working while the interface
    is gradually upgraded to the new general medical-report
    processing system.
    """

    result = process_medical_pdf(
        pdf_path
    )

    return (
        result["text"],
        result["lab_dataframe"],
    )