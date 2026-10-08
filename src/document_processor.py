import re
from pathlib import Path
import pymupdf
import pandas as pd


def extract_text_from_pdf(pdf_path):
    """
    Extract text from all pages of a PDF document.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.

    Returns
    -------
    str
        Extracted text from the PDF.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    document = pymupdf.open(pdf_path)

    extracted_text = ""

    for page in document:
        extracted_text += page.get_text()

    document.close()

    return extracted_text
def extract_lab_results(text):
    """
    Extract laboratory test values and reference ranges
    from clinical report text.
    """

    lab_results = []

    pattern = (
        r"([A-Za-z ]+):\s*([\d.]+)\s*([^\n]+)\n"
        r"Reference Range:\s*([\d.]+)\s*-\s*([\d.]+)"
    )

    matches = re.findall(pattern, text)

    for match in matches:
        test_name, value, unit, reference_low, reference_high = match

        value = float(value)
        reference_low = float(reference_low)
        reference_high = float(reference_high)

        if value < reference_low:
            status = "Low"
        elif value > reference_high:
            status = "High"
        else:
            status = "Normal"

        lab_results.append({
            "test_name": test_name.strip(),
            "value": value,
            "unit": unit.strip(),
            "reference_low": reference_low,
            "reference_high": reference_high,
            "status": status
        })

    return lab_results
def lab_results_to_dataframe(lab_results):
    """
    Convert extracted laboratory results into a Pandas DataFrame.
    """

    return pd.DataFrame(lab_results)
def process_clinical_pdf(pdf_path):
    """
    Process a clinical PDF from extraction to structured lab results.
    """

    text = extract_text_from_pdf(pdf_path)
    lab_results = extract_lab_results(text)
    lab_dataframe = lab_results_to_dataframe(lab_results)

    return text, lab_dataframe