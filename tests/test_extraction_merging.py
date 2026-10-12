from unittest.mock import patch

from src.document_processor import process_medical_pdf


def test_text_and_layout_results_are_preserved():
    text_result = {
        "test_name": "Hemoglobin",
        "value": 13.2,
        "unit": "g/dL",
        "reference_low": 12.0,
        "reference_high": 16.0,
        "status": "Normal",
    }

    layout_result = {
        "test_name": "Glucose",
        "value": 108.0,
        "unit": "mg/dL",
        "reference_low": 70.0,
        "reference_high": 99.0,
        "status": "High",
    }

    with (
        patch(
            "src.document_processor.extract_text_from_pdf",
            return_value="Synthetic laboratory report",
        ),
        patch(
            "src.document_processor.extract_lab_results",
            return_value=[text_result],
        ),
        patch(
            "src.universal_extractor.extract_pdf_layout",
            return_value=[],
        ),
        patch(
            "src.universal_extractor.group_layout_rows",
            return_value=[],
        ),
        patch(
            "src.universal_extractor.extract_labs_from_rows",
            return_value=[layout_result],
        ),
    ):
        result = process_medical_pdf("synthetic.pdf")

    names = {
        item["test_name"]
        for item in result["lab_results"]
    }

    assert names == {"Hemoglobin", "Glucose"}
    assert len(result["lab_dataframe"]) == 2
