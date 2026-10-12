from src.universal_extractor import extract_labs_from_rows


def make_row(*texts):
    return {
        "page": 1,
        "items": [
            {"text": text, "bbox": [i * 100, 0, i * 100 + 80, 10]}
            for i, text in enumerate(texts)
        ],
    }


def test_standard_table():
    results = extract_labs_from_rows([
        make_row("Glucose", "108 H", "70-99", "mg/dL")
    ])
    assert len(results) == 1
    assert results[0]["status"] == "High"


def test_no_abnormal_flag():
    results = extract_labs_from_rows([
        make_row("Hemoglobin", "13.2", "12.0-16.0", "g/dL")
    ])
    assert len(results) == 1
    assert results[0]["status"] == "Normal"


def test_different_column_order():
    results = extract_labs_from_rows([
        make_row("Glucose", "70-99", "108", "mg/dL")
    ])
    assert len(results) == 1
    assert results[0]["value"] == 108.0
