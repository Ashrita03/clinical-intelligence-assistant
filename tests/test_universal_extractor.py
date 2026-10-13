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


def test_unit_between_value_and_range():
    results = extract_labs_from_rows([
        make_row("Marker Alpha", "18.5", "units", "2.0-10.0")
    ])
    assert len(results) == 1
    assert results[0]["value"] == 18.5
    assert results[0]["unit"] == "units"
    assert results[0]["status"] == "High"


def test_upper_limit_reference_range():
    results = extract_labs_from_rows([
        make_row("Marker Beta", "22.8", "mg/L", "Up to 6")
    ])
    assert len(results) == 1
    assert results[0]["status"] == "High"


def test_variable_number_of_measurements():
    rows = [
        make_row(
            f"Marker {i}",
            str(i + 1),
            "0-20",
            "units",
        )
        for i in range(17)
    ]

    results = extract_labs_from_rows(rows)

    assert len(results) == 17
    assert len({r["test_name"] for r in results}) == 17


def test_value_and_unit_in_same_cell():
    results = extract_labs_from_rows([
        make_row("Marker Gamma", "8.4 mg/dL", "3.0-7.0")
    ])
    assert len(results) == 1
    assert results[0]["value"] == 8.4
    assert results[0]["unit"] == "mg/dL"


def test_detect_potentially_missed_measurements():
    from src.universal_extractor import assess_extraction_quality

    rows = [
        make_row("Glucose", "108", "70-99", "mg/dL"),
        make_row("Vitamin B12", "425", "pg/mL"),
        make_row("Patient ID", "123456"),
        make_row("Report Date", "2026-10-12"),
    ]

    extracted = extract_labs_from_rows(rows)
    assessment = assess_extraction_quality(rows, extracted)

    assert assessment["extracted_count"] == 2
    assert assessment["potentially_missed_count"] == 0
    assert assessment["potentially_missed_rows"] == []
    assert any(
        result["test_name"] == "Vitamin B12"
        and result["value"] == 425
        and result["status"] == "Not assessed"
        for result in extracted
    )


def test_hba1c_without_numeric_reference_range():
    results = extract_labs_from_rows([
        make_row(
            "HBA1c RESULT",
            "6.2",
            "%",
            "Normal Glucose tolerance",
            "HPLC",
        )
    ])

    assert len(results) == 1
    assert results[0]["test_name"] == "HBA1c RESULT"
    assert results[0]["value"] == 6.2
    assert results[0]["unit"] == "%"
    assert results[0]["reference_low"] is None
    assert results[0]["reference_high"] is None
    assert results[0]["status"] == "Not assessed"


def test_estimated_average_glucose_without_reference_range():
    results = extract_labs_from_rows([
        make_row("ESTIMATED AVG. GLUCOSE", "131", "mg/dl")
    ])

    assert len(results) == 1
    assert results[0]["value"] == 131
    assert results[0]["unit"] == "mg/dl"
    assert results[0]["status"] == "Not assessed"


def test_administrative_identifiers_are_not_lab_results():
    results = extract_labs_from_rows([
        make_row("Patient ID", "123456", "ID"),
        make_row("Report Date", "2026-10-12", "Date"),
    ])

    assert results == []
