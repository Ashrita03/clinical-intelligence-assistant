from src.ai_worker import build_safe_lab_answer


def test_safe_lab_answer_uses_structured_results():
    report_text = """
    SYNTHETIC LABORATORY REPORT
    Test | Result | Unit | Normal Interval
    Marker Alpha | 18.5 | units | 2.0 - 10.0
    """

    lab_results = [
        {
            "test_name": "Marker Alpha",
            "value": 18.5,
            "unit": "units",
            "reference_low": 2.0,
            "reference_high": 10.0,
            "status": "High",
        }
    ]

    answer = build_safe_lab_answer(
        report_text,
        "What does Marker Alpha indicate?",
        lab_results=lab_results,
    )

    assert answer is not None
    assert "Marker Alpha" in answer
    assert "18.5" in answer
    assert "2.0" in answer
    assert "10.0" in answer


def test_answer_question_uses_structured_results(monkeypatch):
    from src import ai_worker

    lab_results = [
        {
            "test_name": "Marker Alpha",
            "value": 18.5,
            "unit": "units",
            "reference_low": 2.0,
            "reference_high": 10.0,
            "status": "High",
        }
    ]

    def fake_worker(module, payload):
        if module == "src.rag_worker":
            return {
                "success": True,
                "evidence": [
                    {"chunk": "Marker Alpha: 18.5 units"}
                ],
            }
        raise AssertionError(
            f"Unexpected model call: {module}"
        )

    monkeypatch.setattr(
        ai_worker,
        "run_worker",
        fake_worker,
    )

    result = ai_worker.answer_question(
        report_text="Marker Alpha | 18.5 | units | 2.0 - 10.0",
        question="What does Marker Alpha indicate?",
        lab_results=lab_results,
    )

    assert result["success"] is True
    assert "Marker Alpha" in result["answer"]
    assert "18.5" in result["answer"]
    assert "10.0" in result["answer"]
