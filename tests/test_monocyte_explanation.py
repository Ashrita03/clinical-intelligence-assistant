from src.ai_worker import build_monocyte_explanation


def test_high_monocyte_explanation():
    labs = [
        {
            "test_name": "MO%",
            "value": 11.0,
            "unit": "%",
            "reference_low": 1.7,
            "reference_high": 9.3,
            "status": "High",
        }
    ]

    question = (
        "What does a high MO% mean, "
        "and what are some possible causes?"
    )

    answer = build_monocyte_explanation(question, labs)

    assert answer is not None
    assert "monocyte percentage" in answer.lower()
    assert "11.0" in answer
    assert "1.7" in answer
    assert "9.3" in answer
    assert "infections" in answer.lower()
    assert "inflammation" in answer.lower()
    assert "cannot establish a diagnosis" in answer.lower()


def test_unrelated_question_not_intercepted():
    labs = [
        {
            "test_name": "MO%",
            "value": 11.0,
            "unit": "%",
            "reference_low": 1.7,
            "reference_high": 9.3,
            "status": "High",
        }
    ]

    assert build_monocyte_explanation(
        "What does hemoglobin mean?", labs
    ) is None
