import unittest

from src.qa_pipeline import question_supported_by_context


class TestQAPipeline(unittest.TestCase):

    def setUp(self):
        self.context = """
        Patient ID: TEST-001
        Glucose: 108 mg/dL
        Reference Range: 70 - 99
        Potassium: 4.2 mmol/L
        Reference Range: 3.5 - 5.1
        """

    def test_supported_glucose_question(self):
        question = "What was the patient's glucose level?"

        result = question_supported_by_context(
            question,
            self.context
        )

        self.assertTrue(result)

    def test_supported_potassium_question(self):
        question = "What was the patient's potassium level?"

        result = question_supported_by_context(
            question,
            self.context
        )

        self.assertTrue(result)

    def test_unsupported_blood_pressure_question(self):
        question = "What was the patient's blood pressure?"

        result = question_supported_by_context(
            question,
            self.context
        )

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
