import unittest

from src.ai_worker import build_abnormal_lab_answer


class TestAbnormalLabAnswers(unittest.TestCase):

    def test_identifies_high_mo_not_normal_ne(self):
        labs = [
            {
                "test_name": "MO%",
                "value": 11.0,
                "unit": "%",
                "reference_low": 1.7,
                "reference_high": 9.3,
                "status": "High",
            },
            {
                "test_name": "NE%",
                "value": 53.9,
                "unit": "%",
                "reference_low": 40.0,
                "reference_high": 70.0,
                "status": "Normal",
            },
        ]

        answer = build_abnormal_lab_answer(
            "Which laboratory measurement is outside the reported reference range?",
            labs,
        )

        self.assertIsNotNone(answer)
        self.assertIn("MO%", answer)
        self.assertIn("11", answer)
        self.assertIn("High", answer)
        self.assertNotIn("NE%", answer)


if __name__ == "__main__":
    unittest.main()
