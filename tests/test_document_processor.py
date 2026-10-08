import unittest

from src.document_processor import extract_lab_results, lab_results_to_dataframe


class TestDocumentProcessor(unittest.TestCase):

    def setUp(self):
        self.sample_text = """
Hemoglobin: 13.2 g/dL
Reference Range: 12.0 - 16.0

Glucose: 108 mg/dL
Reference Range: 70 - 99

Creatinine: 0.9 mg/dL
Reference Range: 0.6 - 1.2
"""

    def test_extract_lab_results(self):
        results = extract_lab_results(self.sample_text)

        self.assertEqual(len(results), 3)

        self.assertEqual(results[0]["test_name"], "Hemoglobin")
        self.assertEqual(results[0]["value"], 13.2)

        self.assertEqual(results[1]["test_name"], "Glucose")
        self.assertEqual(results[1]["value"], 108.0)
        self.assertEqual(results[1]["unit"], "mg/dL")

        self.assertEqual(results[2]["test_name"], "Creatinine")
        self.assertEqual(results[2]["value"], 0.9)

    def test_lab_status(self):
        results = extract_lab_results(self.sample_text)
        df = lab_results_to_dataframe(results)

        glucose_status = df.loc[
            df["test_name"] == "Glucose", "status"
        ].iloc[0]

        hemoglobin_status = df.loc[
            df["test_name"] == "Hemoglobin", "status"
        ].iloc[0]

        creatinine_status = df.loc[
            df["test_name"] == "Creatinine", "status"
        ].iloc[0]

        self.assertEqual(glucose_status, "High")
        self.assertEqual(hemoglobin_status, "Normal")
        self.assertEqual(creatinine_status, "Normal")


if __name__ == "__main__":
    unittest.main()
