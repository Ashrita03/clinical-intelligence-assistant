import unittest

from src.ml_predictor import (
    train_heart_disease_model,
    predict_heart_disease,
)


class TestMLPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = train_heart_disease_model()

    def test_model_prediction_structure(self):
        patient_data = {
            "age": 63,
            "trestbps": 145,
            "chol": 233,
            "thalach": 150,
            "oldpeak": 2.3,
            "ca": 0,
            "sex": 1,
            "cp": 1,
            "fbs": 1,
            "restecg": 2,
            "exang": 0,
            "slope": 3,
            "thal": 6,
        }

        result = predict_heart_disease(
            self.model,
            patient_data
        )

        self.assertIn("predicted_class", result)
        self.assertIn("probability", result)

    def test_prediction_values_are_valid(self):
        patient_data = {
            "age": 65,
            "trestbps": 160,
            "chol": 300,
            "thalach": 100,
            "oldpeak": 3.0,
            "ca": 3,
            "sex": 1,
            "cp": 4,
            "fbs": 1,
            "restecg": 2,
            "exang": 1,
            "slope": 3,
            "thal": 7,
        }

        result = predict_heart_disease(
            self.model,
            patient_data
        )

        self.assertIn(result["predicted_class"], [0, 1])
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
