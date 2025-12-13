import unittest
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import recall_score

class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load model and data once for all tests"""
        base_dir = Path(__file__).resolve().parent.parent

        cls.model_path = base_dir / "artifacts" / "model.joblib"
        cls.data_path = base_dir / "data.csv"
        cls.target_column = "species"

        try:
            cls.model = joblib.load(cls.model_path)
            cls.input_data = pd.read_csv(cls.data_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model or data: {e}")

    def test_data_integrity_check(self):
        """Ensure required columns exist in the dataset"""
        required_features = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]

        for col in required_features:
            self.assertIn(col, self.input_data.columns, f"Missing feature: {col}")

        self.assertIn(
            self.target_column,
            self.input_data.columns,
            "Target column missing from data",
        )

    def test_model_recall_threshold(self):
        """Model should achieve minimum macro recall"""
        X = self.input_data.drop(columns=[self.target_column])
        y_true = self.input_data[self.target_column]

        y_pred = self.model.predict(X)
        recall = recall_score(y_true, y_pred, average="macro")

        self.assertGreaterEqual(
            recall,
            0.85,
            f"Model recall too low: {recall:.3f}",
        )

if __name__ == "__main__":
    unittest.main()