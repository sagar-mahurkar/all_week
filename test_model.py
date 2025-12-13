import unittest
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import recall_score

class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load model and data once for all tests"""
        base_dir = Path(__file__).resolve().parent  # ✅ FIXED PATH

        cls.model_path = base_dir / "artifacts" / "model.joblib"
        cls.data_path = base_dir / "data.csv"
        cls.target_column = "species"

        if not cls.model_path.exists():
            raise RuntimeError(f"Model not found at {cls.model_path}")

        if not cls.data_path.exists():
            raise RuntimeError(f"Data not found at {cls.data_path}")

        cls.model = joblib.load(cls.model_path)
        cls.input_data = pd.read_csv(cls.data_path)

    def test_data_integrity_check(self):
        required_features = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]

        for col in required_features:
            self.assertIn(col, self.input_data.columns)

    def test_model_recall_threshold(self):
        X = self.input_data.drop(self.target_column, axis=1)
        y = self.input_data[self.target_column]

        predictions = self.model.predict(X)
        recall = recall_score(y, predictions, average="macro")

        self.assertGreaterEqual(
            recall,
            0.80,
            f"Model recall too low: {recall}"
        )

if __name__ == "__main__":
    unittest.main()
