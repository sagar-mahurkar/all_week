# train.py
import os
import mlflow
import joblib
import pandas as pd

from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# -------------------------- Configuration --------------------------
MLFLOW_TRACKING_URI = "http://34.31.68.202:5000"
EXPERIMENT_NAME = "iris-experiment"
MODEL_NAME = "iris-random-forest"
RUN_NAME = "Random Forest Hyperparameter Search"

LOCAL_MODEL_DIR = "artifacts"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "random_forest_model.pkl")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# -------------------------- Data Preparation --------------------------
def prepare_data():
    """Load and split the Iris dataset."""
    data = pd.read_csv("data.csv")
    data = data[
        ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    ]

    train, test = train_test_split(
        data,
        test_size=0.2,
        stratify=data["species"],
        random_state=42,
    )

    X_train = train.drop(columns=["species"])
    y_train = train["species"]
    X_test = test.drop(columns=["species"])
    y_test = test["species"]

    return X_train, y_train, X_test, y_test

# -------------------------- Training & Logging --------------------------
def train_and_log_model(X_train, y_train, X_test, y_test):
    """Train, tune, and log a RandomForest model to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # param_grid = {
    #     "n_estimators": [50, 100],
    #     "criterion": ["gini", "entropy"],
    #     "max_depth": [None, 5, 10],
    #     "min_samples_split": [2, 5],
    # }

    param_grid = {
      "n_estimators": [200, 300],
      "max_depth": [None, 8, 12],
      "min_samples_split": [2, 4],
      "min_samples_leaf": [1, 2],
      "max_features": ["sqrt", "log2"],
      "bootstrap": [True],
    }

    with mlflow.start_run(run_name=RUN_NAME):
        base_model = RandomForestClassifier(random_state=42)

        grid = GridSearchCV(
          base_model,
          param_grid,
          cv=5,
          scoring="accuracy",
          n_jobs=-1,
          verbose=1,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # ---------------- Metrics & Params ----------------
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("cv_accuracy", grid.best_score_)
    mlflow.log_metric("test_accuracy", best_model.score(X_test, y_test))

    # ---------------- Model Signature ----------------
    input_example = X_train.iloc[:5]
    signature = infer_signature(X_train, best_model.predict(X_train))

    # ---------------- Log Model ----------------
    mlflow.sklearn.log_model(
      sk_model=best_model,
      name="model",
      input_example=input_example,
      signature=signature,
      registered_model_name=MODEL_NAME,
    )

    # ---------------- Save Locally ----------------
    joblib.dump(best_model, LOCAL_MODEL_PATH)

    print("✅ Model logged and registered successfully")

    return {
      "best_params": grid.best_params_,
      "cv_accuracy": grid.best_score_,
      "test_accuracy": best_model.score(X_test, y_test),
      "local_model_path": LOCAL_MODEL_PATH,
    }

# -------------------------- Main Entry --------------------------
def main():
    print("\n🚀 Starting Model Training...")
    X_train, y_train, X_test, y_test = prepare_data()
    results = train_and_log_model(X_train, y_train, X_test, y_test)
    print("\n✅ Training Complete!")
    print(results)

if __name__ == "__main__":
    main()
