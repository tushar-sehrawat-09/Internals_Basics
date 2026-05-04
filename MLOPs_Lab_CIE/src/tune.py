import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import json, os, pickle

print("Starting Task 2...")

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

with open("results/step1_s1.json") as f:
    task1 = json.load(f)
best_model_name = task1["best_model"]
print(f"Tuning: {best_model_name}")

df = pd.read_csv("data/training_data.csv")
X = df.drop("review_turnaround_hours", axis=1)
y = df["review_turnaround_hours"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 7, 15],
    "min_samples_split": [2, 4]
}

ModelClass = RandomForestRegressor if best_model_name == "RandomForest" else GradientBoostingRegressor
base_model = ModelClass(random_state=42)

print("Running grid search with 3-fold CV... please wait...")

mlflow.set_experiment("mergegate-review-turnaround-hours")

with mlflow.start_run(run_name="tuning-mergegate") as parent_run:
    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print("Grid search done!")

    for i, params in enumerate(grid.cv_results_["params"]):
        with mlflow.start_run(run_name=f"trial_{i}", nested=True):
            mlflow.log_params(params)
            score = -grid.cv_results_["mean_test_score"][i]
            mlflow.log_metric("cv_mae", score)

    best_params = grid.best_params_
    best_cv_mae = -grid.best_score_

    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)
    best_mae = mean_absolute_error(y_test, preds)

    mlflow.log_params(best_params)
    mlflow.log_metric("best_mae", best_mae)
    mlflow.log_metric("best_cv_mae", best_cv_mae)
    mlflow.sklearn.log_model(best_model, "model")
    print("Logged to MLflow!")

with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

total_trials = len(grid.cv_results_["params"])

output = {
    "search_type": "grid",
    "n_folds": 3,
    "total_trials": total_trials,
    "best_params": best_params,
    "best_mae": round(best_mae, 4),
    "best_cv_mae": round(best_cv_mae, 4),
    "parent_run_name": "tuning-mergegate"
}

with open("results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Total trials: {total_trials}")
print(f"Best params: {best_params}")
print(f"Best MAE: {best_mae:.4f}")
print(f"Best CV MAE: {best_cv_mae:.4f}")
print("Task 2 done! Results saved to results/step2_s2.json")
