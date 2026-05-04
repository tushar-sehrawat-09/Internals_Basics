import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json, os, pickle

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/training_data.csv")
X = df.drop("review_turnaround_hours", axis=1)
y = df["review_turnaround_hours"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("mergegate-review-turnaround-hours")

results = []

models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("domain", "code_review")
        params = model.get_params()
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        results.append({"name": name, "mae": round(mae, 4), "rmse": round(rmse, 4)})
        print(f"{name} → MAE: {mae:.4f}, RMSE: {rmse:.4f}")

best = min(results, key=lambda x: x["mae"])

# Save best model
best_model = models[best["name"]]
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

output = {
    "experiment_name": "mergegate-review-turnaround-hours",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "mae",
    "best_metric_value": best["mae"]
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nTask 1 done! Results saved to results/step1_s1.json")
print(f"Best model: {best['name']} with MAE={best['mae']}")