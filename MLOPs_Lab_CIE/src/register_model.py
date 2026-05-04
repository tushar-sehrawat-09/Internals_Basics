import mlflow
from mlflow.tracking import MlflowClient
import json, os

print("Starting Task 4...")

with open("results/step2_s2.json") as f:
    task2 = json.load(f)

with open("results/step1_s1.json") as f:
    task1 = json.load(f)

client = MlflowClient()
experiment = client.get_experiment_by_name("mergegate-review-turnaround-hours")
print(f"Found experiment: {experiment.experiment_id}")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'tuning-mergegate'",
    order_by=["metrics.best_mae ASC"]
)

run_id = runs[0].info.run_id
best_mae = runs[0].data.metrics["best_mae"]
print(f"Found run: {run_id}")
print(f"Best MAE: {best_mae}")

model_name = "mergegate-review-turnaround-hours-predictor"
model_uri = f"runs:/{run_id}/model"

print("Registering model...")
result = mlflow.register_model(model_uri, model_name)
version = result.version
print(f"Model registered! Version: {version}")

output = {
    "registered_model_name": model_name,
    "version": int(version),
    "run_id": run_id,
    "source_metric": "mae",
    "source_metric_value": round(best_mae, 4)
}

with open("results/step4_s6.json", "w") as f:
    json.dump(output, f, indent=2)

print("Task 4 done! Results saved to results/step4_s6.json")