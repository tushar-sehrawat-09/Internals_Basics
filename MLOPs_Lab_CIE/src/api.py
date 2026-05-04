from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle, json
import uvicorn

print("Loading model...")

app = FastAPI()

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("results/step1_s1.json") as f:
    task1 = json.load(f)
model_name = task1["best_model"]

print(f"Model loaded: {model_name}")

class Features(BaseModel):
    pr_lines_changed: int = Field(..., ge=10, le=2000)
    reviewer_load: int = Field(..., ge=1, le=15)
    file_count: int = Field(..., ge=1, le=50)
    is_critical_path: int = Field(..., ge=0, le=1)

@app.get("/health")
def health():
    return {"status": "running", "model": model_name, "version": "1.0"}

@app.post("/predict")
def predict(features: Features):
    data = [[
        features.pr_lines_changed,
        features.reviewer_load,
        features.file_count,
        features.is_critical_path
    ]]
    prediction = model.predict(data)[0]
    return {"prediction": round(float(prediction), 4)}

print("Starting server on port 9000...")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
