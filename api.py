"""Flask prediction API. Educational demonstration only."""
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, request, jsonify

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "decision_tree_pipeline.joblib"
app = Flask(__name__)
model = None

def get_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Run python src/train.py first.")
        model = joblib.load(MODEL_PATH)
    return model

@app.get("/health")
def health():
    return jsonify({"status": "ok", "message": "Educational demo API is running."})

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object containing patient attributes."}), 400
    required = [
        "Age", "Gender", "Family_History", "Consanguinity",
        "Genetic_Test_Result", "Symptom_Severity", "Developmental_Delay",
        "Neurological_Symptoms", "Metabolic_Abnormality", "Chromosomal_Abnormality"
    ]
    missing = [c for c in required if c not in payload]
    if missing:
        return jsonify({"error": "Missing fields", "fields": missing}), 400
    try:
        row = pd.DataFrame([{c: payload[c] for c in required}])
        fitted = get_model()
        prediction = str(fitted.predict(row)[0])
        probabilities = fitted.predict_proba(row)[0]
        classes = fitted.named_steps["classifier"].classes_
        probability_map = {str(k): round(float(v), 4) for k, v in zip(classes, probabilities)}
        return jsonify({
            "predicted_category": prediction,
            "class_probabilities": probability_map,
            "notice": "Synthetic educational output only; not a medical diagnosis."
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
