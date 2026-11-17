"""Streamlit interface for the Explainable Genetic Disorder Risk Assessment demo."""
from pathlib import Path
import json
import requests
import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

st.set_page_config(page_title="Genetic Disorder Risk Assessment Demo", page_icon="🧬", layout="wide")
st.title("🧬 Explainable Genetic Disorder Risk Assessment System")
st.caption("Decision Tree Classifier | Synthetic educational dataset")
st.warning(
    "Educational demonstration only. This tool does not assess real medical risk, "
    "does not diagnose genetic disorders, and must not guide healthcare decisions."
)

st.subheader("Enter synthetic example attributes")
with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        family = st.selectbox("Family history", ["Yes", "No"])
        consanguinity = st.selectbox("Consanguinity", ["Yes", "No"])
        genetic = st.selectbox("Genetic test result", ["Positive", "Negative", "Not Available"])
    with c2:
        severity = st.selectbox("Symptom severity", ["None", "Mild", "Moderate", "Severe"])
        developmental = st.selectbox("Developmental delay", ["Yes", "No"])
        neurological = st.selectbox("Neurological symptoms", ["Yes", "No"])
        metabolic = st.selectbox("Metabolic abnormality", ["Yes", "No"])
        chromosomal = st.selectbox("Chromosomal abnormality", ["Yes", "No"])
    submitted = st.form_submit_button("Run educational prediction")

if submitted:
    payload = {
        "Age": age, "Gender": gender, "Family_History": family,
        "Consanguinity": consanguinity, "Genetic_Test_Result": genetic,
        "Symptom_Severity": severity, "Developmental_Delay": developmental,
        "Neurological_Symptoms": neurological, "Metabolic_Abnormality": metabolic,
        "Chromosomal_Abnormality": chromosomal
    }
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=payload, timeout=10)
        result = response.json()
        if response.ok:
            st.success(f"Predicted synthetic category: {result['predicted_category']}")
            st.write("Model class probabilities (not real-world risk):")
            st.dataframe(pd.DataFrame(
                result["class_probabilities"].items(), columns=["Synthetic class", "Model probability"]
            ), use_container_width=True)
        else:
            st.error(result.get("error", "API request failed"))
    except requests.RequestException:
        st.error("Cannot connect to Flask API. Start it first using: python app/api.py")

st.divider()
st.subheader("Model explainability and performance")
metrics_file = REPORTS / "model_metrics.json"
importance_file = REPORTS / "feature_importance.csv"
tree_image = REPORTS / "decision_tree.png"
importance_image = REPORTS / "feature_importance.png"

if metrics_file.exists():
    metrics = json.loads(metrics_file.read_text())
    m1, m2, m3 = st.columns(3)
    m1.metric("Training accuracy", f"{metrics['training_accuracy']:.1%}")
    m2.metric("Testing accuracy", f"{metrics['testing_accuracy']:.1%}")
    m3.metric("Test records", metrics["test_rows"])
    st.caption("A large training/testing gap can indicate overfitting. Accuracy alone is not enough to assess a classifier.")
else:
    st.info("Train the model first: python src/train.py")

if importance_file.exists():
    st.markdown("**Important features**")
    st.dataframe(pd.read_csv(importance_file).head(15), use_container_width=True)
if importance_image.exists():
    st.image(str(importance_image), caption="Feature importance from the trained Decision Tree")
if tree_image.exists():
    st.image(str(tree_image), caption="Decision Tree visualization (first levels)")

st.subheader("How to interpret this demo")
st.write(
    "Feature importance describes how much each encoded input contributed to the fitted tree's "
    "splits on this synthetic dataset. It does not establish biological causation or clinical importance."
)
