"""
30-Day Hospital Readmission Risk — Deployment App

Loads the trained model from train_model.py and serves it as an
interactive Gradio web app. Run train_model.py first to generate
readmission_model.pkl and model_columns.pkl.

Run: python app.py
"""

import joblib
import pandas as pd
import gradio as gr

model = joblib.load("readmission_model.pkl")
model_columns = joblib.load("model_columns.pkl")

AGE_BRACKETS = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]


def predict_readmission(
    time_in_hospital, num_medications, num_lab_procedures, number_diagnoses, age
):
    """Build a single-row input matching the model's training columns."""
    row = pd.DataFrame([dict.fromkeys(model_columns, 0)])

    row["time_in_hospital"] = time_in_hospital
    row["num_medications"] = num_medications
    row["num_lab_procedures"] = num_lab_procedures
    row["number_diagnoses"] = number_diagnoses

    age_col = f"age_{age}"
    if age_col in row.columns:
        row[age_col] = 1

    prob = model.predict_proba(row)[0][1]
    return f"Readmission risk: {prob:.1%}"


demo = gr.Interface(
    fn=predict_readmission,
    inputs=[
        gr.Number(label="Days in hospital"),
        gr.Number(label="Number of medications"),
        gr.Number(label="Number of lab procedures"),
        gr.Number(label="Number of diagnoses"),
        gr.Dropdown(AGE_BRACKETS, label="Age bracket"),
    ],
    outputs="text",
    title="30-Day Readmission Risk",
    description=(
        "Estimates a patient's risk of hospital readmission within 30 days "
        "of discharge. Built on the UCI Diabetes 130-US Hospitals dataset. "
        "Portfolio/learning project — not validated for clinical use."
    ),
)

if __name__ == "__main__":
    demo.launch(share=True)
