# 30-Day Hospital Readmission Risk Prediction

A complete, end-to-end data science project: problem framing → leakage-aware
data cleaning → baseline vs. model comparison → deployed interactive app.

**Live demo:** https://52bc95c131131c1e45.gradio.live (the Gradio link from
testing is temporary)

## Problem

Hospitals want to reduce unnecessary 30-day readmissions. If a discharge
nurse could flag high-risk patients at discharge, the hospital could invest
extra follow-up (a call, closer discharge planning) in the patients most
likely to need it.

- **Decision:** discharge nurses use the model's output to prioritize
  follow-up calls and discharge planning for high-risk patients.
- **Target:** was the patient readmitted within 30 days of discharge? (binary)
- **Baseline:** always predict "not readmitted" (the majority class).
- **Metric:** PR-AUC and recall — not accuracy. Only ~7-11% of patients are
  readmitted within 30 days, so a model that always says "no" scores ~90%+
  accuracy while catching zero at-risk patients. Accuracy is actively
  misleading on imbalanced problems like this one.

## Data

[Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296)
(UCI Machine Learning Repository, CC BY 4.0) — ~100K hospital encounters for
patients diagnosed with diabetes across 130 US hospitals, 1999-2008.

## Approach

1. **Data audit** — checked class balance, missing data by column, and
   duplicate patients before writing a single line of modeling code.
2. **Leakage fix** — ~30K rows belonged to patients with multiple hospital
   visits. Kept only each patient's first encounter so no patient's data
   could appear in both train and test sets.
3. **Baseline** — a `DummyClassifier` predicting the majority class every
   time, to set the floor any real model has to beat.
4. **Models compared** — Logistic Regression vs. Random Forest, evaluated on
   PR-AUC (not accuracy) given the ~7% positive rate.
5. **Deployment** — a Gradio web app that takes patient details and returns
   a readmission risk score, so the model is usable by someone who isn't a
   data scientist.

## Results

| Model | PR-AUC | Recall (class 1) | Precision (class 1) |
|---|---|---|---|
| Baseline (majority class) | 0.071 | 0.00 | — |
| Logistic Regression | **0.139** | 0.54 | 0.11 |
| Random Forest | 0.131 | 0.56 | 0.11 |

Logistic Regression and Random Forest performed roughly equivalently,
suggesting the signal in this feature set is close to linear — a more
complex model doesn't unlock extra performance here. Final model: Logistic
Regression, for its simplicity and interpretability.

**What the numbers mean in practice:** the model catches about half of all
true 30-day readmissions, at roughly 1-in-9 flagged patients actually being
readmitted (precision 0.11). Compared to calling everyone (0% efficient) or
no one (0% recall), this turns an undifferentiated patient population into
a meaningfully prioritized list. Whether a ~9:1 false-positive ratio is
worth the follow-up cost is a decision for hospital operations, not a
modeling question — this project surfaces the number needed to have that
conversation.

## Repo structure

```
├── README.md
├── requirements.txt
├── src/
│   ├── train_model.py      # data prep, leakage fix, baseline, model comparison
│   └── app.py               # Gradio deployment app
```

## How to run

```bash
pip install -r requirements.txt
python src/train_model.py      # downloads data, trains model, saves readmission_model.pkl
python src/app.py              # launches local Gradio app
```

## Limitations & next steps

This is a learning/portfolio project, not a clinically validated tool:

- Dataset is ~20 years old and US-specific; would not generalize as-is to
  other healthcare systems without revalidation.
- Precision (0.11) is low in absolute terms — a real deployment would need
  threshold tuning against the hospital's actual follow-up capacity.
- Dataset includes race and gender; any real deployment would need a
  fairness audit across these groups before use.
- Next step for real improvement: feature engineering (e.g., prior
  admission count, specific high-risk diagnosis flags) rather than further
  model swapping — model choice plateaued quickly in this project.

## Author

Abdul-Hakeem Adeoye Adediran, medical doctor and data scientist applying data science to clinical problems.
