# Responsible Credit AI

A responsive, local web prototype for **Regulatory-Aware AI for Explainable and Auditable Credit Decisions**. It uses the UCI Statlog German Credit benchmark dataset.

## What it does

- Trains Logistic Regression, Random Forest, Extra Trees, and Histogram Gradient Boosting on the **German Credit dataset**.
- Compares the full 20-feature set with a predeclared 8-feature compact baseline using repeated stratified 5-fold cross-validation.
- Holds out 20% of the records for a final evaluation, not model selection.
- Chooses a bad-risk classification threshold from out-of-fold training predictions using UCI's asymmetric error-cost matrix.
- Reports accuracy, majority baseline, balanced accuracy, bad-risk precision/recall/F1, ROC AUC, Brier score, confusion matrix, and expected cost.
- Offers responsive prediction, one-feature reference sensitivity explanations, group metrics, human-review actions, a hash-linked audit log, and input-distribution monitoring.

The model predicts the dataset's historical good/bad risk label. It does **not** make or recommend actual loan approval decisions.

## Dataset and attribution

The source is the UCI [Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog%2Bgerman%2Bcredit%2Bdata), 1,000 cases with 20 input attributes. The project's `data/raw/german.data` is included for reproducibility. Category labels follow the UCI data documentation.

Cite the source in coursework as: Hofmann, H. (1994). *Statlog (German Credit Data)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77.

## Run it on Windows

Python 3.10+ is required. From this project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m app.train_model
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. API reference: <http://127.0.0.1:8000/docs>.

The training run writes `app/model.joblib` and `reports/model_report.json`. Local predictions and reviews are stored in `data/audit.sqlite`. The public benchmark file is included under `data/raw/`; no web download is required.

## Learn and train in Jupyter

Open `notebooks/train_and_evaluate.ipynb` in VS Code or Jupyter using this project's `.venv` kernel. Run its cells from top to bottom. It explores the labelled data, runs the same leakage-safe training and held-out evaluation as `python -m app.train_model`, then displays the saved model comparison, metrics, confusion matrices, and a sample prediction. Training can take several minutes. The final cell verifies that the web app's `app/model.joblib` was updated.

The model does **not** learn from an ordinary website prediction: a prediction has no known correct answer. To improve it with additional examples, those examples need verified labels, and the training and evaluation process must be rerun on a versioned dataset. The local demo's review buttons record human-review actions; they do not create training labels or change the model.

## How evaluation avoids the notebook's main pitfalls

1. Stratified train/test split occurs before model fitting or feature selection.
2. Preprocessing lives inside each scikit-learn pipeline and is fit within CV folds.
3. Model and feature-set comparison uses repeated stratified 5-fold CV on training rows only.
4. The cost-aware threshold uses out-of-fold training predictions; the test labels do not choose the model, features, or threshold.
5. The test set is used once for the final report and group-level descriptive checks.
6. Correct UCI category meanings are centralized in `app/dataset.py`. Numeric ordinal fields are included in the model.
7. Fairness rates flag small groups; the 1,000-row benchmark does not establish real-world fairness.

## Important limitations

- The dataset is small and historical. CV and test estimates have uncertainty; one split cannot prove generalization.
- The threshold follows the dataset's stated cost ratio (false good-risk classification is five times the cost of false bad-risk classification). That is a benchmark assumption, not a bank policy.
- Explanations are one-feature what-if sensitivities against a typical training profile. They are not causal claims or SHAP values.
- Fairness reports describe errors across dataset groups; they are not a legal compliance verdict.
- Drift monitoring compares the last local demo inputs to training distributions. It cannot detect model accuracy drift without new labelled outcomes.
- The app is for local classroom use, has no login, and must not store real applicant details or be deployed as a lending system.

## Tests

```powershell
python -m pytest
```
