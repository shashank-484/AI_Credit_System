"""German Credit data loading, schema, and human-readable category labels."""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "raw" / "german.data"

COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "dependents", "telephone", "foreign_worker", "credit_risk",
]

CATEGORICAL_FEATURES = [
    "checking_status", "credit_history", "purpose", "savings", "employment",
    "personal_status_sex", "other_debtors", "property", "other_installment_plans",
    "housing", "job", "telephone", "foreign_worker",
]
NUMERIC_FEATURES = [
    "duration", "credit_amount", "installment_rate", "residence_since", "age",
    "existing_credits", "dependents",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

FIELD_LABELS = {
    "checking_status": "Checking account status",
    "duration": "Loan duration (months)",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "credit_amount": "Credit amount (DM)",
    "savings": "Savings account / bonds",
    "employment": "Employment duration",
    "installment_rate": "Installment rate (% of disposable income)",
    "personal_status_sex": "Personal status and sex",
    "other_debtors": "Other debtors / guarantors",
    "residence_since": "Years at current residence (coded)",
    "property": "Property",
    "age": "Age (years)",
    "other_installment_plans": "Other installment plans",
    "housing": "Housing",
    "existing_credits": "Existing credits at this bank",
    "job": "Job category",
    "dependents": "People supported by applicant",
    "telephone": "Telephone registered in applicant's name",
    "foreign_worker": "Foreign worker status",
}

CATEGORY_LABELS = {
    "checking_status": {
        "A11": "Overdrawn / balance below 0 DM",
        "A12": "Balance from 0 to under 200 DM",
        "A13": "At least 200 DM / salary assignment",
        "A14": "No checking account",
    },
    "credit_history": {
        "A30": "No prior credit / paid duly",
        "A31": "All credits at this bank paid duly",
        "A32": "Existing credits paid duly so far",
        "A33": "Past payment delays",
        "A34": "Critical account / other existing credits",
    },
    "purpose": {
        "A40": "New car", "A41": "Used car", "A42": "Furniture / equipment",
        "A43": "Radio / television", "A44": "Domestic appliances", "A45": "Repairs",
        "A46": "Education", "A48": "Retraining", "A49": "Business", "A410": "Other",
    },
    "savings": {
        "A61": "Under 100 DM", "A62": "100 to under 500 DM", "A63": "500 to under 1,000 DM",
        "A64": "At least 1,000 DM", "A65": "Unknown / no savings account",
    },
    "employment": {
        "A71": "Unemployed", "A72": "Under 1 year", "A73": "1 to under 4 years",
        "A74": "4 to under 7 years", "A75": "7 years or more",
    },
    "personal_status_sex": {
        "A91": "Male: divorced / separated", "A92": "Female: divorced / separated / married",
        "A93": "Male: single", "A94": "Male: married / widowed", "A95": "Female: single",
    },
    "other_debtors": {"A101": "None", "A102": "Co-applicant", "A103": "Guarantor"},
    "property": {
        "A121": "Real estate", "A122": "Building savings / life insurance",
        "A123": "Car or other property", "A124": "Unknown / no property",
    },
    "other_installment_plans": {"A141": "Bank", "A142": "Stores", "A143": "None"},
    "housing": {"A151": "Rent", "A152": "Own", "A153": "Free of charge"},
    "job": {
        "A171": "Unemployed / unskilled non-resident", "A172": "Unskilled resident",
        "A173": "Skilled employee / official", "A174": "Management / self-employed / highly qualified",
    },
    "telephone": {"A191": "No telephone", "A192": "Registered in applicant's name"},
    "foreign_worker": {"A201": "Yes", "A202": "No"},
    "installment_rate": {"1": "At least 35%", "2": "25% to under 35%", "3": "20% to under 25%", "4": "Under 20%"},
    "residence_since": {"1": "Under 1 year", "2": "1 to under 4 years", "3": "4 to under 7 years", "4": "7 years or more"},
}


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Read the UCI German Credit whitespace-delimited 20-feature file."""
    if not path.exists():
        raise FileNotFoundError(
            f"German Credit data not found at {path}. Place the UCI german.data file in data/raw/."
        )
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, engine="python")
    if frame.shape != (1000, 21):
        raise ValueError(f"Expected 1,000 rows and 21 columns; found {frame.shape[0]} rows and {frame.shape[1]} columns.")
    y = frame.pop("credit_risk").astype(int).map({1: 0, 2: 1})
    if y.isna().any():
        raise ValueError("Target must contain only 1 (good risk) and 2 (bad risk).")
    X = frame[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="raise")
    return X, y


def display_value(field: str, value: object) -> str:
    value_str = str(value)
    return CATEGORY_LABELS.get(field, {}).get(value_str, value_str)
