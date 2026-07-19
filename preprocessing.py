"""
Preprocessing for the OULAD-style student performance dataset.

Key decision: `date_unregistration` and `id_student` are DROPPED.
- date_unregistration is populated (non -1) almost exclusively for students
  who withdrew -- it is recorded as a *consequence* of withdrawing, not a
  predictor available at decision time. Keeping it lets the model "cheat"
  by reading off the label instead of learning real early-warning signal.
- id_student is a row identifier with no genuine predictive meaning.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

LEAKY_COLS = ["date_unregistration", "id_student"]
TARGET = "final_result"

CATEGORICAL_COLS = [
    "code_module", "code_presentation", "gender", "region",
    "highest_education", "imd_band", "age_band", "disability",
]
NUMERIC_COLS = [
    "num_of_prev_attempts", "studied_credits", "avg_score", "max_score",
    "min_score", "total_assessments", "avg_weight", "total_clicks",
    "avg_clicks", "max_clicks", "total_interactions", "active_days",
    "date_registration", "score_range", "clicks_per_active_day",
    "interactions_per_assessment", "credits_per_attempt", "registered_late",
]

CLASS_ORDER = ["Distinction", "Fail", "Pass", "Withdrawn"]  # matches original y encoding


def engineer_features(df):
    df = df.copy()
    df["score_range"] = df["max_score"] - df["min_score"]
    df["clicks_per_active_day"] = df["total_clicks"] / df["active_days"].replace(0, np.nan)
    df["clicks_per_active_day"] = df["clicks_per_active_day"].fillna(0)
    df["interactions_per_assessment"] = df["total_interactions"] / df["total_assessments"].replace(0, np.nan)
    df["interactions_per_assessment"] = df["interactions_per_assessment"].fillna(0)
    df["credits_per_attempt"] = df["studied_credits"] / (df["num_of_prev_attempts"] + 1)
    df["registered_late"] = (df["date_registration"] > 0).astype(int)
    return df


def load_clean_dataset(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in LEAKY_COLS if c in df.columns])
    df = engineer_features(df)
    return df


def build_column_transformer():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )


def get_label_encoder():
    le = LabelEncoder()
    le.classes_ = pd.array(CLASS_ORDER)
    return le
