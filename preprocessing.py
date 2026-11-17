"""Data loading and cleaning utilities for the educational demo."""
from pathlib import Path
import pandas as pd

TARGET = "Disorder_Category"

def load_and_clean_data(path=None):
    if path is None:
        path = Path(__file__).resolve().parents[1] / "dataset" / "genetic_disorder_synthetic.csv"
    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"Dataset must contain target column: {TARGET}")
    # Treat zero/negative ages as invalid and let the pipeline impute them.
    if "Age" in data.columns:
        data.loc[data["Age"] <= 0, "Age"] = pd.NA
    data = data.drop_duplicates().reset_index(drop=True)
    return data
