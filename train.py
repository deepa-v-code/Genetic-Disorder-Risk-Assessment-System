"""Train, tune, evaluate and serialize the Decision Tree pipeline."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import load_and_clean_data, TARGET

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

def train():
    data = load_and_clean_data()
    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42, class_weight="balanced"))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Small grid with regularization to reduce overfitting risk.
    params = {
        "classifier__max_depth": [3, 5, 7, None],
        "classifier__min_samples_leaf": [2, 5, 10],
        "classifier__ccp_alpha": [0.0, 0.001]
    }
    search = GridSearchCV(
        pipeline, params, cv=5, scoring="accuracy", n_jobs=-1,
        refit=True
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    metrics = {
        "training_accuracy": round(float(train_acc), 4),
        "testing_accuracy": round(float(test_acc), 4),
        "best_parameters": search.best_params_,
        "classification_report": classification_report(
            y_test, test_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "classes": model.named_steps["classifier"].classes_.tolist(),
        "dataset_rows": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test))
    }

    joblib.dump(model, MODEL_DIR / "decision_tree_pipeline.joblib")
    (REPORT_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Export readable tree diagram (first levels, to keep the image legible).
    fitted_preprocessor = model.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    clf = model.named_steps["classifier"]
    plt.figure(figsize=(22, 12))
    plot_tree(
        clf, feature_names=feature_names, class_names=clf.classes_,
        filled=True, rounded=True, max_depth=3, fontsize=7
    )
    plt.title("Decision Tree (first 4 levels) - Synthetic Educational Dataset")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "decision_tree.png", dpi=160)
    plt.close()

    importances = clf.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names, "importance": importances
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    top = importance_df.head(15).sort_values("importance")
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Feature importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_importance.png", dpi=160)
    plt.close()

    print("Training completed.")
    print(f"Training accuracy: {train_acc:.2%}")
    print(f"Testing accuracy:  {test_acc:.2%}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Saved model: {MODEL_DIR / 'decision_tree_pipeline.joblib'}")
    print(f"Saved reports in: {REPORT_DIR}")

if __name__ == "__main__":
    train()
