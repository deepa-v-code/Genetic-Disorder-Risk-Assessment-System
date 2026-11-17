"""Print saved model evaluation metrics."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
metrics_path = ROOT / "reports" / "model_metrics.json"

if not metrics_path.exists():
    raise SystemExit("Metrics not found. Run: python src/train.py")

metrics = json.loads(metrics_path.read_text())
print("MODEL EVALUATION")
print("----------------")
print(f"Training accuracy: {metrics['training_accuracy']:.2%}")
print(f"Testing accuracy:  {metrics['testing_accuracy']:.2%}")
print("\nClassification report:")
for label, values in metrics["classification_report"].items():
    if isinstance(values, dict):
        print(f"{label}: precision={values['precision']:.3f}, "
              f"recall={values['recall']:.3f}, f1={values['f1-score']:.3f}")
print("\nConfusion matrix:")
for row in metrics["confusion_matrix"]:
    print(row)
