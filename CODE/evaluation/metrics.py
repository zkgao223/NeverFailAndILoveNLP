from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(predictions, model_name):
    true_labels = [p["true_label"] for p in predictions]
    pred_labels = [p["predicted_label"] for p in predictions]

    cm = confusion_matrix(true_labels, pred_labels)
    report_dict = classification_report(
        true_labels,
        pred_labels,
        target_names=["Negative", "Positive"],
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        true_labels,
        pred_labels,
        target_names=["Negative", "Positive"],
        zero_division=0,
    )

    results = {
        "model": model_name,
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels, zero_division=0),
        "recall": recall_score(true_labels, pred_labels, zero_division=0),
        "f1": f1_score(true_labels, pred_labels, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }

    print(f"\n=== {model_name} ===")
    print(report_text)
    print(f"Confusion Matrix:\n{cm}")

    return results
