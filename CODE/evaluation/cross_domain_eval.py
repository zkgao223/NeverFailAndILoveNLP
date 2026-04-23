import csv
import json
from pathlib import Path

from evaluation.metrics import evaluate_model
from models.finetuned_model import FinetunedPredictor

MODEL_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "MISC"


def load_json(path):
   # Load a JSON file from disk.
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    # Save Python data to a JSON file.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_predictions(predictor, test_data):
    # Run one fine-tuned model on the full test set
    # and keep the output format consistent with the other models.

    predictions = []
    for item in test_data:
        result = predictor.predict(item["text"])
        predictions.append({
            "text": item["text"],
            "true_label": item["label"],
            "predicted_label": result["label"],
            "confidence": result.get("confidence")
        })
    return predictions


def build_summary_rows(all_results):
    # Keep only the metrics we want in the comparison table.

    rows = []
    for result in all_results:
        rows.append({
            "Model": result["model"],
            "Accuracy": f"{result['accuracy']:.4f}",
            "Precision": f"{result['precision']:.4f}",
            "Recall": f"{result['recall']:.4f}",
            "F1": f"{result['f1']:.4f}",
        })
    return rows


def save_csv_table(rows, path):
    # Write the summary table to a CSV file. more beautiful
    if not rows:
        return

    fieldnames = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_markdown_table(rows, path):
    # Write the summary table to a Markdown file
    if not rows:
        return

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        lines.append(
            f"| {row['Model']} | {row['Accuracy']} | {row['Precision']} | "
            f"{row['Recall']} | {row['F1']} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary_table(rows):
    # Print a simple aligned table in the terminal.
    if not rows:
        print("No rows to display.")
        return

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    col_widths = {}

    for header in headers:
        max_width = len(header)
        for row in rows:
            max_width = max(max_width, len(str(row[header])))
        col_widths[header] = max_width

    def format_row(row_dict):
        return " | ".join(
            str(row_dict[h]).ljust(col_widths[h]) for h in headers
        )

    print("\n=== Evaluation Summary Table ===")
    print(format_row({h: h for h in headers}))
    print("-+-".join("-" * col_widths[h] for h in headers))
    for row in rows:
        print(format_row(row))


def main():
    baseline_path = PROJECT_ROOT / "MISC" / "baseline_predictions.json"
    pretrained_path = PROJECT_ROOT / "MISC" / "pretrained_predictions.json"
    test_path = PROJECT_ROOT / "MISC" / "test.json"

    distilbert_model_path = (
        PROJECT_ROOT / "MISC" / "distilbert-finetuned" / "best_model"
    )
    roberta_model_path = (
        PROJECT_ROOT / "MISC" / "roberta-finetuned" / "best_model"
    )

    # Check all required inputs before running evaluation.
    required_paths = [
        baseline_path,
        pretrained_path,
        test_path,
        distilbert_model_path,
        roberta_model_path,
    ]
    for path in required_paths:
        if not path.exists():
            print(f"Missing required file/folder: {path}")
            return

    baseline_preds = load_json(baseline_path)
    pretrained = load_json(pretrained_path)
    test_data = load_json(test_path)

    # Make sure the zero-shot prediction file has the expected keys.
    required_pretrained_keys = ["roberta", "distilbert"]
    for key in required_pretrained_keys:
        if key not in pretrained:
            print(f"Missing key in pretrained_predictions.json: {key}")
            return

    # Load both fine-tuned models.
    distilbert_predictor = FinetunedPredictor(distilbert_model_path)
    roberta_predictor = FinetunedPredictor(roberta_model_path)

    # Run both fine-tuned models on the same IMDb test set.
    distilbert_finetuned_preds = build_predictions(
        distilbert_predictor, test_data
    )
    roberta_finetuned_preds = build_predictions(
        roberta_predictor, test_data
    )

    # Evaluate all models under the same metric pipeline.
    all_results = [
        evaluate_model(baseline_preds, "SentiWordNet Baseline"),
        evaluate_model(pretrained["roberta"], "RoBERTa (zero-shot)"),
        evaluate_model(pretrained["distilbert"],
                       "DistilBERT-SST2 (zero-shot)"),
        evaluate_model(distilbert_finetuned_preds, "DistilBERT (fine-tuned)"),
        evaluate_model(roberta_finetuned_preds, "RoBERTa (fine-tuned)")
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the full raw evaluation output.
    save_json(all_results, OUTPUT_DIR / "results.json")

    # Save prediction files for later error analysis.
    save_json(
        distilbert_finetuned_preds,
        OUTPUT_DIR / "distilbert_finetuned_predictions.json"
    )
    save_json(
        roberta_finetuned_preds,
        OUTPUT_DIR / "roberta_finetuned_predictions.json"
    )

    # Build a cleaner comparison table for the report.
    summary_rows = build_summary_rows(all_results)

    save_csv_table(summary_rows, OUTPUT_DIR / "results_table.csv")
    save_markdown_table(summary_rows, OUTPUT_DIR / "results_table.md")

    # Also print the table in the terminal for quick checking.
    print_summary_table(summary_rows)

    print("\nSaved files:")
    print(" -", OUTPUT_DIR / "results.json")
    print(" -", OUTPUT_DIR / "results_table.csv")
    print(" -", OUTPUT_DIR / "results_table.md")
    print(" -", OUTPUT_DIR / "distilbert_finetuned_predictions.json")
    print(" -", OUTPUT_DIR / "roberta_finetuned_predictions.json")


if __name__ == "__main__":
    main()
