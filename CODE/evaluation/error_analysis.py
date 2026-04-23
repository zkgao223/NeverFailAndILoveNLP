import json
import random
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_SIZE = 30
RANDOM_SEED = 42

MODEL_TO_FILE = {
    "distilbert": PROJECT_ROOT / "MISC" / "distilbert_finetuned_predictions.json",
    "roberta": PROJECT_ROOT / "MISC" / "roberta_finetuned_predictions.json",
}

MODEL_TO_OUTPUT = {
    "distilbert": PROJECT_ROOT / "MISC" / "error_analysis_distilbert.json",
    "roberta": PROJECT_ROOT / "MISC" / "error_analysis_roberta.json",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    # Create a command-line argument parser
    parser = argparse.ArgumentParser(
        description="Manual error analysis for fine-tuned models"
    )
    # Add a required argument called --model
    parser.add_argument(
        "--model",
        choices=["distilbert", "roberta"],
        required=True,
        help="Choose which fine-tuned model to analyse"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Select the input error file and output file
    error_file = MODEL_TO_FILE[args.model]
    output_file = MODEL_TO_OUTPUT[args.model]
    # Check whether the prediction file exists
    if not error_file.exists():
        print(f"Missing file: {error_file}")
        print("Please run cross_domain_eval.py first.")
        return

    preds = load_json(error_file)
    # true_label != predicted_label means the model got it wrong
    errors = [p for p in preds if p["true_label"] != p["predicted_label"]]
    print(f"Model: {args.model}")
    print(f"Total misclassified samples: {len(errors)}")

    random.seed(RANDOM_SEED)
    sample_errors = random.sample(errors, min(SAMPLE_SIZE, len(errors)))
    # Create a dictionary to store manually labelled error categories
    error_categories = {
        "negation": [],
        "sarcasm": [],
        "domain_mismatch": [],
        "ambiguous": [],
        "other": [],
    }

    for i, e in enumerate(sample_errors, start=1):
        print(f"\n[{i}/{len(sample_errors)}]")
        text = e["text"]

        print(f"Text length: {len(text)}")
        print(f"Text (start): {text[:700]}")
        print(f"Text (end): {text[-300:]}")
        print(f"True: {e['true_label']}, Pred: {e['predicted_label']}")
        if "confidence" in e and e["confidence"] is not None:
            print(f"Confidence: {e['confidence']}")

        cat = input(
            "Category (negation/sarcasm/domain_mismatch/ambiguous/other): "
        ).strip().lower()

        if cat not in error_categories:
            cat = "other"

        note = input("Short note about the error (optional): ").strip()

        record = {
            "text": e["text"],
            "true_label": e["true_label"],
            "predicted_label": e["predicted_label"],
            "confidence": e.get("confidence"),
            "note": note,
        }

        error_categories[cat].append(record)

    summary = {k: len(v) for k, v in error_categories.items()}

    print("\nSummary:")
    print(summary)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "sample_size": len(sample_errors),
                "summary": summary,
                "details": error_categories,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
