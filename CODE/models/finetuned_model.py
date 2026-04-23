from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

ID2LABEL = {0: "negative", 1: "positive"}
MAX_LENGTH = 128


class FinetunedPredictor:
    def __init__(self, model_path=None):
        """
        Default path:
        models/distilbert-finetuned/best_model
        """
        project_root = Path(__file__).resolve().parent.parent

        if model_path is None:
            model_path = project_root / "models" / "distilbert-finetuned" / "best_model"
        else:
            model_path = Path(model_path)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred = torch.argmax(probs).item()

        return {
            "label": pred,
            "confidence": round(probs[pred].item(), 4),
            "label_text": ID2LABEL[pred],
        }
