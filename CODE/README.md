# COMP6713 Sentiment Analysis

Binary sentiment analysis comparing five models: SentiWordNet baseline, zero-shot DistilBERT, zero-shot RoBERTa, fine-tuned DistilBERT and fine-tuned RoBERTa. Includes a Gradio web demo and cli inference script.

## Project Structure

```
CODE/
├── data/
│   ├── build_dataset.py          # Download and preprocess datasets
│   ├── clean.py                  # Text cleaning
│   ├── domain_analysis.py
│   └── label_mapping.py          # GoEmotions to binary label mapping
├── models/
│   ├── baseline.py               # SentiWordNet lexicon baseline
│   ├── pretrained_distilbert.py  # Zero-shot DistilBERT (SST-2)
│   ├── pretrained_roberta.py     # Zero-shot RoBERTa (Twitter)
│   ├── finetune_distilbert.py    # Fine-tune DistilBERT + W&B hyperparameter sweep
│   ├── finetune_roberta.ipynb    # Fine-tune RoBERTa + W&B hyperparameter sweep
│   ├── finetuned_model.py        # Shared model loader for fine-tuned checkpoints
│   ├── run_predictions.py        # Batch inference on test set
│   ├── distilbert-finetuned/
│   │   └── best_model/           # Fine-tuned DistilBERT checkpoint (see below)
│   └── roberta-finetuned/
│       └── best_model/           # Fine-tuned RoBERTa checkpoint (see below)
├── evaluation/
│   ├── metrics.py                # Evaluation metrics utilities
│   ├── cross_domain_eval.py      # Cross-domain evaluation
│   └── error_analysis.py        # Error analysis
├── pre-trained_model_analysis/
│   ├── diff_analysis.py          # Difference analysis between model predictions
│   └── evaluate_metrics.py      # Evaluate and visualise pre-trained model metrics
├── demo/
│   ├── app.py                    # Gradio web interface
│   └── predict.py                # CLI inference script
└── requirements.txt
```

## Setup

**Python 3.10 / 3.11**

```bash
pip install -r requirements.txt
```

For the baseline model, download NLTK data once:

```python
import nltk
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("sentiwordnet")
```

## Fine-tuned Model Checkpoint Download

Required files in `models/distilbert-finetuned/best_model/`:
`config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `vocab.txt`, `training_args.bin`

Required files in `models/roberta-finetuned/best_model/`:
`config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `vocab.txt`, `training_args.bin`, `merges.txt`

### Recommended: Google Drive

Download both model checkpoints from the link below and place the files into the corresponding paths:

```
https://drive.google.com/drive/folders/13V4ibSTlBUi7hmaRvE6TnD9R1wCGuUWb?usp=sharing
```

### Alternative: Git LFS

The checkpoints are also tracked via Git LFS. After cloning, run:

```bash
git clone https://github.com/zkgao223/comp6713-sentiment-analysis.git
cd comp6713-sentiment-analysis
git lfs pull
```

Note: large files are not always downloaded automatically on clone, so `git lfs pull` may be required.

## Usage

**Prepare datasets** (downloads GoEmotions + IMDb automatically):

```bash
python -m data.build_dataset
```

**Domain analysis** (optional, for report analysis):

```bash
python -m data.domain_analysis
```

**Batch predictions on test set:**

```bash
python -m models.run_predictions
```

**Fine-tune DistilBERT** (requires `wandb login`; GPU recommended):

```bash
python -m models.finetune_distilbert --debug   # CPU, 200 samples
python -m models.finetune_distilbert           # full training
python -m models.finetune_distilbert --sweep   # W&B hyperparameter sweep (4 runs)
```

**Fine-tune RoBERTa** (requires `wandb login`; GPU recommended):

```bash
# Run the cells in models/finetune_roberta.ipynb
```

**CLI inference:**

```bash
python demo/predict.py --text "Great movie!" --model finetuned-distilbert
```

Three options available:

- `--text`: input string for prediction.
- `--file`: path to a file where each line is treated as a separate input. Either `--text` or `--file` is required; if both are provided, `--file` is ignored.
- `--model`: model to use. Accepted values: `baseline`, `distilbert`, `roberta`, `finetuned-roberta`, `finetuned-distilbert` (default).

**Gradio demo:**

```bash
python demo/app.py
```
