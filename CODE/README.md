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

Please login on wandb before running the training codes:
```bash
wandb login
```

## Fine-tuned Model Checkpoint Download

Required files in **`models/distilbert-finetuned/best_model/`**:
`config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `vocab.txt`, `training_args.bin`

Required files in **`models/roberta-finetuned/best_model/`**:
`config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `vocab.txt`, `training_args.bin`, `merges.txt`

### Recommended: Google Drive

Download both model checkpoints from the link below and place the files into the corresponding paths:

```
https://drive.google.com/drive/folders/13V4ibSTlBUi7hmaRvE6TnD9R1wCGuUWb?usp=sharing
```



## Dataset Setup

Please download the datasets from the following Google Drive link:
https://drive.google.com/drive/folders/1ks8KV4R3NTLBmdEHYXp890r2QTBTuEJ_?usp=drive_link

After downloading:

1. Extract the downloaded files
2. Place them into the following directories:

```
MISC/
├── go_emotions/
│   └── google-research-datasets___go_emotions/
│       ├── dataset_info.json
│       └── *.arrow
└── imdb/
    └── stanfordnlp___imdb/
        ├── dataset_info.json
        └── *.arrow
```

Notes:
- The script will not run if the structure is incorrect
- If `train.json`, `val.json`, and `test.json` already exist in the `MISC` directory, the script will skip dataset downloading and directly load these files instead of rebuilding the dataset.


## Usage

**Prepare datasets** ( GoEmotions + IMDb ):

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

## Important Requirement Before Running the Evaluation

Before running `cross_domain_eval.py`, you must first **download all files from the `models` folder in Google Drive into the `MISC` folder**.

Please make sure that:

- **all contents from the Google Drive `models` folder are downloaded completely into `MISC/`;**
- **none of the file names are changed;**
- **the original file names must stay exactly the same. For example: MISC/distilbert-finetuned/best_model.**

## Important Warning

**Do not rename any files from the `models` folder.**

The evaluation code depends on the original file names. If any file name is changed, the program may fail to load the model correctly.

## How to Run the Evaluation Code


Use the following command:

```bash
# Execute the following command once.
python -m evaluation.metrics
python -m evaluation.cross_domain_eval
python -m evaluation.error_analysis
```
