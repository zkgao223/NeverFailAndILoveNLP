# Member Contribution


## Tianji Liu (z5613025)

### 1.Data Cleaning (`CODE/data/clean.py`)

Details:

- Implemented a text cleaning function to remove noise such as HTML tags, URLs, Reddit mentions, and normalize whitespace for consistent input format.



### 2.Label Mapping  (`CODE/data/label_mapping.py`)

Details:

- Designed a mapping function to convert GoEmotions multi-label annotations into binary sentiment labels by defining positive and negative categories and filtering out neutral or ambiguous samples.



### 3. Data Processing & Dataset Construction (`CODE/data/build_dataset.py`)

Details:

- Built a complete data preprocessing pipeline using HuggingFace datasets. Built the training and validation sets from the GoEmotions training split, then performed a stratified 80/20 split.
- Converted multi-label emotions into **binary sentiment labels (positive/negative)** using a custom mapping strategy (`label_mapping.py`), filtering ambiguous samples.
- Applied text cleaning (`clean.py`) to remove HTML tags, URLs, Reddit mentions, and normalize whitespace.
- Constructed standardized datasets (`train.json`, `val.json`, `test.json`) with unified format `{ "text": ..., "label": ... }`.
- Performed **stratified train-validation splitting** to maintain label distribution.
- Used the IMDb test split as the final test set.
- Generated a dataset summary report (`data_splits_report.txt`) with key statistics.



### 4. Domain Analysis (`CODE/data/domain_analysis.py`)

Details:

- Implemented a domain analysis module to compare training and testing datasets (train/val/test splits) to quantify domain shift.
- Computed label distribution statistics (positive/negative ratios) across train, validation, and test splits.
- Analyzed text length characteristics using descriptive statistics (mean, median, std, percentiles).
- Performed vocabulary-level analysis, including token counts, unique vocabulary size, and lexical richness (hapax).
- Measured **cross-domain vocabulary overlap** using Jaccard similarity to identify shared and domain-specific terms.
- Conducted **out-of-vocabulary (OOV) analysis** to quantify unseen words in the test set.
- Analyzed text style features (sentence count, punctuation, uppercase and digit ratios).
- Grouped samples into length buckets and computed label-conditioned statistics.
- Aggregated multiple metrics into a **domain shift summary** for overall comparison.
- Generated analysis outputs (`domain_analysis_report.txt`, `domain_analysis_results.json`) for reproducibility.

---

## Zixu Cheng (z5628034)

### Role
- Responsible for building the three sentiment models, running reproducible experiments, and conducting in-depth prediction consistency analysis.

### 1. Model Implementation (`CODE/models/baseline.py`, `CODE/models/pretrained_roberta.py`, `CODE/models/pretrained_distilbert.py`)
Details:
- Developed the NLTK/SentiWordNet lexicon-based baseline model using sentiment score aggregation.
- Built fine-tuning and inference pipelines for RoBERTa and DistilBERT pre-trained models.
- Standardized model input/output structures and ensured consistent coding standards across all models.
- Established unified prediction interfaces to support automated batch inference and result output.

### 2. Experiment Execution & Result Generation (`CODE/models/run_predictions.py`)
Details:
- Developed a one-click batch prediction script to automate the inference workflow for all three models.
- Loaded and validated the standardized test dataset (`test.json`), confirming a total of 25,000 test samples.
- Executed end-to-end inference on the full test set to generate structured prediction results for each model.
- Saved the baseline model predictions as a standalone JSON file (`baseline_predictions.json`).
- Stored the RoBERTa and DistilBERT predictions in a unified JSON file (`pretrained_predictions.json`) for centralized management.
- Generated standardized JSON outputs containing text content, true labels, and predicted labels uniformly.

### 3. Prediction Consistency & Discrepancy Analysis (`CODE/pre-trained model analysis/diff_analysis.py`)
Details:
- Designed and implemented quantitative analysis to measure prediction agreement across three models.
- Calculated consensus rates and classified three types of prediction disagreement patterns.
- Quantified the proportion of fully consistent samples and model-specific divergent cases.
- Identified performance weaknesses of the lexicon-based baseline by analyzing divergent samples.
- Provided data support for evaluating model reliability and cross-method performance gaps.

### 4. Result Visualization & Performance Metrics Calculation (`CODE/pre-trained model analysis/metrics_visualization.py`)
Details:
- Loaded standardized model prediction results from JSON files (`baseline_predictions.json`, `pretrained_predictions.json`).
- Extracted and cleaned valid true labels and predicted labels from all model outputs, ensuring data consistency for evaluation.
- Calculated core quantitative metrics, including accuracy, precision, recall, and weighted F1-score for the three models.
- Generated a formatted console performance report, comparing metric values and calculating performance improvements between models.
- Designed a grouped bar chart to visualize performance differences across the NLTK baseline, RoBERTa, and DistilBERT models.
- Saved the high-resolution comparison chart (`final_metrics_comparison.png`, 300 DPI) for project reports and presentation slides.
- Provided complete quantitative and visual support for model performance analysis and conclusion summarization.

---

## Yanyan Zhu (z5565492)

### 1. DistilBERT Fine-tuning Pipeline (`CODE/models/finetune_distilbert.py`)
Details:
- Built a complete DistilBERT fine-tuning pipeline (based on `distilbert-base-uncased`) using HuggingFace Trainer API.
- Integrated W&B for experiment tracking and hyperparameter search using grid search (`method: grid`). The search space included:
    - `learning_rate`: [2e-5, 3e-5]
    - `num_train_epochs`: [2, 3]
    - `per_device_train_batch_size`: 32 (fixed)
    - `weight_decay`: 0.01 (fixed)
- 4 **sweep trials** in total, using F1 score as the optimisation objective.
- Best hyperparameters found: `learning_rate = 3e-5, epochs = 2`, achieving `F1 = 0.929` and `Accuracy = 0.908` on the GoEmotions validation set.
- Provided 3 modes to run the training code: `--debug` (CPU, 200 samples), default (full training), and `--sweep` (W&B grid search).
- Resolved GCP T4 GPU unavailability by migrating training to Google Colab.

### 2. Inference Interface (`CODE/models/finetune_distilbert.py` - `FinetunedPredictor`)
Details:
- Implemented a unified `FinetunedPredictor` class as a shared inference interface for Members D and E.
- Accepts a single text string and returns `label` (0/1), `confidence` (float), and `label_text` (`negative`/`positive`).
- Loads model and tokenizer from a local checkpoint path with automatic `model.eval()` and `torch.no_grad()`.

### 3. Model Distribution
Details:
- Uploaded trained model weights (`model.safetensors`, `config.json`, tokenizer files) to GitHub via Git LFS and to Google Cloud Storage (`gs://comp6713-sentiment-data/models/distilbert-finetuned/`).
- Backed up model checkpoint to Google Drive for team access.
- Verified `config.json` correctly contains `id2label`, `label2id`, and `num_labels` fields for downstream compatibility.

---

## Zekai Gao (z5548102)

### 1. Fine-tuning RoBERTa (`CODE/models/finetune_roberta.ipynb`)
Details:
- Build a integrated RoBERTa fine-tuning pipeline (based on roberta-base)
- Integrated W&B for hyperparameter search using Bayesian Optimization (`method: bayes`). The searching space including:
    - `learning rate`: log-uniform distribution, [1e-5, 5e-5]
    - `batch_size`: [32, 64, 128] (the training run on 24GB RTX 4090)
    - `num_train_epochs`: [2, 3, 4]
    - `weigh_decay`: [0.0, 0.01, 0.1]
    - `warmup_ratio`: uniform distribution, [0.0, 0.2]
- 20 **sweep trials** in total and used the F1 score as optimal object.
- Final best HP found: `learning rate = 3e-5, epochs = 2`, and the F1 score on validation dataset achieved `F1 = 0.929`
- Provided 3 way to run the training code (Debug/Single/Sweep).

### 2. CLI Inference Script (`CODE/demo/predict.py`)
Details:
- Implemented a CLI script that supports **5 models**: `baseline`, `distilbert`, `roberta`, `finetuned-distilbert` and `finetuned-roberta`.
- Two supported input way: `--text` for single text and `--file` for batch processing a file that contains one text each line.
- Output the prediction label, label text and confidence.

### 3. Gradio Demo (`CODE/demo/app.py`)
Details:
- Build a visual and interactive Web interface using Gradio Blocks. It supports 6 options (5 single model options and a "All Models" option).
- Implemented a multi-model prediction result comparision table (**in "All Model" schema**): shows the prediction and confidence from each model.
- Implemented a **"Majority Voting"** mechanism: Provide final prediction through voting by four models (DistilBERT, RoBERTa, Fine-tuned DistilBERT and Fine-tuned RoBERTa)
- Set up several sample inputs for quick test.

---

## QinglinTan (z5579853)

### 1. Quantitative Evaluation (`CODE/evaluation/metrics.py`)
Details:

- Calculated the main evaluation metrics for the models, including Accuracy, Precision, Recall, and F1 score.
- Compared the baseline model, pre-trained models, and fine-tuned model under the same test setting.
- Organised the evaluation results into a clear format for model comparison.



### 2. Cross-domain Evaluation (`CODE/evaluation/cross_domain_eval.py`)
Details:

- Conducted cross-domain evaluation by testing the models on IMDb movie reviews.
- Analysed model performance when training and testing data came from different domains.
- Helped explain the effect of domain shift on the final results.



### 3. Error Analysis (`CODE/evaluation/error_analysis.py`)
Details:

- Manually checked misclassified examples from the model outputs.
- Grouped common error types, such as negation, sarcasm, domain mismatch, and ambiguous cases.
- Used these examples to explain why some predictions were incorrect.



### 4. Report Writing (`Final Report`)
Details:

- Wrote the evaluation-related parts and a little bit another part of the report.
- Helped describe the quantitative results, cross-domain findings, and error analysis.
- Contributed to the discussion and conclusion section based on the evaluation results.