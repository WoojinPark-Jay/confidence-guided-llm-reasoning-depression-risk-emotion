# Confidence-Guided LLM Reasoning for Depression-Risk Emotion Classification

This repository organizes the code and data workflow for a Reddit-based depression-risk-related proxy emotion classification project.

The project studies a confidence-guided two-phase framework for research-oriented proxy emotion classification in Reddit text. The codebase is being organized step by step so collaborators can reproduce the data preparation, preprocessing, and later modeling experiments.

## Pipeline Stages

### Stage 0. Subreddit Source Data Preparation

Notebook:

- `notebooks/00_reddit_subreddit_data_preparation.ipynb`

Script:

- `src/prepare_subreddit_data.py`

Purpose:

- Load subreddit-level Reddit archive dumps or existing raw parquet files.
- Convert `created_utc` timestamps.
- Select common post-level columns.
- Save subreddit-level prepared parquet files under `data/01_subreddit_preparation/`.

This stage consolidates the older per-subreddit notebooks such as `reddit_data_preprocessing_depression.ipynb`, `reddit_data_preprocessing_AnxietyDepression.ipynb`, and related notebooks.

### Stage 1. Whole Reddit Dataset Preprocessing

Notebook:

- `notebooks/01_reddit_data_preprocessing.ipynb`

Script:

- `src/preprocess_reddit.py`

Purpose:

- Load 11 subreddit-level preprocessed parquet files.
- Build the three proxy classes: Depression, Neutral, and Happy.
- Concatenate title and body text.
- Clean and normalize text.
- Compute TextBlob polarity scores.
- Apply sentiment-aware filtering.
- Apply class balancing.
- Export final modeling data and summary tables.

### Stage 2. Modeling

Local sample notebooks:

- `notebooks/local/02_distilbert_classification_sample_fine_tuning.ipynb`
- `notebooks/local/03_llama_classification_sample_fine_tuning.ipynb`
- `notebooks/local/04_mistral_classification_sample_fine_tuning.ipynb`

Colab large-run notebooks:

- `notebooks/colab/10_distilbert_confidence_threshold_wandb_colab.ipynb`
- `notebooks/colab/11_llama2_confidence_threshold_wandb_colab.ipynb`
- `notebooks/colab/12_mistral_confidence_threshold_wandb_colab.ipynb`

Script helpers:

- `src/modeling_data.py`
- `src/colab_notebooks/*.py`

Purpose:

- DistilBERT baseline and fine-tuning workflow.
- Llama 2 / Mistral first-stage classifier workflows.
- Load the final preprocessed dataset from `data/02_preprocessing_outputs/`.
- Sample a configurable number of records per class.
- Start with `SAMPLES_PER_CLASS = 1000` for quick testing.
- Increase the value later, for example to 20000 or 40000, for larger runs.
- Export sampled train/validation/test files under `data/03_modeling_inputs/`.
- Report both standard metrics and direct prediction counts, for example `Correct predictions: 267 / 300`.

Current default modeling size:

- `SAMPLES_PER_CLASS = 1000`
- `TRAIN_RATIO = 0.75`
- `VALIDATION_RATIO = 0.15`
- `TEST_RATIO = 0.10`
- 1000 Depression, 1000 Neutral, and 1000 Happy records
- 3000 total records before splitting
- 2250 train records, 450 validation records, and 300 test records
- These values are defined near the top of each modeling notebook so collaborators can change the run size and split ratios in one place.

Local CPU/Mac execution note:

- The Llama and Mistral notebooks keep the original full-model workflow for GPU runs.
- When CUDA is not available, they automatically switch to tiny debug checkpoints so collaborators can verify the notebook flow locally.
- To run the full Llama or Mistral checkpoints, use a CUDA GPU environment and disable the local tiny-model fallback only after confirming the environment can load the full model.


### Stage 3. Supplementary Mixed Emotion Stress-Test Dataset

Dataset files:

- `data/supplementary/mixed_emotion/mixed_emotion_stress_test_v2_2_300.csv`
- `data/supplementary/mixed_emotion/mixed_emotion_stress_test_v2_2_300.xlsx`
- `data/supplementary/mixed_emotion/mixed_emotion_stress_test_v2_2_300.jsonl`

Script:

- `scripts/generate_mixed_emotion_dataset.py`

Manuscript support:

- `docs/mixed_emotion_dataset_v2_2_manuscript_insert.md`
- `data/supplementary/mixed_emotion/appendix_mixed_emotion_dataset_protocol.md`

Purpose:

- Provide a controlled synthetic stress-test set for emotionally ambiguous examples.
- Include 300 examples, balanced across Depression, Neutral, and Happy proxy emotion labels.
- Include five ambiguity scenario types: blended emotion co-occurrence, positive-to-distress shift, distress-to-recovery shift, neutral framing with subtle affect, and conflicting cues with a dominant trajectory.
- Use this dataset only for supplementary robustness evaluation, not for Phase 1 training, hyperparameter tuning, or confidence-threshold selection.

## Project Layout

```text
confidence-guided-selective-llm-reasoning/
  data/
    00_raw_reddit_archives/           # local archive dumps or raw parquet files; ignored by Git
    01_subreddit_preparation/         # subreddit-level prepared parquet files; ignored by Git
    02_preprocessing_outputs/         # final preprocessing CSV outputs; ignored by Git
    03_modeling_inputs/               # sampled train/validation/test CSVs; ignored by Git
    supplementary/
      mixed_emotion/                  # 300-example synthetic mixed-emotion stress-test dataset
  notebooks/
    00_reddit_subreddit_data_preparation.ipynb
    01_reddit_data_preprocessing.ipynb
    local/
      02_distilbert_classification_sample_fine_tuning.ipynb
      03_llama_classification_sample_fine_tuning.ipynb
      04_mistral_classification_sample_fine_tuning.ipynb
    colab/
      10_distilbert_confidence_threshold_wandb_colab.ipynb
      11_llama2_confidence_threshold_wandb_colab.ipynb
      12_mistral_confidence_threshold_wandb_colab.ipynb
  src/
    prepare_subreddit_data.py
    preprocess_reddit.py
    modeling_data.py
    colab_notebooks/
      distilbert_confidence_threshold_wandb_colab.py
      llama2_confidence_threshold_wandb_colab.py
      mistral_confidence_threshold_wandb_colab.py
  scripts/
    generate_mixed_emotion_dataset.py
  docs/
  reports/figures/
  requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The preprocessing script downloads required NLTK resources on first run if they are missing.

## Run Stage 0

From the project root:

```bash
python src/prepare_subreddit_data.py
```

Expected outputs:

- `data/01_subreddit_preparation/*_preprocessed.parquet`
- `data/01_subreddit_preparation/subreddit_preparation_summary.csv`

## Run Stage 1

From the project root:

```bash
python src/preprocess_reddit.py
```

Expected outputs:

- `data/02_preprocessing_outputs/final_preprocessed_whole_df.csv`
- `data/02_preprocessing_outputs/final_preprocessed_df.csv`
- `data/02_preprocessing_outputs/class_counts_before_filtering.csv`
- `data/02_preprocessing_outputs/class_counts_after_filtering.csv`
- `data/02_preprocessing_outputs/final_class_distribution.csv`
- `data/02_preprocessing_outputs/filtering_summary.csv`

## Run Stage 2

Open one of the model notebooks and start with:

```python
SAMPLES_PER_CLASS = 1000
```

The modeling notebooks use this value to sample the same number of records from each class. For example, `1000` creates a 3000-record modeling set before train/validation/test splitting.

Default split:

- Train: 75%, or 2250 records when `SAMPLES_PER_CLASS = 1000`
- Validation: 15%, or 450 records when `SAMPLES_PER_CLASS = 1000`
- Test: 10%, or 300 records when `SAMPLES_PER_CLASS = 1000`

Each model evaluation prints the metric dictionary, the direct number of correct predictions, the number of incorrect predictions, and a normalized confusion matrix.

Example evaluation output:

```text
Correct predictions: 267 / 300 (89.00%)
Incorrect predictions: 33 / 300
```

For DistilBERT, the hidden-state visualization section uses `FEATURE_EXTRACTION_SAMPLE_SIZE = 300` by default to keep local exploratory analysis faster. This limit applies to the feature-extraction/visualization step, not to the main fine-tuning split.

The Llama and Mistral notebooks use the explicit train/validation/test CSV splits generated from these ratios. They do not re-split the train set after the CSV split, so the final notebook training/evaluation counts remain aligned with the values above.

Generated modeling inputs:

- `data/03_modeling_inputs/sample_1000_per_class/train_dataset.csv`
- `data/03_modeling_inputs/sample_1000_per_class/validation_dataset.csv`
- `data/03_modeling_inputs/sample_1000_per_class/test_dataset.csv`

## Run Stage 2 In Colab

Use the notebooks under `notebooks/colab/` for larger GPU training runs.

Recommended Colab order:

1. Open the target notebook from GitHub in Colab.
2. Run the dependency installation cell, then restart the runtime once if Colab asks.
3. Set `SAMPLES_PER_CLASS` near the top of the notebook. The default is `300` per class for a smoke test; increase it to values such as `1000`, `20000`, or `40000` for larger runs.
4. Keep `WANDB_SWEEP_MODE = "new"` for an independent sweep. Use `continue_existing` only when you have access to the target W&B entity/project/sweep.
5. Do not paste W&B tokens into notebook source code. Add `WANDB_API_KEY` in Colab Secrets or enter it only when the secure `wandb.login()` prompt appears.

The Colab notebooks save W&B sweep results, predictions, metrics, threshold tables, figures, and final model outputs into their local Colab output directories. These generated outputs are ignored by Git and should be shared through W&B artifacts, cloud storage, or another agreed research storage location.


## Run Stage 3 / Inspect Mixed Emotion Dataset

The supplementary Mixed Emotion Dataset v2.2 is committed because it is small and intended to support reproducible stress-test evaluation.

Open the spreadsheet version directly:

```text
data/supplementary/mixed_emotion/mixed_emotion_stress_test_v2_2_300.xlsx
```

Or regenerate the dataset from the project root:

```bash
python scripts/generate_mixed_emotion_dataset.py
```

Dataset design summary:

- Depression: 100 examples
- Neutral: 100 examples
- Happy: 100 examples
- Total: 300 examples
- Scenario types: 5
- Examples per scenario type: 60
- Intended use: supplementary controlled stress-test only
- Not used for training or threshold selection

## Important Git Note

The local parquet, zst, and generated CSV files are large. They are intentionally ignored by Git.

For GitHub collaboration, use Git LFS, cloud storage, or an agreed shared data location rather than committing large data directly.

## Current Data Status

The local working copy currently contains the available Reddit archive/raw files and the subreddit-level preprocessed parquet files needed to run the notebooks. These data files are not pushed to GitHub.

Collaborators should place shared data files in the paths documented in `data/README.md`.
