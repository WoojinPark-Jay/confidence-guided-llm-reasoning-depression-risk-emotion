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

Notebooks:

- `notebooks/02_distilbert_classification_sample_fine_tuning.ipynb`
- `notebooks/03_llama_classification_sample_fine_tuning.ipynb`
- `notebooks/04_mistral_classification_sample_fine_tuning.ipynb`

Script helper:

- `src/modeling_data.py`

Purpose:

- DistilBERT baseline and fine-tuning workflow.
- Llama 2 / Mistral first-stage classifier workflows.
- Load the final preprocessed dataset from `data/02_preprocessing_outputs/`.
- Sample a configurable number of records per class.
- Start with `SAMPLES_PER_CLASS = 1000` for quick testing.
- Increase the value later, for example to 20000 or 40000, for larger runs.
- Export sampled train/validation/test files under `data/03_modeling_inputs/`.

## Project Layout

```text
confidence-guided-selective-llm-reasoning/
  data/
    00_raw_reddit_archives/           # local archive dumps or raw parquet files; ignored by Git
    01_subreddit_preparation/         # subreddit-level prepared parquet files; ignored by Git
    02_preprocessing_outputs/         # final preprocessing CSV outputs; ignored by Git
    03_modeling_inputs/               # sampled train/validation/test CSVs; ignored by Git
  notebooks/
    00_reddit_subreddit_data_preparation.ipynb
    01_reddit_data_preprocessing.ipynb
    02_distilbert_classification_sample_fine_tuning.ipynb
    03_llama_classification_sample_fine_tuning.ipynb
    04_mistral_classification_sample_fine_tuning.ipynb
  src/
    prepare_subreddit_data.py
    preprocess_reddit.py
    modeling_data.py
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

Generated modeling inputs:

- `data/03_modeling_inputs/sample_1000_per_class/train_dataset.csv`
- `data/03_modeling_inputs/sample_1000_per_class/validation_dataset.csv`
- `data/03_modeling_inputs/sample_1000_per_class/test_dataset.csv`

## Important Git Note

The local parquet, zst, and generated CSV files are large. They are intentionally ignored by Git.

For GitHub collaboration, use Git LFS, cloud storage, or an agreed shared data location rather than committing large data directly.

## Current Data Status

The local working copy currently contains the available Reddit archive/raw files and the subreddit-level preprocessed parquet files needed to run the notebooks. These data files are not pushed to GitHub.

Collaborators should place shared data files in the paths documented in `data/README.md`.
