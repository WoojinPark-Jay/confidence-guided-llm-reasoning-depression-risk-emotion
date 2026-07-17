# Confidence-Guided LLM Reasoning for Depression-Risk Emotion Classification

This repository organizes the code and data workflow for the IEEE paper project:

**Confidence-Guided Selective LLM Reasoning for Proxy Emotion Classification in Depression-Risk-Related Social Media Text**

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
- Save subreddit-level preprocessed parquet files under `data/interim/subreddit_preprocessed/`.

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

Planned next:

- DistilBERT baseline and fine-tuning workflow.
- Llama 2 / Mistral first-stage classifier workflows.
- Confidence-threshold routing analysis.
- Phase 2 reasoning experiments.

## Project Layout

```text
confidence-guided-selective-llm-reasoning/
  data/
    raw/reddit_archives/              # local archive dumps or raw parquet files; ignored by Git
    interim/subreddit_raw/            # generated raw parquet files; ignored by Git
    interim/subreddit_preprocessed/   # generated/required subreddit parquet inputs; ignored by Git
    processed/                        # generated final CSV outputs; ignored by Git
  notebooks/
    00_reddit_subreddit_data_preparation.ipynb
    01_reddit_data_preprocessing.ipynb
  src/
    prepare_subreddit_data.py
    preprocess_reddit.py
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

- `data/interim/subreddit_raw/*.parquet`
- `data/interim/subreddit_preprocessed/*_preprocessed.parquet`
- `data/interim/subreddit_preprocessed/subreddit_preparation_summary.csv`

## Run Stage 1

From the project root:

```bash
python src/preprocess_reddit.py
```

Expected outputs:

- `data/processed/final_preprocessed_whole_df.csv`
- `data/processed/final_preprocessed_df.csv`
- `data/processed/class_counts_before_filtering.csv`
- `data/processed/class_counts_after_filtering.csv`
- `data/processed/final_class_distribution.csv`
- `data/processed/filtering_summary.csv`

## Important Git Note

The local parquet, zst, and generated CSV files are large. They are intentionally ignored by Git.

For GitHub collaboration, use Git LFS, cloud storage, or an agreed shared data location rather than committing large data directly.

## Current Data Status

The local working copy currently contains the available Reddit archive/raw files and the subreddit-level preprocessed parquet files needed to run the notebooks. These data files are not pushed to GitHub.

Collaborators should place shared data files in the paths documented in `data/README.md`.
