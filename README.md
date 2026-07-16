# Confidence-Guided Selective LLM Reasoning for Proxy Emotion Classification

This repository organizes the IEEE paper project for depression-risk-related proxy emotion classification in Reddit text.

## Current Stage

Stage 1 is Reddit data preprocessing:

- Load 11 subreddit-level preprocessed parquet files.
- Build the three proxy classes: Depression, Neutral, and Happy.
- Clean and normalize text.
- Compute TextBlob polarity scores.
- Apply sentiment-aware filtering.
- Apply class balancing.
- Export final modeling data and summary tables.

## Project Layout

```text
confidence-guided-selective-llm-reasoning/
  data/
    interim/subreddit_preprocessed/   # local parquet inputs; do not commit large data directly
    processed/                        # generated CSV outputs and summary tables
  notebooks/
    01_reddit_data_preprocessing.ipynb
  src/
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

The local parquet and generated CSV files are large. For GitHub collaboration, use Git LFS, cloud storage, or an agreed shared data location rather than committing large data directly.

