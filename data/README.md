# Data Directory

This project uses local Reddit archive dumps, raw parquet files, and subreddit-level preprocessed parquet files.

Large data files are ignored by Git. Share them through Git LFS, cloud storage, or another agreed data channel.

## Stage 0 Inputs

Place available Reddit archive dumps or raw parquet files under:

```text
data/00_raw_reddit_archives/
```

Currently supported raw/archive names:

- `AnxietyDepression_submissions.zst` or `anxiety_depression_submissions.parquet`
- `depression_submissions.zst` or `depression_submissions.parquet`
- `technology_submissions.zst` or `technology_submissions.parquet`
- `askscience_discussion_submissions.zst` or `askscience_discussion_submissions.parquet`
- `webdev_submissions.zst` or `webdev_discussion_submissions.parquet`
- `datascience_submissions.zst` or `datascience_submissions.parquet`
- `Positivity_submissions.zst` or `positivity_submissions.parquet`
- `MadeMeSmile_submissions.zst` or `mademesmile_submissions.parquet`
- `UnexpectedlyWholesome_submissions.zst` or `unexpectedlyWholesome_submissions.parquet`
- `CongratsLikeImFive_submissions.zst` or `congrats_submissions.parquet`
- `happy_submissions.zst` or `happy_submissions.parquet`

If a raw/archive source is unavailable but the corresponding preprocessed parquet already exists, Stage 0 can continue from the existing preprocessed file.

## Stage 0 Outputs / Stage 1 Inputs

Expected subreddit-level files under `data/01_subreddit_preparation/`:

- `anxiety_depression_submissions_preprocessed.parquet`
- `depression_submissions_preprocessed.parquet`
- `technology_submissions_preprocessed.parquet`
- `askscience_discussion_submissions_preprocessed.parquet`
- `webdev_discussion_submissions_preprocessed.parquet`
- `datascience_submissions_preprocessed.parquet`
- `positivity_submissions_preprocessed.parquet`
- `mademesmile_submissions_preprocessed.parquet`
- `unexpectedlyWholesome_submissions_preprocessed.parquet`
- `congrats_submissions_preprocessed.parquet`
- `happy_submissions_preprocessed.parquet`

## Stage 1 Outputs

Generated outputs under `data/02_preprocessing_outputs/`:

- `final_preprocessed_whole_df.csv`
- `final_preprocessed_df.csv`
- `class_counts_before_filtering.csv`
- `class_counts_after_filtering.csv`
- `final_class_distribution.csv`
- `filtering_summary.csv`

## Stage 2 Inputs

Modeling notebooks read:

- `data/02_preprocessing_outputs/final_preprocessed_df.csv`

They create sampled modeling inputs under:

```text
data/03_modeling_inputs/
```

For the first quick run, the notebooks use `SAMPLES_PER_CLASS = 1000`, which creates a balanced 3000-record dataset before train/validation/test splitting.
