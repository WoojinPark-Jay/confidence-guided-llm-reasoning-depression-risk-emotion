# Data Directory

This project uses local Reddit subreddit-level parquet inputs for Stage 1 preprocessing.

Expected input files under `data/interim/subreddit_preprocessed/`:

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

Large data files should usually be shared via Git LFS or external storage rather than committed directly to GitHub.

