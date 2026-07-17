from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "00_raw_reddit_archives"
RAW_PARQUET_DIR = RAW_DIR
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "01_subreddit_preparation"

KEEP_COLUMNS = [
    "subreddit",
    "author",
    "over_18",
    "link_flair_text",
    "title",
    "selftext",
    "url",
    "created_utc",
]


@dataclass(frozen=True)
class SubredditSource:
    name: str
    variable_name: str
    zst_file: str | None
    raw_parquet_file: str
    preprocessed_file: str
    nrows: int | None = None


SOURCES = [
    SubredditSource(
        name="AnxietyDepression",
        variable_name="anxiety_depression_submissions",
        zst_file="AnxietyDepression_submissions.zst",
        raw_parquet_file="anxiety_depression_submissions.parquet",
        preprocessed_file="anxiety_depression_submissions_preprocessed.parquet",
        nrows=2_000_000,
    ),
    SubredditSource(
        name="depression",
        variable_name="depression_submissions",
        zst_file="depression_submissions.zst",
        raw_parquet_file="depression_submissions.parquet",
        preprocessed_file="depression_submissions_preprocessed.parquet",
        nrows=300_000,
    ),
    SubredditSource(
        name="technology",
        variable_name="technology_submissions",
        zst_file="technology_submissions.zst",
        raw_parquet_file="technology_submissions.parquet",
        preprocessed_file="technology_submissions_preprocessed.parquet",
        nrows=600_000,
    ),
    SubredditSource(
        name="AskScienceDiscussion",
        variable_name="askscience_discussion_submissions",
        zst_file="askscience_discussion_submissions.zst",
        raw_parquet_file="askscience_discussion_submissions.parquet",
        preprocessed_file="askscience_discussion_submissions_preprocessed.parquet",
        nrows=1_000_000,
    ),
    SubredditSource(
        name="webdev",
        variable_name="webdev_discussion_submissions",
        zst_file="webdev_submissions.zst",
        raw_parquet_file="webdev_discussion_submissions.parquet",
        preprocessed_file="webdev_discussion_submissions_preprocessed.parquet",
        nrows=1_000_000,
    ),
    SubredditSource(
        name="datascience",
        variable_name="datascience_submissions",
        zst_file="datascience_submissions.zst",
        raw_parquet_file="datascience_submissions.parquet",
        preprocessed_file="datascience_submissions_preprocessed.parquet",
        nrows=4_000_000,
    ),
    SubredditSource(
        name="Positivity",
        variable_name="positivity_submissions",
        zst_file="Positivity_submissions.zst",
        raw_parquet_file="positivity_submissions.parquet",
        preprocessed_file="positivity_submissions_preprocessed.parquet",
        nrows=4_000_000,
    ),
    SubredditSource(
        name="MadeMeSmile",
        variable_name="mademesmile_submissions",
        zst_file="MadeMeSmile_submissions.zst",
        raw_parquet_file="mademesmile_submissions.parquet",
        preprocessed_file="mademesmile_submissions_preprocessed.parquet",
        nrows=100_000,
    ),
    SubredditSource(
        name="UnexpectedlyWholesome",
        variable_name="unexpectedlyWholesome_submissions",
        zst_file="UnexpectedlyWholesome_submissions.zst",
        raw_parquet_file="unexpectedlyWholesome_submissions.parquet",
        preprocessed_file="unexpectedlyWholesome_submissions_preprocessed.parquet",
        nrows=100_000,
    ),
    SubredditSource(
        name="CongratsLikeImFive",
        variable_name="congrats_submissions",
        zst_file="CongratsLikeImFive_submissions.zst",
        raw_parquet_file="congrats_submissions.parquet",
        preprocessed_file="congrats_submissions_preprocessed.parquet",
        nrows=30_000,
    ),
    SubredditSource(
        name="happy",
        variable_name="happy_submissions",
        zst_file="happy_submissions.zst",
        raw_parquet_file="happy_submissions.parquet",
        preprocessed_file="happy_submissions_preprocessed.parquet",
        nrows=2_000_000,
    ),
]


def convert_created_utc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "created_utc" in df.columns:
        df["created_utc"] = df["created_utc"].apply(
            lambda value: datetime.utcfromtimestamp(value)
            if isinstance(value, (int, float))
            else value
        )
    return df


def load_raw_source(source: SubredditSource, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    zst_path = raw_dir / source.zst_file if source.zst_file else None
    raw_parquet_path = raw_dir / source.raw_parquet_file

    if zst_path and zst_path.exists():
        df = pd.read_json(
            zst_path,
            compression={"method": "zstd", "max_window_size": 2_147_483_648},
            lines=True,
            nrows=source.nrows,
        )
        return convert_created_utc(df)

    if raw_parquet_path.exists():
        return pd.read_parquet(raw_parquet_path, engine="pyarrow")

    raise FileNotFoundError(
        f"No raw source found for {source.name}. Expected one of: "
        f"{zst_path if zst_path else 'n/a'} or {raw_parquet_path}"
    )


def make_preprocessed(df: pd.DataFrame) -> pd.DataFrame:
    available_columns = [column for column in KEEP_COLUMNS if column in df.columns]
    missing_columns = [column for column in KEEP_COLUMNS if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing expected columns: {missing_columns}")
    return df[available_columns].copy()


def prepare_one_source(source: SubredditSource) -> pd.DataFrame:
    RAW_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_source(source)
    raw_df.to_parquet(RAW_PARQUET_DIR / source.raw_parquet_file, engine="pyarrow")

    preprocessed_df = make_preprocessed(raw_df)
    preprocessed_df.to_parquet(PREPROCESSED_DIR / source.preprocessed_file, engine="pyarrow")
    return preprocessed_df


def main() -> None:
    summary = []
    for source in SOURCES:
        try:
            df = prepare_one_source(source)
            summary.append(
                {
                    "subreddit": source.name,
                    "preprocessed_file": source.preprocessed_file,
                    "records": len(df),
                    "status": "created",
                }
            )
            print(f"[created] {source.name}: {len(df):,} records")
        except FileNotFoundError as exc:
            existing = PREPROCESSED_DIR / source.preprocessed_file
            if existing.exists():
                df = pd.read_parquet(existing, engine="pyarrow")
                summary.append(
                    {
                        "subreddit": source.name,
                        "preprocessed_file": source.preprocessed_file,
                        "records": len(df),
                        "status": "used_existing_preprocessed",
                    }
                )
                print(f"[existing] {source.name}: {len(df):,} records ({exc})")
            else:
                raise

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(PREPROCESSED_DIR / "subreddit_preparation_summary.csv", index=False)
    print("\nSummary")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
