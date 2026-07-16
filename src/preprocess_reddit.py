from __future__ import annotations

import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
from textblob import TextBlob


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "interim" / "subreddit_preprocessed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


SUBREDDIT_FILES = {
    "anxiety_depression_submissions_preprocessed.parquet": "Depression_Group",
    "depression_submissions_preprocessed.parquet": "Depression_Group",
    "technology_submissions_preprocessed.parquet": "Neutral_Group",
    "askscience_discussion_submissions_preprocessed.parquet": "Neutral_Group",
    "webdev_discussion_submissions_preprocessed.parquet": "Neutral_Group",
    "datascience_submissions_preprocessed.parquet": "Neutral_Group",
    "positivity_submissions_preprocessed.parquet": "Happy_Group",
    "mademesmile_submissions_preprocessed.parquet": "Happy_Group",
    "unexpectedlyWholesome_submissions_preprocessed.parquet": "Happy_Group",
    "congrats_submissions_preprocessed.parquet": "Happy_Group",
    "happy_submissions_preprocessed.parquet": "Happy_Group",
}

DEPRESSION_SUBREDDITS = {"depression", "AnxietyDepression"}
NEUTRAL_SUBREDDITS = {"technology", "datascience", "AskScienceDiscussion", "webdev"}
HAPPY_SUBREDDITS = {
    "Positivity",
    "happy",
    "MadeMeSmile",
    "UnexpectedlyWholesome",
    "CongratsLikeImFive",
}


def ensure_nltk_resources() -> None:
    import nltk

    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package)


def alpha_filter(word: str) -> bool:
    return bool(re.compile(r"^[^a-z]+$").match(word))


def clean_words(text: object) -> str:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    if pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("\n", "")
    text = text.replace("[removed]", "")
    text = text.replace("[deleted]", "")
    text = text.replace("[View Poll]", "")
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    tokens = text.split(" ")

    nltk_stopwords = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in nltk_stopwords]
    tokens = [token for token in tokens if not alpha_filter(token)]
    tokens = ["".join(char for char in token if char not in string.punctuation) for token in tokens]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    custom_stopwords = {
        "",
        " ",
        "title",
        "amp",
        "x200b",
        "reddit",
        "com",
        "www",
        "http",
        "https",
    }
    tokens = [token for token in tokens if token not in custom_stopwords]
    return " ".join(tokens)


def define_polarity(text: object) -> float:
    return TextBlob("" if pd.isna(text) else str(text)).sentiment.polarity


def categorize_group(subreddit: object) -> str | None:
    if pd.isna(subreddit):
        return None
    subreddit = str(subreddit)
    if subreddit in DEPRESSION_SUBREDDITS:
        return "Depression_Group"
    if subreddit in NEUTRAL_SUBREDDITS:
        return "Neutral_Group"
    if subreddit in HAPPY_SUBREDDITS:
        return "Happy_Group"
    return None


def load_subreddit_frames(input_dir: Path = INPUT_DIR) -> pd.DataFrame:
    frames = []
    missing = []
    for filename in SUBREDDIT_FILES:
        path = input_dir / filename
        if not path.exists():
            missing.append(path)
            continue
        frames.append(pd.read_parquet(path, engine="pyarrow"))
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required parquet input files:\n{missing_text}")
    return pd.concat(frames, ignore_index=True)


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title"] = df["title"].fillna("")
    df["selftext"] = df["selftext"].fillna("")
    df["title_with_selftext"] = df["title"] + " " + df["selftext"]
    df["title_with_selftext_cleaned"] = df["title_with_selftext"].apply(clean_words)
    df["polarity"] = df["title_with_selftext_cleaned"].apply(define_polarity)
    df["class_group"] = df["subreddit"].apply(categorize_group)
    return df


def class_counts(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Depression_Group", "Neutral_Group", "Happy_Group"]
    counts = df["class_group"].value_counts(dropna=False).reindex(order).fillna(0).astype(int)
    return counts.rename_axis("class_group").reset_index(name="count")


def sentiment_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    depression_df = df[df["class_group"] == "Depression_Group"]
    neutral_df = df[df["class_group"] == "Neutral_Group"]
    happy_df = df[df["class_group"] == "Happy_Group"]

    before_counts = class_counts(df)

    depression_df = depression_df[depression_df["polarity"] < 0.2]
    neutral_df = neutral_df[(neutral_df["polarity"] > -0.3) & (neutral_df["polarity"] < 0.3)]
    happy_df = happy_df[happy_df["polarity"] > -0.2]

    filtered_df = pd.concat([depression_df, neutral_df, happy_df], ignore_index=True)
    after_counts = class_counts(filtered_df)
    return filtered_df, before_counts, after_counts


def balance_classes(filtered_df: pd.DataFrame) -> pd.DataFrame:
    depression_df = filtered_df[filtered_df["class_group"] == "Depression_Group"]
    neutral_df = filtered_df[filtered_df["class_group"] == "Neutral_Group"]
    happy_df = filtered_df[filtered_df["class_group"] == "Happy_Group"]

    depression_df = depression_df.sample(frac=0.19, replace=True, random_state=1)
    neutral_df = neutral_df.sample(frac=0.24, replace=True, random_state=1)

    return pd.concat([depression_df, neutral_df, happy_df], ignore_index=True)


def build_filtering_summary(before_counts: pd.DataFrame, after_counts: pd.DataFrame) -> pd.DataFrame:
    before = before_counts.rename(columns={"count": "before_polarity_filtering"})
    after = after_counts.rename(columns={"count": "after_polarity_filtering"})
    summary = before.merge(after, on="class_group", how="outer").fillna(0)
    summary["excluded_by_filtering"] = (
        summary["before_polarity_filtering"] - summary["after_polarity_filtering"]
    )
    summary["retention_rate"] = (
        summary["after_polarity_filtering"] / summary["before_polarity_filtering"] * 100
    ).round(2)

    total = pd.DataFrame(
        {
            "class_group": ["Total"],
            "before_polarity_filtering": [summary["before_polarity_filtering"].sum()],
            "after_polarity_filtering": [summary["after_polarity_filtering"].sum()],
            "excluded_by_filtering": [summary["excluded_by_filtering"].sum()],
            "retention_rate": [
                round(
                    summary["after_polarity_filtering"].sum()
                    / summary["before_polarity_filtering"].sum()
                    * 100,
                    2,
                )
            ],
        }
    )
    return pd.concat([summary, total], ignore_index=True)


def save_outputs(
    filtered_df: pd.DataFrame,
    final_df: pd.DataFrame,
    before_counts: pd.DataFrame,
    after_counts: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_dir / "final_preprocessed_whole_df.csv", index=False)
    final_df.to_csv(output_dir / "final_preprocessed_df.csv", index=False)

    before_counts.to_csv(output_dir / "class_counts_before_filtering.csv", index=False)
    after_counts.to_csv(output_dir / "class_counts_after_filtering.csv", index=False)

    final_counts = class_counts(final_df)
    final_counts.to_csv(output_dir / "final_class_distribution.csv", index=False)

    filtering_summary = build_filtering_summary(before_counts, after_counts)
    filtering_summary.to_csv(output_dir / "filtering_summary.csv", index=False)


def main() -> None:
    ensure_nltk_resources()
    raw_df = load_subreddit_frames()
    feature_df = add_text_features(raw_df)
    filtered_df, before_counts, after_counts = sentiment_filter(feature_df)
    final_df = balance_classes(filtered_df)
    save_outputs(filtered_df, final_df, before_counts, after_counts)

    print("Class counts before polarity filtering:")
    print(before_counts.to_string(index=False))
    print("\nClass counts after polarity filtering:")
    print(after_counts.to_string(index=False))
    print("\nFinal class distribution:")
    print(class_counts(final_df).to_string(index=False))
    print(f"\nSaved outputs under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

