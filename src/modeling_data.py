from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "02_preprocessing_outputs" / "final_preprocessed_df.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "03_modeling_inputs"

LABEL_TO_ID: Mapping[str, int] = {
    "Depression_Group": 0,
    "Neutral_Group": 1,
    "Happy_Group": 2,
}

ID_TO_LABEL: Mapping[int, str] = {
    0: "Depression_Group",
    1: "Neutral_Group",
    2: "Happy_Group",
}


def load_modeling_dataframe(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    text_column: str = "title_with_selftext_cleaned",
) -> pd.DataFrame:
    """Load the final preprocessed dataset and keep only modeling columns."""
    input_path = Path(input_path)
    df = pd.read_csv(input_path, low_memory=False)

    if text_column not in df.columns:
        fallback = "title_with_selftext"
        if fallback not in df.columns:
            raise ValueError(
                f"Could not find text column '{text_column}' or fallback '{fallback}'. "
                f"Available columns: {list(df.columns)}"
            )
        text_column = fallback

    required_columns = [text_column, "class_group"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df_model = df[required_columns].rename(
        columns={text_column: "text", "class_group": "label_str"}
    )
    df_model = df_model.dropna(subset=["text", "label_str"]).copy()
    df_model["label"] = df_model["label_str"].map(LABEL_TO_ID)
    df_model = df_model.dropna(subset=["label"]).copy()
    df_model["label"] = df_model["label"].astype(int)
    return df_model[["text", "label", "label_str"]]


def sample_per_class(
    df_model: pd.DataFrame,
    samples_per_class: int = 1000,
    random_state: int = 42,
    replace: bool | None = None,
) -> pd.DataFrame:
    """Return a balanced sample with the requested number of rows per class."""
    sampled_frames = []

    for label_id, label_name in ID_TO_LABEL.items():
        class_df = df_model[df_model["label"] == label_id]
        if class_df.empty:
            raise ValueError(f"No rows found for label {label_id}: {label_name}")

        use_replace = len(class_df) < samples_per_class if replace is None else replace
        sampled_frames.append(
            class_df.sample(
                n=samples_per_class,
                replace=use_replace,
                random_state=random_state,
            )
        )

    sampled_df = pd.concat(sampled_frames, ignore_index=True)
    return sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def split_modeling_dataframe(
    df_sampled: pd.DataFrame,
    train_size: float = 0.75,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split sampled data into train, validation, and test sets."""
    test_size = 1.0 - train_size - validation_size
    if test_size <= 0:
        raise ValueError("train_size + validation_size must be less than 1.0")

    train_df, temp_df = train_test_split(
        df_sampled,
        train_size=train_size,
        random_state=random_state,
        stratify=df_sampled["label"],
    )
    relative_validation_size = validation_size / (validation_size + test_size)
    validation_df, test_df = train_test_split(
        temp_df,
        train_size=relative_validation_size,
        random_state=random_state,
        stratify=temp_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def prepare_modeling_splits(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_name: str = "sample_1000_per_class",
    text_column: str = "title_with_selftext_cleaned",
    samples_per_class: int = 1000,
    random_state: int = 42,
    export_csv: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, sample, split, and optionally save modeling datasets."""
    df_model = load_modeling_dataframe(input_path=input_path, text_column=text_column)
    df_sampled = sample_per_class(
        df_model=df_model,
        samples_per_class=samples_per_class,
        random_state=random_state,
    )
    train_df, validation_df, test_df = split_modeling_dataframe(
        df_sampled=df_sampled,
        random_state=random_state,
    )

    if export_csv:
        output_path = Path(output_dir) / run_name
        output_path.mkdir(parents=True, exist_ok=True)
        train_df[["text", "label"]].to_csv(output_path / "train_dataset.csv", index=False)
        validation_df[["text", "label"]].to_csv(
            output_path / "validation_dataset.csv", index=False
        )
        test_df[["text", "label"]].to_csv(output_path / "test_dataset.csv", index=False)
        df_sampled[["text", "label", "label_str"]].to_csv(
            output_path / "sampled_dataset.csv", index=False
        )

    return train_df, validation_df, test_df
