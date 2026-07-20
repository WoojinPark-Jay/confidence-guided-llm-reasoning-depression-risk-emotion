# This file is generated from the matching Colab notebook for code review/reuse.
# The notebook remains the primary runnable artifact.

# %% [markdown]
# # DistilBERT Colab Training Notebook
# 
# This Colab-oriented notebook keeps the collaborator workflow structure while making the run safer for GitHub collaboration.
# 
# Run setup:
# 
# 1. Open this notebook in Google Colab.
# 2. Use a GPU runtime for Llama 2 or Mistral. DistilBERT can run on a smaller GPU.
# 3. Keep W&B API keys out of the notebook. Either add `WANDB_API_KEY` in Colab Secrets or let the `wandb.login()` prompt ask for the key.
# 4. Start with `SAMPLES_PER_CLASS = 300` for a smoke test, then increase it for full training.
# 5. Use `WANDB_SWEEP_MODE = "new"` unless you intentionally have access to an existing collaborator sweep.
# 
# Do not commit W&B tokens, generated model folders, W&B run folders, or large local data files.

# %% [markdown]
# 
# # DistilBERT Reddit Classification + W&B Macro-F1 Tuning + Confidence Threshold Analysis
# 
# This notebook runs the following sequence:
# 
# 1. Load the Reddit dataset.
# 2. Sample 300 rows from each of Depression, Neutral, and Happy.
# 3. Create stratified train, validation, and held-out test splits.
# 4. Run or reuse a W&B hyperparameter sweep.
# 5. Select the best **finished** W&B run using validation macro F1.
# 6. Create `BEST_HYPERPARAMETERS`.
# 7. Train a fresh final DistilBERT model using `BEST_HYPERPARAMETERS`.
# 8. Evaluate the held-out test set.
# 9. Select the confidence threshold using the validation set only.
# 10. Apply the fixed threshold to the held-out test set.
# 
# `BEST_HYPERPARAMETERS` is therefore never referenced before the W&B best-run
# selection step has completed.
# 
# The unrelated `dair-ai/emotion` dataset is not used.

# %%

# Install dependencies in Colab/Jupyter if needed.
# %pip install -q -U transformers datasets accelerate scikit-learn pandas matplotlib scipy wandb

# %%

import os
import gc
import json
import random
import inspect
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from IPython.display import display
from datasets import Dataset, DatasetDict
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

try:
    from transformers.trainer_utils import SaveStrategy
except ImportError:
    SaveStrategy = None

warnings.filterwarnings("ignore")

SEED = 42


def set_seed_everywhere(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed_everywhere(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
print("PyTorch:", torch.__version__)

# %% [markdown]
# 
# ## 1. Configuration
# 
# `TEXT_COLUMN` and `LABEL_COLUMN` may be set manually. When left as `None`, the notebook detects them from common candidate names.
# 
# The sampler supports two modes:
# 
# - `"reservoir"`: reads the entire CSV in chunks and obtains an approximately unbiased sample from each class.
# - `"first_balanced"`: stops once 300 examples per class are collected. Faster, but potentially biased if the CSV is ordered.
# 
# For a paper-quality experiment, use `"reservoir"`. For a quick code smoke test, use `"first_balanced"`.

# %%

DATA_URL = (
    "https://media.githubusercontent.com/media/"
    "Branden-Kang/LLaMA-2/main/data/final_preprocessed_df2.csv"
)

MODEL_NAME = "distilbert-base-uncased"

TEXT_COLUMN = "title_with_selftext_cleaned"
LABEL_COLUMN = "class_group"

TEXT_COLUMN_CANDIDATES = [
    "title_with_selftext_cleaned",
    "Title_with_selftext_cleaned",
    "text",
    "Text",
    "cleaned_text",
    "content",
    "selftext",
    "title_with_selftext",
    "Title_with_selftext",
]

LABEL_COLUMN_CANDIDATES = [
    "class_group",
    "label",
    "Label",
    "labels",
    "class",
    "Class",
    "emotion",
    "Emotion",
    "group",
    "Group",
    "category",
    "Category",
]

# -------------------------------------------------------------------
# Dataset sampling
# -------------------------------------------------------------------
# Default Colab smoke-test size: 300 rows per class, 900 total rows.
# For larger paper-scale runs, change this value to 1000, 20000, 40000, etc.
SAMPLES_PER_CLASS = 300

# "first_balanced": fast smoke test; stops after enough rows are collected.
# "reservoir": reads the full CSV and reduces source-order sampling bias.
SAMPLING_MODE = "first_balanced"
# Use "reservoir" for paper-quality sampling when running the full dataset.
CSV_CHUNK_SIZE = 5_000

TRAIN_RATIO = 0.75
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.10

# With SAMPLES_PER_CLASS=300 this gives approximately:
# train=675, validation=135, test=90.

MAX_LENGTH = 256
EARLY_STOPPING_PATIENCE = 2

# -------------------------------------------------------------------
# W&B configuration
# -------------------------------------------------------------------
# Recommended:
#   "new" creates a new sweep optimized for validation macro F1.
#
# Other modes:
#   "continue_existing": add trials to an existing compatible sweep.
#   "reuse_best": add no trials and reuse the best finished existing run.
#   "disabled": skip W&B and use DEFAULT_HYPERPARAMETERS.
WANDB_SWEEP_MODE = "new"

WANDB_ENTITY = None
WANDB_PROJECT = "confidence-guided-distilbert-colab"
EXISTING_SWEEP_ID = ""

# Leave WANDB_ENTITY as None to use the account/team selected at login.
# To continue a collaborator sweep instead, set for example:
# WANDB_ENTITY = "kangsy413"
# WANDB_PROJECT = "my-bert-sweep"
# EXISTING_SWEEP_ID = "8v1kemw0"
# WANDB_SWEEP_MODE = "continue_existing"

WANDB_SWEEP_NAME = "distilbert-reddit-validation-macro-f1"
WANDB_SWEEP_COUNT = 4
WANDB_MODE = "online"

WANDB_OBJECTIVE_METRIC = "validation_f1_macro"
WANDB_OBJECTIVE_GOAL = "maximize"

# Existing sweeps are controlled by their server-side objective.
# Keep this True to prevent continuing a loss-based sweep as if it were
# a Macro-F1 sweep.
REQUIRE_EXISTING_SWEEP_METRIC_MATCH = True

WANDB_API_RETRIES = 12
WANDB_API_RETRY_SECONDS = 5
LOG_FINAL_TRAINING_TO_WANDB = True

# False is more appropriate for the 900-row smoke test.
# True reproduces the older search space more closely.
USE_REFERENCE_WANDB_SEARCH_SPACE = False

if USE_REFERENCE_WANDB_SEARCH_SPACE:
    WANDB_BATCH_SIZE_VALUES = [32, 64, 128]
    WANDB_EPOCH_VALUES = [5, 10, 15]
else:
    WANDB_BATCH_SIZE_VALUES = [16, 32, 64]
    WANDB_EPOCH_VALUES = [3, 5, 10]

WANDB_WEIGHT_DECAY_VALUES = [1e-2, 1e-3, 1e-4]
WANDB_LEARNING_RATE_MIN = 1e-5
WANDB_LEARNING_RATE_MAX = 2e-4

DEFAULT_HYPERPARAMETERS = {
    "learning_rate": 3.8e-5,
    "batch_size": 16,
    "epochs": 3,
    "weight_decay": 0.01,
}

VALID_WANDB_SWEEP_MODES = {
    "new",
    "continue_existing",
    "reuse_best",
    "disabled",
}

if WANDB_SWEEP_MODE not in VALID_WANDB_SWEEP_MODES:
    raise ValueError(
        "WANDB_SWEEP_MODE must be one of "
        f"{sorted(VALID_WANDB_SWEEP_MODES)}."
    )

USE_EXISTING_SWEEP = WANDB_SWEEP_MODE in {
    "continue_existing",
    "reuse_best",
}

# -------------------------------------------------------------------
# Confidence-threshold analysis
# -------------------------------------------------------------------
THRESHOLD_GRID = np.round(np.arange(0.50, 1.00, 0.01), 2)
TARGET_SELECTIVE_RISK = 0.05
MIN_ACCEPTED_SAMPLES = 10
REPORT_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]

OUTPUT_DIR = Path(
    "./distilbert_reddit_wandb_f1_threshold_outputs"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SWEEP_RESULTS_PATH = (
    OUTPUT_DIR / "wandb_sweep_results.jsonl"
)

print("Output directory:", OUTPUT_DIR.resolve())
print("W&B sweep mode:", WANDB_SWEEP_MODE)
print("W&B entity/project:", f"{WANDB_ENTITY or 'default-account'}/{WANDB_PROJECT}")
print(
    "Selection objective:",
    WANDB_OBJECTIVE_METRIC,
    WANDB_OBJECTIVE_GOAL,
)

if USE_EXISTING_SWEEP:
    print("Existing sweep ID:", EXISTING_SWEEP_ID)

# %% [markdown]
# 
# ## W&B mode selection
# 
# ### Create a new Macro-F1 sweep
# 
# ```python
# WANDB_SWEEP_MODE = "new"
# ```
# 
# This is the recommended mode because the W&B sweep, Trainer checkpoint
# selection, early stopping, and final best-run selection all use validation
# macro F1.
# 
# ### Continue an existing compatible sweep
# 
# ```python
# WANDB_SWEEP_MODE = "continue_existing"
# EXISTING_SWEEP_ID = "entity/project/sweep_id"
# ```
# 
# The code checks the existing server-side metric. If it is not
# `validation_f1_macro` with goal `maximize`, execution stops instead of mixing
# two different selection criteria.
# 
# ### Reuse an existing sweep without adding trials
# 
# ```python
# WANDB_SWEEP_MODE = "reuse_best"
# EXISTING_SWEEP_ID = "kangsy413/my-bert-sweep/8v1kemw0"
# ```
# 
# Only runs whose W&B state is `finished` and that contain a valid
# `validation_f1_macro` value are eligible.
# 
# ### Disable W&B
# 
# ```python
# WANDB_SWEEP_MODE = "disabled"
# ```
# 
# This uses `DEFAULT_HYPERPARAMETERS`.

# %% [markdown]
# ## 2. Detect columns and sample 300 rows per class

# %%

CANONICAL_CLASS_TO_ID = {
    "Depression": 0,
    "Neutral": 1,
    "Happy": 2,
}
ID_TO_CLASS = {v: k for k, v in CANONICAL_CLASS_TO_ID.items()}


def detect_column(columns: Iterable[str], preferred: Optional[str], candidates: List[str]) -> str:
    columns = list(columns)

    if preferred is not None:
        if preferred not in columns:
            raise KeyError(
                f"Configured column '{preferred}' was not found. "
                f"Available columns: {columns}"
            )
        return preferred

    for candidate in candidates:
        if candidate in columns:
            return candidate

    lower_to_original = {str(col).lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    raise KeyError(
        "Could not automatically detect a required column. "
        f"Available columns: {columns}"
    )


def normalize_label(value) -> Optional[int]:
    '''
    Normalize common Reddit class representations to:
      Depression = 0
      Neutral = 1
      Happy = 2
    '''
    if pd.isna(value):
        return None

    # Numeric labels already encoded as 0, 1, 2.
    if isinstance(value, (int, np.integer)):
        return int(value) if int(value) in ID_TO_CLASS else None

    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        int_value = int(value)
        return int_value if int_value in ID_TO_CLASS else None

    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")

    if normalized in {"0", "depression", "depressed", "depression_group"}:
        return 0
    if normalized in {"1", "neutral", "neutral_group"}:
        return 1
    if normalized in {"2", "happy", "happiness", "happy_group", "positive"}:
        return 2

    if "depress" in normalized:
        return 0
    if "neutral" in normalized:
        return 1
    if "happy" in normalized:
        return 2

    return None


def inspect_csv_schema(
    csv_url: str,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
) -> Tuple[str, str, pd.DataFrame]:
    preview = pd.read_csv(csv_url, nrows=20, low_memory=False)
    detected_text = detect_column(preview.columns, text_column, TEXT_COLUMN_CANDIDATES)
    detected_label = detect_column(preview.columns, label_column, LABEL_COLUMN_CANDIDATES)

    print("Detected text column :", detected_text)
    print("Detected label column:", detected_label)
    print("Available columns     :", list(preview.columns))
    print("\nRaw label examples:")
    print(preview[detected_label].value_counts(dropna=False).head(10))

    return detected_text, detected_label, preview


detected_text_col, detected_label_col, preview_df = inspect_csv_schema(
    DATA_URL,
    text_column=TEXT_COLUMN,
    label_column=LABEL_COLUMN,
)

# %%

def sample_balanced_from_csv(
    csv_url: str,
    text_col: str,
    label_col: str,
    samples_per_class: int = 300,
    chunksize: int = 5_000,
    mode: str = "first_balanced",
    seed: int = 42,
) -> pd.DataFrame:
    '''
    Sample an equal number of Depression, Neutral, and Happy examples.

    mode="first_balanced":
        Fast smoke-test mode. Stops after collecting enough rows for all classes.

    mode="reservoir":
        Reads the complete CSV and performs per-class reservoir sampling.
        This is less sensitive to source-file ordering.
    '''
    if mode not in {"first_balanced", "reservoir"}:
        raise ValueError("mode must be 'first_balanced' or 'reservoir'")

    rng = random.Random(seed)
    target_ids = [0, 1, 2]
    reservoirs: Dict[int, List[dict]] = {class_id: [] for class_id in target_ids}
    seen_counts: Dict[int, int] = {class_id: 0 for class_id in target_ids}

    usecols = [text_col, label_col]

    for chunk_index, chunk in enumerate(
        pd.read_csv(
            csv_url,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        ),
        start=1,
    ):
        chunk = chunk.dropna(subset=[text_col, label_col]).copy()
        chunk["label"] = chunk[label_col].map(normalize_label)
        chunk = chunk[chunk["label"].isin(target_ids)].copy()
        chunk["text"] = chunk[text_col].astype(str).str.strip()
        chunk = chunk[chunk["text"].str.len() > 0]

        if mode == "first_balanced":
            # Shuffle within each chunk before filling remaining slots.
            chunk = chunk.sample(frac=1.0, random_state=seed + chunk_index)

            for class_id in target_ids:
                remaining = samples_per_class - len(reservoirs[class_id])
                if remaining <= 0:
                    continue

                candidates = chunk[chunk["label"] == class_id][["text", "label"]]
                if not candidates.empty:
                    reservoirs[class_id].extend(
                        candidates.head(remaining).to_dict("records")
                    )

            if all(len(reservoirs[c]) >= samples_per_class for c in target_ids):
                print(f"Early stop after chunk {chunk_index}.")
                break

        else:
            # Per-class reservoir sampling over the full CSV stream.
            for row in chunk[["text", "label"]].to_dict("records"):
                class_id = int(row["label"])
                seen_counts[class_id] += 1

                if len(reservoirs[class_id]) < samples_per_class:
                    reservoirs[class_id].append(row)
                else:
                    replacement_index = rng.randint(0, seen_counts[class_id] - 1)
                    if replacement_index < samples_per_class:
                        reservoirs[class_id][replacement_index] = row

        if chunk_index % 10 == 0:
            sizes = {ID_TO_CLASS[k]: len(v) for k, v in reservoirs.items()}
            print(f"Processed {chunk_index} chunks; current sample sizes: {sizes}")

    sample_rows = []
    for class_id in target_ids:
        class_rows = reservoirs[class_id]
        if len(class_rows) < samples_per_class:
            raise ValueError(
                f"Only {len(class_rows)} rows were collected for "
                f"{ID_TO_CLASS[class_id]}; required {samples_per_class}."
            )
        sample_rows.extend(class_rows[:samples_per_class])

    sampled_df = pd.DataFrame(sample_rows)
    sampled_df["label"] = sampled_df["label"].astype(int)
    sampled_df["label_name"] = sampled_df["label"].map(ID_TO_CLASS)

    # Final shuffle after balancing.
    sampled_df = sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sampled_df.insert(0, "sample_id", np.arange(len(sampled_df)))

    return sampled_df


sampled_df = sample_balanced_from_csv(
    csv_url=DATA_URL,
    text_col=detected_text_col,
    label_col=detected_label_col,
    samples_per_class=SAMPLES_PER_CLASS,
    chunksize=CSV_CHUNK_SIZE,
    mode=SAMPLING_MODE,
    seed=SEED,
)

print("\nBalanced sample shape:", sampled_df.shape)
print(sampled_df["label_name"].value_counts())
display(sampled_df.head())

sampled_df.to_csv(OUTPUT_DIR / "balanced_reddit_sample_900.csv", index=False)

# %% [markdown]
# 
# ## 3. Stratified train/validation/test split
# 
# For 900 examples, the expected split sizes are approximately:
# 
# - Train: 675
# - Validation: 135
# - Test: 90
# 
# The validation set is used to select the routing threshold. The held-out test set is not used during threshold selection.

# %%

def stratified_three_way_split(
    df: pd.DataFrame,
    train_ratio: float = 0.75,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=df["label"],
    )

    relative_test_ratio = test_ratio / (validation_ratio + test_ratio)

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


train_df, validation_df, test_df = stratified_three_way_split(
    sampled_df,
    train_ratio=TRAIN_RATIO,
    validation_ratio=VALIDATION_RATIO,
    test_ratio=TEST_RATIO,
    seed=SEED,
)

for name, split_df in {
    "train": train_df,
    "validation": validation_df,
    "test": test_df,
}.items():
    print(f"\n{name}: {len(split_df)}")
    print(split_df["label_name"].value_counts().sort_index())

train_df.to_csv(OUTPUT_DIR / "train_sample.csv", index=False)
validation_df.to_csv(OUTPUT_DIR / "validation_sample.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test_sample.csv", index=False)

# %% [markdown]
# ## 4. Build Hugging Face datasets and tokenize

# %%

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dataset_dict = DatasetDict({
    "train": Dataset.from_pandas(
        train_df[["sample_id", "text", "label"]],
        preserve_index=False,
    ),
    "validation": Dataset.from_pandas(
        validation_df[["sample_id", "text", "label"]],
        preserve_index=False,
    ),
    "test": Dataset.from_pandas(
        test_df[["sample_id", "text", "label"]],
        preserve_index=False,
    ),
})


def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


tokenized_datasets = dataset_dict.map(
    tokenize_batch,
    batched=True,
    desc="Tokenizing",
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print(tokenized_datasets)

# %% [markdown]
# 
# ## 5. Define DistilBERT training helpers
# 
# The W&B sweep evaluates one fresh DistilBERT model for each sampled
# hyperparameter configuration. The validation set is used for sweep evaluation,
# and the test set remains untouched until the final model has been selected.

# %%

id2label = {
    0: "Depression",
    1: "Neutral",
    2: "Happy",
}
label2id = {
    label: class_id
    for class_id, label in id2label.items()
}


def create_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def resolve_save_strategy() -> str:
    '''
    Use "best" when supported by the installed Transformers version.
    Otherwise use epoch-level saving.
    '''
    if SaveStrategy is not None:
        values = {
            member.value
            for member in SaveStrategy
        }

        if "best" in values:
            return "best"

    return "epoch"


class ManualWandbMetricsCallback(TrainerCallback):
    '''
    Log metrics manually without enabling Trainer's built-in W&B integration.

    This avoids attempts to overwrite sweep-locked config values such as
    weight_decay.
    '''

    def __init__(self, wandb_run=None):
        self.wandb_run = wandb_run

    @staticmethod
    def scalar_metrics(
        values: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        if not values:
            return {}

        result = {}

        for key, value in values.items():
            if isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):
                result[key] = float(value)

        return result

    def _run_is_active(self) -> bool:
        if self.wandb_run is None:
            return False

        # W&B marks a run as finished after run.finish(). A finished run must
        # not receive additional log calls.
        return not bool(
            getattr(
                self.wandb_run,
                "_is_finished",
                False,
            )
        )

    def on_log(
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs,
    ):
        if not self._run_is_active():
            return

        scalar_logs = self.scalar_metrics(logs)

        if scalar_logs:
            self.wandb_run.log(
                {
                    f"trainer/{key}": value
                    for key, value in scalar_logs.items()
                },
                step=state.global_step,
            )

    def on_evaluate(
        self,
        args,
        state,
        control,
        metrics=None,
        **kwargs,
    ):
        if not self._run_is_active():
            return

        metrics = self.scalar_metrics(metrics)

        mapping = {
            "eval_loss": "validation_loss",
            "eval_accuracy": "validation_accuracy",
            "eval_precision_macro": "validation_precision_macro",
            "eval_recall_macro": "validation_recall_macro",
            "eval_f1_macro": "validation_f1_macro",
            "eval_runtime": "validation_runtime",
        }

        payload = {
            mapping.get(key, key): value
            for key, value in metrics.items()
        }

        if payload:
            self.wandb_run.log(
                payload,
                step=state.global_step,
            )


def create_training_arguments(
    output_dir: Path,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    weight_decay: float,
    run_name: Optional[str] = None,
) -> TrainingArguments:
    save_strategy = resolve_save_strategy()

    kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": float(learning_rate),
        "per_device_train_batch_size": int(batch_size),
        "per_device_eval_batch_size": int(batch_size),
        "num_train_epochs": int(epochs),
        "weight_decay": float(weight_decay),
        "logging_strategy": "epoch",
        "save_strategy": save_strategy,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "save_total_limit": 1,
        "report_to": [],
        "run_name": run_name,
        "seed": SEED,
        "data_seed": SEED,
        "fp16": torch.cuda.is_available(),
        "push_to_hub": False,
    }

    signature = inspect.signature(
        TrainingArguments.__init__
    )

    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    training_arguments = TrainingArguments(**kwargs)

    print(
        "Training arguments:",
        {
            "learning_rate": training_arguments.learning_rate,
            "batch_size": (
                training_arguments
                .per_device_train_batch_size
            ),
            "epochs": training_arguments.num_train_epochs,
            "weight_decay": training_arguments.weight_decay,
            "save_strategy": str(
                training_arguments.save_strategy
            ),
            "best_metric": (
                training_arguments
                .metric_for_best_model
            ),
        },
    )

    return training_arguments


def create_trainer(
    model,
    training_args: TrainingArguments,
    wandb_run=None,
) -> Trainer:
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=(
                EARLY_STOPPING_PATIENCE
            )
        )
    ]

    if wandb_run is not None:
        callbacks.append(
            ManualWandbMetricsCallback(
                wandb_run
            )
        )

    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": (
            tokenized_datasets["train"]
        ),
        "eval_dataset": (
            tokenized_datasets["validation"]
        ),
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "callbacks": callbacks,
    }

    signature = inspect.signature(
        Trainer.__init__
    )

    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer

    return Trainer(**kwargs)


def release_training_objects(*objects) -> None:
    for obj in objects:
        del obj

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %% [markdown]
# 
# ## 6. W&B Macro-F1 hyperparameter sweep
# 
# Each sweep trial initializes a fresh DistilBERT model. Multiple training runs
# are therefore expected. A trial may stop before its configured maximum epoch
# when validation macro F1 does not improve for the configured early-stopping
# patience.

# %%

def build_sweep_path(
    entity: str,
    project: str,
    sweep_id: str,
) -> str:
    clean_id = str(sweep_id).strip().strip("/")

    if not clean_id:
        raise ValueError(
            "EXISTING_SWEEP_ID is empty."
        )

    parts = clean_id.split("/")

    if len(parts) == 1:
        return (
            f"{entity}/{project}/{parts[0]}"
        )

    if len(parts) == 3:
        return clean_id

    raise ValueError(
        "Sweep ID must be a short ID or "
        "entity/project/sweep_id."
    )


def get_sweep_metric_spec(
    sweep,
) -> Tuple[Optional[str], Optional[str]]:
    sweep_config = dict(
        sweep.config or {}
    )
    metric_config = (
        sweep_config.get("metric") or {}
    )

    if not isinstance(metric_config, dict):
        return None, None

    metric_name = metric_config.get("name")
    metric_goal = metric_config.get("goal")

    if metric_goal is not None:
        metric_goal = str(metric_goal).lower()

    return metric_name, metric_goal


def load_existing_sweep(
    entity: str,
    project: str,
    sweep_id: str,
):
    import wandb

    sweep_path = build_sweep_path(
        entity,
        project,
        sweep_id,
    )

    api = wandb.Api()
    sweep = api.sweep(sweep_path)

    metric_name, metric_goal = (
        get_sweep_metric_spec(sweep)
    )

    print("Existing W&B sweep")
    print("  path  :", sweep_path)
    print("  state :", sweep.state)
    print("  runs  :", len(sweep.runs))
    print("  metric:", metric_name)
    print("  goal  :", metric_goal)
    print("  URL   :", sweep.url)

    return sweep_path, sweep


def validate_existing_sweep_objective(
    sweep,
) -> None:
    metric_name, metric_goal = (
        get_sweep_metric_spec(sweep)
    )

    matches = (
        metric_name == WANDB_OBJECTIVE_METRIC
        and metric_goal == WANDB_OBJECTIVE_GOAL
    )

    if matches:
        print(
            "Existing sweep objective is compatible."
        )
        return

    message = (
        "Existing sweep objective mismatch. "
        f"Existing metric={metric_name!r}, "
        f"goal={metric_goal!r}; "
        f"required metric="
        f"{WANDB_OBJECTIVE_METRIC!r}, "
        f"goal={WANDB_OBJECTIVE_GOAL!r}. "
        "Set WANDB_SWEEP_MODE='new' to create "
        "a Macro-F1 sweep."
    )

    if REQUIRE_EXISTING_SWEEP_METRIC_MATCH:
        raise RuntimeError(message)

    print("WARNING:", message)


def login_wandb_for_colab() -> None:
    if WANDB_SWEEP_MODE == "disabled" or WANDB_MODE == "disabled":
        print("W&B disabled; skipping login.")
        return

    import wandb

    os.environ["WANDB_MODE"] = WANDB_MODE

    api_key = os.environ.get("WANDB_API_KEY")

    try:
        from google.colab import userdata

        api_key = api_key or userdata.get("WANDB_API_KEY")
    except Exception:
        pass

    if api_key:
        wandb.login(key=api_key)
    else:
        print(
            "No WANDB_API_KEY found in environment or Colab Secrets. "
            "A secure W&B prompt will appear. Do not hard-code the key."
        )
        wandb.login()


if WANDB_SWEEP_MODE != "disabled":
    import wandb

    login_wandb_for_colab()


if WANDB_SWEEP_MODE == "new":
    sweep_configuration = {
        "method": "bayes",
        "name": WANDB_SWEEP_NAME,
        "metric": {
            "name": WANDB_OBJECTIVE_METRIC,
            "goal": WANDB_OBJECTIVE_GOAL,
        },
        "parameters": {
            "batch_size": {
                "values": (
                    WANDB_BATCH_SIZE_VALUES
                ),
            },
            "epochs": {
                "values": WANDB_EPOCH_VALUES,
            },
            "weight_decay": {
                "values": (
                    WANDB_WEIGHT_DECAY_VALUES
                ),
            },
            "learning_rate": {
                "distribution": "uniform",
                "min": (
                    WANDB_LEARNING_RATE_MIN
                ),
                "max": (
                    WANDB_LEARNING_RATE_MAX
                ),
            },
        },
    }

    print("New sweep configuration:")
    print(
        json.dumps(
            sweep_configuration,
            indent=2,
        )
    )

elif WANDB_SWEEP_MODE in {
    "continue_existing",
    "reuse_best",
}:
    (
        RESOLVED_SWEEP_PATH,
        EXISTING_SWEEP_OBJECT,
    ) = load_existing_sweep(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        sweep_id=EXISTING_SWEEP_ID,
    )

    if (
        WANDB_SWEEP_MODE
        == "continue_existing"
    ):
        validate_existing_sweep_objective(
            EXISTING_SWEEP_OBJECT
        )

else:
    print(
        "W&B disabled; fixed defaults will be used."
    )

# %%

def append_jsonl(
    path: Path,
    record: Dict[str, Any],
) -> None:
    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                default=str,
            )
            + "\n"
        )


def get_wandb_config_value(
    config,
    canonical_name: str,
    aliases: Optional[List[str]] = None,
):
    aliases = aliases or []
    config_dict = dict(config)

    for name in [
        canonical_name,
        *aliases,
    ]:
        if (
            name in config_dict
            and config_dict[name] is not None
        ):
            return config_dict[name]

    raise KeyError(
        f"Missing W&B parameter "
        f"{canonical_name!r}. "
        f"Available keys: "
        f"{sorted(config_dict.keys())}"
    )


def run_wandb_trial() -> None:
    import wandb

    with wandb.init() as run:
        config = run.config

        hyperparameters = {
            "learning_rate": float(
                get_wandb_config_value(
                    config,
                    "learning_rate",
                    aliases=["lr"],
                )
            ),
            "batch_size": int(
                get_wandb_config_value(
                    config,
                    "batch_size",
                    aliases=[
                        "per_device_train_batch_size",
                        "train_batch_size",
                    ],
                )
            ),
            "epochs": int(
                get_wandb_config_value(
                    config,
                    "epochs",
                    aliases=[
                        "num_train_epochs",
                    ],
                )
            ),
            "weight_decay": float(
                get_wandb_config_value(
                    config,
                    "weight_decay",
                )
            ),
        }

        # Use new keys prefixed with "resolved_" so sweep-locked
        # parameters are not overwritten.
        run.config.update(
            {
                "resolved_learning_rate": (
                    hyperparameters[
                        "learning_rate"
                    ]
                ),
                "resolved_batch_size": (
                    hyperparameters[
                        "batch_size"
                    ]
                ),
                "resolved_epochs": (
                    hyperparameters["epochs"]
                ),
                "resolved_weight_decay": (
                    hyperparameters[
                        "weight_decay"
                    ]
                ),
                "model_name": MODEL_NAME,
                "samples_per_class": (
                    SAMPLES_PER_CLASS
                ),
                "max_length": MAX_LENGTH,
                "seed": SEED,
                "selection_metric": (
                    WANDB_OBJECTIVE_METRIC
                ),
            },
            allow_val_change=True,
        )

        print("Resolved hyperparameters:")
        print(
            json.dumps(
                hyperparameters,
                indent=2,
            )
        )

        trial_output_dir = (
            OUTPUT_DIR
            / "wandb_trials"
            / run.id
        )

        model = create_model()

        training_args = (
            create_training_arguments(
                output_dir=trial_output_dir,
                learning_rate=(
                    hyperparameters[
                        "learning_rate"
                    ]
                ),
                batch_size=(
                    hyperparameters[
                        "batch_size"
                    ]
                ),
                epochs=(
                    hyperparameters["epochs"]
                ),
                weight_decay=(
                    hyperparameters[
                        "weight_decay"
                    ]
                ),
                run_name=run.name,
            )
        )

        trainer_trial = create_trainer(
            model,
            training_args,
            wandb_run=run,
        )

        try:
            train_result = (
                trainer_trial.train()
            )

            # Best macro-F1 checkpoint is loaded here.
            validation_metrics = (
                trainer_trial.evaluate()
            )

            eval_loss = float(
                validation_metrics[
                    "eval_loss"
                ]
            )
            eval_f1 = float(
                validation_metrics[
                    "eval_f1_macro"
                ]
            )
            eval_accuracy = float(
                validation_metrics[
                    "eval_accuracy"
                ]
            )

            final_metrics = {
                "validation_loss": eval_loss,
                "validation_f1_macro": eval_f1,
                "validation_accuracy": (
                    eval_accuracy
                ),
                "trial_completed": 1,
            }

            run.log(final_metrics)
            run.summary.update(final_metrics)

            record = {
                "wandb_run_id": run.id,
                "wandb_run_name": run.name,
                **hyperparameters,
                "eval_loss": eval_loss,
                "eval_f1_macro": eval_f1,
                "eval_accuracy": eval_accuracy,
                "train_loss": float(
                    train_result.training_loss
                ),
                "best_checkpoint": (
                    trainer_trial
                    .state
                    .best_model_checkpoint
                ),
                "best_metric": (
                    trainer_trial
                    .state
                    .best_metric
                ),
            }

            append_jsonl(
                SWEEP_RESULTS_PATH,
                record,
            )

            print("Completed sweep trial:")
            print(
                json.dumps(
                    record,
                    indent=2,
                    default=str,
                )
            )

        finally:
            del trainer_trial
            del model

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_sweep_agent_safely(
    sweep_path: str,
    trial_count: int,
) -> None:
    import wandb

    try:
        wandb.agent(
            sweep_path,
            function=run_wandb_trial,
            count=trial_count,
        )
    except KeyboardInterrupt:
        print(
            "Sweep was manually interrupted. "
            "The next step will use only runs "
            "that W&B reports as finished."
        )


if WANDB_SWEEP_MODE == "new":
    if SWEEP_RESULTS_PATH.exists():
        SWEEP_RESULTS_PATH.unlink()

    created_sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
    )

    ACTIVE_SWEEP_PATH = build_sweep_path(
        WANDB_ENTITY,
        WANDB_PROJECT,
        created_sweep_id,
    )

    print(
        "Created sweep:",
        ACTIVE_SWEEP_PATH,
    )

    run_sweep_agent_safely(
        ACTIVE_SWEEP_PATH,
        WANDB_SWEEP_COUNT,
    )

elif (
    WANDB_SWEEP_MODE
    == "continue_existing"
):
    ACTIVE_SWEEP_PATH = (
        RESOLVED_SWEEP_PATH
    )

    print(
        "Continuing sweep:",
        ACTIVE_SWEEP_PATH,
    )

    run_sweep_agent_safely(
        ACTIVE_SWEEP_PATH,
        WANDB_SWEEP_COUNT,
    )

elif WANDB_SWEEP_MODE == "reuse_best":
    ACTIVE_SWEEP_PATH = (
        RESOLVED_SWEEP_PATH
    )

    print(
        "No new trials will run. "
        "Best finished run will be loaded from:",
        ACTIVE_SWEEP_PATH,
    )

else:
    ACTIVE_SWEEP_PATH = None

# %%

def normalize_hyperparameters(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    aliases = {
        "learning_rate": [
            "learning_rate",
            "lr",
            "resolved_learning_rate",
        ],
        "batch_size": [
            "batch_size",
            "per_device_train_batch_size",
            "train_batch_size",
            "resolved_batch_size",
        ],
        "epochs": [
            "epochs",
            "num_train_epochs",
            "resolved_epochs",
        ],
        "weight_decay": [
            "weight_decay",
            "resolved_weight_decay",
        ],
    }

    normalized = {}

    for target_name, names in aliases.items():
        value = None

        for name in names:
            if (
                name in config
                and config[name] is not None
            ):
                value = config[name]
                break

        if value is None:
            raise KeyError(
                f"Best run is missing "
                f"{target_name!r}. "
                f"Available keys: "
                f"{sorted(config.keys())}"
            )

        normalized[target_name] = value

    return {
        "learning_rate": float(
            normalized["learning_rate"]
        ),
        "batch_size": int(
            normalized["batch_size"]
        ),
        "epochs": int(
            normalized["epochs"]
        ),
        "weight_decay": float(
            normalized["weight_decay"]
        ),
    }


def get_finished_run_metric(
    run,
    metric_name: str,
) -> Optional[float]:
    if str(run.state).lower() != "finished":
        return None

    value = run.summary.get(
        metric_name
    )

    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    return value


def select_best_finished_run(
    sweep,
    metric_name: str,
    goal: str,
):
    run_records = []
    eligible_runs = []

    for run in sweep.runs:
        metric_value = (
            get_finished_run_metric(
                run,
                metric_name,
            )
        )

        run_records.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "run_url": run.url,
            }
        )

        if metric_value is not None:
            eligible_runs.append(
                (run, metric_value)
            )

    runs_df = pd.DataFrame(
        run_records
    )

    if not eligible_runs:
        raise ValueError(
            "No finished W&B run contains "
            f"{metric_name!r}. "
            "Run the sweep first or use "
            "WANDB_SWEEP_MODE='disabled'."
        )

    eligible_runs.sort(
        key=lambda item: item[1],
        reverse=(goal == "maximize"),
    )

    return eligible_runs[0][0], runs_df


def load_best_finished_hyperparameters(
    sweep_path: str,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    pd.DataFrame,
]:
    import wandb

    last_error = None

    for attempt in range(
        1,
        WANDB_API_RETRIES + 1,
    ):
        try:
            api = wandb.Api()
            sweep = api.sweep(
                sweep_path
            )

            (
                best_run,
                runs_df,
            ) = select_best_finished_run(
                sweep,
                metric_name=(
                    WANDB_OBJECTIVE_METRIC
                ),
                goal=(
                    WANDB_OBJECTIVE_GOAL
                ),
            )

            best_hyperparameters = (
                normalize_hyperparameters(
                    dict(best_run.config)
                )
            )

            metadata = {
                "sweep_path": sweep_path,
                "sweep_name": sweep.name,
                "sweep_state": sweep.state,
                "sweep_url": sweep.url,
                "selection_metric": (
                    WANDB_OBJECTIVE_METRIC
                ),
                "selection_goal": (
                    WANDB_OBJECTIVE_GOAL
                ),
                "best_run_id": best_run.id,
                "best_run_name": (
                    best_run.name
                ),
                "best_run_url": best_run.url,
                "best_run_metric_value": (
                    float(
                        best_run.summary[
                            WANDB_OBJECTIVE_METRIC
                        ]
                    )
                ),
                "best_run_summary": dict(
                    best_run.summary
                ),
                "best_run_config": dict(
                    best_run.config
                ),
            }

            return (
                best_hyperparameters,
                metadata,
                runs_df,
            )

        except ValueError as error:
            last_error = error

            if attempt >= WANDB_API_RETRIES:
                break

            print(
                "No eligible finished run "
                f"visible yet "
                f"({attempt}/"
                f"{WANDB_API_RETRIES}). "
                f"Retrying in "
                f"{WANDB_API_RETRY_SECONDS}s..."
            )

            time.sleep(
                WANDB_API_RETRY_SECONDS
            )

    raise RuntimeError(
        "Could not load a finished W&B "
        "run with validation macro F1."
    ) from last_error


# ---------------------------------------------------------------
# BEST_HYPERPARAMETERS is assigned in this cell, before final model
# training is executed in the next cell.
# ---------------------------------------------------------------
if WANDB_SWEEP_MODE in {
    "new",
    "continue_existing",
    "reuse_best",
}:
    (
        BEST_HYPERPARAMETERS,
        BEST_WANDB_RUN_METADATA,
        WANDB_RUN_CANDIDATES,
    ) = load_best_finished_hyperparameters(
        ACTIVE_SWEEP_PATH
    )

    WANDB_RUN_CANDIDATES.to_csv(
        OUTPUT_DIR
        / "wandb_run_states_and_metrics.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR
        / "best_wandb_run_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            BEST_WANDB_RUN_METADATA,
            file,
            indent=2,
            default=str,
        )

else:
    BEST_HYPERPARAMETERS = (
        DEFAULT_HYPERPARAMETERS.copy()
    )
    BEST_WANDB_RUN_METADATA = {}
    WANDB_RUN_CANDIDATES = (
        pd.DataFrame()
    )


if not isinstance(
    BEST_HYPERPARAMETERS,
    dict,
):
    raise TypeError(
        "BEST_HYPERPARAMETERS was not "
        "created as a dictionary."
    )


required_hyperparameter_keys = {
    "learning_rate",
    "batch_size",
    "epochs",
    "weight_decay",
}

missing_hyperparameter_keys = (
    required_hyperparameter_keys
    - set(BEST_HYPERPARAMETERS)
)

if missing_hyperparameter_keys:
    raise KeyError(
        "BEST_HYPERPARAMETERS is missing: "
        f"{sorted(missing_hyperparameter_keys)}"
    )


print("BEST_HYPERPARAMETERS loaded:")
print(
    json.dumps(
        BEST_HYPERPARAMETERS,
        indent=2,
    )
)

with open(
    OUTPUT_DIR
    / "best_hyperparameters.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        BEST_HYPERPARAMETERS,
        file,
        indent=2,
    )

# %% [markdown]
# 
# ## 7. Train the final model after loading `BEST_HYPERPARAMETERS`
# 
# This cell must run only after the preceding cell has printed:
# 
# ```text
# BEST_HYPERPARAMETERS loaded:
# ```
# 
# The final model is initialized again from the pretrained DistilBERT checkpoint
# and trained using the selected hyperparameters. The test set remains untouched
# until the next section.

# %%

if "BEST_HYPERPARAMETERS" not in globals():
    raise RuntimeError(
        "BEST_HYPERPARAMETERS is not defined. "
        "Run the W&B sweep and best-run "
        "selection cells before this cell."
    )


model = create_model()
final_run = None

if (
    WANDB_SWEEP_MODE != "disabled"
    and LOG_FINAL_TRAINING_TO_WANDB
    and WANDB_MODE != "disabled"
):
    import wandb

    final_run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=(
            "final-training-best-"
            "validation-macro-f1-config"
        ),
        job_type="final-training",
        config={
            **BEST_HYPERPARAMETERS,
            "model_name": MODEL_NAME,
            "samples_per_class": (
                SAMPLES_PER_CLASS
            ),
            "max_length": MAX_LENGTH,
            "seed": SEED,
            "selection_metric": (
                WANDB_OBJECTIVE_METRIC
            ),
        },
        reinit=True,
    )


training_args = create_training_arguments(
    output_dir=(
        OUTPUT_DIR
        / "final_model_training"
    ),
    learning_rate=(
        BEST_HYPERPARAMETERS[
            "learning_rate"
        ]
    ),
    batch_size=(
        BEST_HYPERPARAMETERS[
            "batch_size"
        ]
    ),
    epochs=(
        BEST_HYPERPARAMETERS[
            "epochs"
        ]
    ),
    weight_decay=(
        BEST_HYPERPARAMETERS[
            "weight_decay"
        ]
    ),
    run_name=(
        "final-training-best-"
        "validation-macro-f1-config"
    ),
)


trainer = create_trainer(
    model,
    training_args,
    wandb_run=final_run,
)


train_result = trainer.train()

# load_best_model_at_end=True restores the epoch with the best
# validation macro F1 before this evaluation.
validation_metrics = (
    trainer.evaluate()
)

print("Final validation metrics:")
print(
    json.dumps(
        validation_metrics,
        indent=2,
    )
)

print(
    "Best checkpoint:",
    trainer.state.best_model_checkpoint,
)
print(
    "Best validation macro F1:",
    trainer.state.best_metric,
)


if final_run is not None:
    final_payload = {
        "final_validation_loss": float(
            validation_metrics[
                "eval_loss"
            ]
        ),
        "final_validation_f1_macro": (
            float(
                validation_metrics[
                    "eval_f1_macro"
                ]
            )
        ),
        "final_validation_accuracy": (
            float(
                validation_metrics[
                    "eval_accuracy"
                ]
            )
        ),
        "final_best_checkpoint": (
            trainer
            .state
            .best_model_checkpoint
        ),
        "final_best_metric": float(
            trainer.state.best_metric
        ),
    }

    final_run.log(final_payload)
    final_run.summary.update(
        final_payload
    )

    # Do not finish the W&B run here. The same Trainer instance is used in
    # the next section for held-out test evaluation. The run is finished
    # immediately after the test metrics have been logged.


trainer.save_model(
    OUTPUT_DIR / "best_model"
)
tokenizer.save_pretrained(
    OUTPUT_DIR / "best_model"
)


with open(
    OUTPUT_DIR
    / "final_validation_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            **validation_metrics,
            "best_checkpoint": (
                trainer
                .state
                .best_model_checkpoint
            ),
            "best_metric": (
                trainer.state.best_metric
            ),
            "best_hyperparameters": (
                BEST_HYPERPARAMETERS
            ),
        },
        file,
        indent=2,
        default=str,
    )

# %% [markdown]
# 
# ## 8. Standard held-out test performance
# 
# The final W&B run remains active through this evaluation. After the held-out
# test metrics are logged, the custom W&B callback is removed from the Trainer
# and the run is finished. This prevents later `trainer.evaluate()` or
# `trainer.predict()` calls from trying to write to a finished run.

# %%

test_metrics = trainer.evaluate(
    eval_dataset=tokenized_datasets["test"],
    metric_key_prefix="test",
)

print(json.dumps(test_metrics, indent=2))

with open(
    OUTPUT_DIR / "test_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        test_metrics,
        file,
        indent=2,
    )


# Log held-out test metrics while the final W&B run is still active.
if final_run is not None:
    test_payload = {
        key: float(value)
        for key, value in test_metrics.items()
        if isinstance(
            value,
            (int, float, np.integer, np.floating),
        )
    }

    final_run.log(test_payload)
    final_run.summary.update(test_payload)

    # The Trainer will be reused for prediction and threshold analysis, but
    # those later operations must not try to log to a finished W&B run.
    trainer.remove_callback(
        ManualWandbMetricsCallback
    )

    final_run.finish()
    final_run = None

print(
    "Held-out test evaluation completed. "
    "The final W&B run is now closed."
)

# %% [markdown]
# 
# ## 9. Generate prediction tables with MSP confidence
# 
# For each sample:
# 
# - predicted label: class with the largest softmax probability
# - confidence score: maximum softmax probability (MSP)

# %%

def prediction_dataframe(
    trainer: Trainer,
    tokenized_split: Dataset,
    original_df: pd.DataFrame,
) -> pd.DataFrame:
    prediction_output = trainer.predict(tokenized_split)
    logits = prediction_output.predictions

    # Some models may return a tuple.
    if isinstance(logits, tuple):
        logits = logits[0]

    probabilities = softmax(logits, axis=1)
    predicted_ids = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)

    result = original_df[["sample_id", "text", "label", "label_name"]].copy()
    result["predicted_label"] = predicted_ids.astype(int)
    result["predicted_label_name"] = result["predicted_label"].map(ID_TO_CLASS)
    result["confidence"] = confidence
    result["prob_depression"] = probabilities[:, 0]
    result["prob_neutral"] = probabilities[:, 1]
    result["prob_happy"] = probabilities[:, 2]
    result["phase1_correct"] = result["predicted_label"] == result["label"]

    return result


validation_predictions = prediction_dataframe(
    trainer,
    tokenized_datasets["validation"],
    validation_df,
)

test_predictions = prediction_dataframe(
    trainer,
    tokenized_datasets["test"],
    test_df,
)

validation_predictions.to_csv(
    OUTPUT_DIR / "validation_predictions.csv",
    index=False,
)
test_predictions.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False,
)

print("Validation accuracy:", validation_predictions["phase1_correct"].mean())
print("Test accuracy      :", test_predictions["phase1_correct"].mean())
display(validation_predictions.head())

# %% [markdown]
# 
# ## 10. Risk–coverage threshold analysis
# 
# For a candidate threshold `tau`:
# 
# - Accepted by Phase 1: `confidence >= tau`
# - Routed to Phase 2: `confidence < tau`
# - Coverage: accepted count / total count
# - Selective risk: Phase 1 error rate among accepted rows
# - Error capture rate: proportion of all Phase 1 errors that are routed
# 
# The threshold is selected **only on the validation set**.

# %%

def calculate_threshold_metrics(
    predictions_df: pd.DataFrame,
    tau: float,
) -> Dict[str, float]:
    accepted = predictions_df["confidence"] >= tau
    routed = ~accepted
    errors = ~predictions_df["phase1_correct"]

    n_total = len(predictions_df)
    n_accepted = int(accepted.sum())
    n_routed = int(routed.sum())
    n_errors = int(errors.sum())
    routed_errors = int((routed & errors).sum())

    coverage = n_accepted / n_total if n_total else np.nan
    routing_rate = n_routed / n_total if n_total else np.nan

    if n_accepted > 0:
        accepted_accuracy = predictions_df.loc[
            accepted, "phase1_correct"
        ].mean()
        selective_risk = 1.0 - accepted_accuracy
    else:
        accepted_accuracy = np.nan
        selective_risk = np.nan

    error_capture_rate = (
        routed_errors / n_errors if n_errors > 0 else np.nan
    )

    return {
        "tau": float(tau),
        "n_total": n_total,
        "n_accepted": n_accepted,
        "n_routed": n_routed,
        "coverage": coverage,
        "routing_rate": routing_rate,
        "accepted_accuracy": accepted_accuracy,
        "selective_risk": selective_risk,
        "phase1_errors": n_errors,
        "routed_phase1_errors": routed_errors,
        "error_capture_rate": error_capture_rate,
    }


def threshold_sweep(
    predictions_df: pd.DataFrame,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            calculate_threshold_metrics(predictions_df, float(tau))
            for tau in thresholds
        ]
    )


validation_sweep = threshold_sweep(
    validation_predictions,
    THRESHOLD_GRID,
)

display(validation_sweep.head())

# %%

def select_threshold_by_risk_coverage(
    sweep_df: pd.DataFrame,
    target_selective_risk: float,
    min_accepted_samples: int = 10,
) -> Tuple[float, str, pd.Series]:
    '''
    Select the threshold with the greatest coverage among thresholds satisfying:

        selective_risk <= target_selective_risk

    The threshold is not guaranteed to be 0.85. It depends on:
      - validation predictions
      - model training outcome
      - threshold grid
      - target selective risk
      - minimum accepted-sample requirement
    '''
    candidates = sweep_df[
        sweep_df["selective_risk"].notna()
        & (sweep_df["selective_risk"] <= target_selective_risk)
        & (sweep_df["n_accepted"] >= min_accepted_samples)
    ].copy()

    if not candidates.empty:
        selected_row = candidates.sort_values(
            by=["coverage", "tau"],
            ascending=[False, True],
        ).iloc[0]
        status = "risk_constraint_satisfied"
    else:
        valid_rows = sweep_df[
            sweep_df["selective_risk"].notna()
            & (sweep_df["n_accepted"] >= min_accepted_samples)
        ].copy()

        if valid_rows.empty:
            raise ValueError(
                "No threshold has enough accepted validation samples. "
                "Lower MIN_ACCEPTED_SAMPLES or revise THRESHOLD_GRID."
            )

        selected_row = valid_rows.sort_values(
            by=["selective_risk", "coverage", "tau"],
            ascending=[True, False, True],
        ).iloc[0]
        status = "fallback_minimum_observed_risk"

        print(
            "WARNING: No threshold satisfied the configured selective-risk "
            "constraint. The threshold with the lowest observed validation "
            "selective risk was selected as a fallback."
        )

    return float(selected_row["tau"]), status, selected_row


selected_tau, selection_status, selected_validation_row = (
    select_threshold_by_risk_coverage(
        validation_sweep,
        target_selective_risk=TARGET_SELECTIVE_RISK,
        min_accepted_samples=MIN_ACCEPTED_SAMPLES,
    )
)

print("Selected threshold:", selected_tau)
print("Selection status  :", selection_status)
print("\nSelected validation metrics:")
display(selected_validation_row.to_frame("value"))

def to_python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value

selection_metadata = {
    "selected_tau": float(selected_tau),
    "selection_status": selection_status,
    "target_selective_risk": float(TARGET_SELECTIVE_RISK),
    "minimum_accepted_samples": int(MIN_ACCEPTED_SAMPLES),
    "selected_validation_metrics": {
        key: to_python_scalar(value)
        for key, value in selected_validation_row.to_dict().items()
    },
}

with open(
    OUTPUT_DIR / "selected_threshold.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(selection_metadata, f, indent=2)

# %% [markdown]
# 
# ### Important interpretation
# 
# `selected_tau` is a data-dependent result, not a hard-coded constant.
# 
# For example, if the output is `selected_tau = 0.83`:
# 
# - `confidence < 0.83` → routed to Phase 2
# - `confidence >= 0.83` → accepted by Phase 1
# 
# It will equal `0.85` only when `0.85` is the highest-coverage candidate satisfying the configured validation risk constraint.

# %% [markdown]
# ## 11. Apply the fixed validation-selected threshold to the test set

# %%

def apply_routing(
    predictions_df: pd.DataFrame,
    tau: float,
) -> pd.DataFrame:
    routed_df = predictions_df.copy()
    routed_df["selected_tau"] = tau
    routed_df["is_routed"] = routed_df["confidence"] < tau
    routed_df["routing_decision"] = np.where(
        routed_df["is_routed"],
        "Phase2",
        "Phase1_accept",
    )
    return routed_df


test_routed = apply_routing(test_predictions, selected_tau)

test_selected_metrics = calculate_threshold_metrics(
    test_predictions,
    selected_tau,
)

print("Fixed threshold applied to held-out test set:", selected_tau)
print(json.dumps(test_selected_metrics, indent=2))

display(
    test_routed[
        [
            "sample_id",
            "label_name",
            "predicted_label_name",
            "confidence",
            "phase1_correct",
            "routing_decision",
        ]
    ].sort_values("confidence")
)

test_routed.to_csv(
    OUTPUT_DIR / "test_predictions_with_routing.csv",
    index=False,
)

with open(
    OUTPUT_DIR / "test_routing_metrics.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(test_selected_metrics, f, indent=2)

# %% [markdown]
# ## 12. Paper-friendly threshold sensitivity tables

# %%

validation_report_table = threshold_sweep(
    validation_predictions,
    REPORT_THRESHOLDS,
)

test_report_table = threshold_sweep(
    test_predictions,
    REPORT_THRESHOLDS,
)

validation_report_table.to_csv(
    OUTPUT_DIR / "validation_threshold_sensitivity.csv",
    index=False,
)

test_report_table.to_csv(
    OUTPUT_DIR / "test_threshold_sensitivity.csv",
    index=False,
)

print("Validation threshold sensitivity")
display(validation_report_table)

print("Held-out test threshold sensitivity")
display(test_report_table)

# %% [markdown]
# ## 13. Figures

# %%

# Risk–coverage curve on validation data.
plot_df = validation_sweep.dropna(subset=["selective_risk"]).copy()

plt.figure(figsize=(7, 5))
plt.plot(
    plot_df["coverage"],
    plot_df["selective_risk"],
    marker="o",
    markersize=3,
)

selected_mask = np.isclose(plot_df["tau"], selected_tau)
if selected_mask.any():
    selected_plot_row = plot_df[selected_mask].iloc[0]
    plt.scatter(
        [selected_plot_row["coverage"]],
        [selected_plot_row["selective_risk"]],
        s=100,
        label=f"selected tau={selected_tau:.2f}",
    )
    plt.legend()

plt.xlabel("Coverage")
plt.ylabel("Selective risk")
plt.title("Validation Risk–Coverage Curve")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "validation_risk_coverage_curve.png",
    dpi=200,
)
plt.show()

# %%

# Confidence distributions for correct and incorrect validation predictions.
correct_conf = validation_predictions.loc[
    validation_predictions["phase1_correct"], "confidence"
]
incorrect_conf = validation_predictions.loc[
    ~validation_predictions["phase1_correct"], "confidence"
]

plt.figure(figsize=(7, 5))
plt.hist(
    [correct_conf, incorrect_conf],
    bins=10,
    alpha=0.7,
    label=["Correct", "Incorrect"],
)
plt.axvline(
    selected_tau,
    linestyle="--",
    label=f"selected tau={selected_tau:.2f}",
)
plt.xlabel("Maximum softmax probability")
plt.ylabel("Number of validation samples")
plt.title("Validation Confidence Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "validation_confidence_distribution.png",
    dpi=200,
)
plt.show()

# %% [markdown]
# 
# ## 14. Standard classification report and confusion matrix
# 
# This remains the ordinary Phase 1 test performance and is separate from the routing analysis.

# %%

print(
    classification_report(
        test_predictions["label"],
        test_predictions["predicted_label"],
        target_names=[ID_TO_CLASS[i] for i in range(3)],
        digits=4,
        zero_division=0,
    )
)

cm = confusion_matrix(
    test_predictions["label"],
    test_predictions["predicted_label"],
    labels=[0, 1, 2],
)

cm_df = pd.DataFrame(
    cm,
    index=[f"true_{ID_TO_CLASS[i]}" for i in range(3)],
    columns=[f"pred_{ID_TO_CLASS[i]}" for i in range(3)],
)

display(cm_df)
cm_df.to_csv(OUTPUT_DIR / "test_confusion_matrix.csv")

# %% [markdown]
# 
# ## 15. Optional: integrate real Phase 2 predictions
# 
# The current notebook can decide which test rows should be routed, but it cannot calculate the true final two-phase accuracy until a Phase 2 model produces predictions for those routed rows.
# 
# Prepare a CSV containing:
# 
# - `sample_id`
# - `phase2_predicted_label` as 0, 1, or 2
# 
# Only routed rows require a Phase 2 prediction. After merging that file, run the cell below.

# %%

# Example:
#
# PHASE2_PREDICTIONS_PATH = "./phase2_predictions.csv"
#
# phase2_df = pd.read_csv(PHASE2_PREDICTIONS_PATH)
#
# required_columns = {"sample_id", "phase2_predicted_label"}
# missing_columns = required_columns - set(phase2_df.columns)
# if missing_columns:
#     raise KeyError(f"Missing Phase 2 columns: {missing_columns}")
#
# end_to_end_df = test_routed.merge(
#     phase2_df[["sample_id", "phase2_predicted_label"]],
#     on="sample_id",
#     how="left",
#     validate="one_to_one",
# )
#
# routed_without_phase2 = (
#     end_to_end_df["is_routed"]
#     & end_to_end_df["phase2_predicted_label"].isna()
# )
#
# if routed_without_phase2.any():
#     missing_ids = end_to_end_df.loc[
#         routed_without_phase2, "sample_id"
#     ].tolist()
#     raise ValueError(
#         "Phase 2 predictions are missing for routed sample IDs: "
#         f"{missing_ids}"
#     )
#
# end_to_end_df["final_prediction"] = np.where(
#     end_to_end_df["is_routed"],
#     end_to_end_df["phase2_predicted_label"].astype("Int64"),
#     end_to_end_df["predicted_label"],
# )
#
# end_to_end_df["final_correct"] = (
#     end_to_end_df["final_prediction"] == end_to_end_df["label"]
# )
#
# print("Phase 1 test accuracy:", end_to_end_df["phase1_correct"].mean())
# print("Final two-phase accuracy:", end_to_end_df["final_correct"].mean())
# print("Routing rate:", end_to_end_df["is_routed"].mean())
#
# end_to_end_df.to_csv(
#     OUTPUT_DIR / "reddit_test_end_to_end_predictions.csv",
#     index=False,
# )

# %% [markdown]
# 
# ## 16. Notes
# 
# - `BEST_HYPERPARAMETERS` is created in cell order after the sweep and before final training.
# - Interrupted, crashed, failed, killed, and running W&B runs are excluded.
# - Multiple trainings are normal because each W&B trial is an independent model.
# - Early stopping monitors validation macro F1.
# - The held-out test set is not used for hyperparameter or threshold selection.
# - The 900-row experiment is a smoke test and should not replace the full-data paper result.
# - A true two-phase end-to-end accuracy requires Phase 2 predictions for routed test rows.

