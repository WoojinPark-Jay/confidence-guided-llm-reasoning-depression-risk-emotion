"""
DistilBERT_Reddit_WandB_Advanced_Confidence_Threshold_E2E
"""


# %% [markdown]
# # DistilBERT Reddit Classification + W&B + Advanced Confidence-Threshold Analysis


# %% [cell 1]

# Install dependencies in Colab/Jupyter if needed.
# Notebook-only command: %pip install -q -U transformers datasets accelerate scikit-learn pandas matplotlib scipy wandb


# %% [cell 2]

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
# ## 1. Configuration


# %% [cell 4]

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
SAMPLES_PER_CLASS = 300

# "first_balanced": fast smoke test; stops after enough rows are collected.
# "reservoir": reads the full CSV and reduces source-order sampling bias.
SAMPLING_MODE = "first_balanced"
CSV_CHUNK_SIZE = 5_000

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.10
CALIBRATION_RATIO = 0.10
TEST_RATIO = 0.10

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

WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or None
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "confidence-guided-distilbert-colab")
EXISTING_SWEEP_ID = os.environ.get("WANDB_SWEEP_ID", "")

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
# Advanced confidence-threshold analysis
# -------------------------------------------------------------------
# The model-selection validation split is used only for W&B tuning,
# early stopping, and best-checkpoint selection.
#
# The calibration split is used for temperature scaling and routing
# threshold selection. The held-out test split is not used for either.
PRIMARY_CONFIDENCE_METHOD = "temperature_scaled_msp"

CONFIDENCE_METHODS = {
    "raw_msp": "raw_msp",
    "temperature_scaled_msp": "calibrated_msp",
    "entropy_certainty": "raw_entropy_certainty",
    "probability_margin": "raw_margin",
}

TARGET_SELECTIVE_RISK = 0.05
RISK_CONFIDENCE_DELTA = 0.05

# "upper_bound" uses a one-sided Clopper-Pearson upper confidence bound.
# "empirical" uses the observed accepted-set error rate.
RISK_CONTROL_METHOD = "upper_bound"

# For a smoke test, an explicit fallback keeps the pipeline executable
# when the small calibration split cannot satisfy the risk constraint.
# For the final paper experiment, set this to False.
ALLOW_EXPLICIT_RISK_FALLBACK = True

MIN_ACCEPTED_COUNT = 10
MIN_ACCEPTED_FRACTION = 0.10

ALPHA_SENSITIVITY_VALUES = [
    0.01,
    0.03,
    0.05,
    0.10,
]

ECE_BIN_COUNTS = [
    10,
    15,
    20,
]
PRIMARY_ECE_BINS = 15

BOOTSTRAP_ITERATIONS = 1000
THRESHOLD_STABILITY_BOOTSTRAPS = 300
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

ROUTING_BUDGETS = [
    0.10,
    0.20,
    0.25,
]

RUN_CLASS_CONDITIONAL_THRESHOLD_ABLATION = True

# Optional empirical cost-sensitive ablation. Disabled by default because
# misclassification weights require a defensible domain-specific rationale.
RUN_COST_SENSITIVE_ABLATION = False

COST_MATRIX = np.asarray([
    [0.0, 2.0, 3.0],  # true Depression -> predicted D/N/H
    [1.0, 0.0, 1.0],  # true Neutral
    [1.0, 1.0, 0.0],  # true Happy
], dtype=np.float64)

# Optional external stress-test evaluation. The Reddit-selected temperature
# and threshold are reused without re-selection.
MIXED_EMOTION_CSV_PATH = None
MIXED_EMOTION_TEXT_COLUMN = "text"
MIXED_EMOTION_LABEL_COLUMN = "label"
MIXED_EMOTION_SCENARIO_COLUMN = "scenario_type"

# Optional real Phase 2 result integration.
PHASE2_PREDICTIONS_PATH = None


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
print("W&B entity/project:", f"{WANDB_ENTITY}/{WANDB_PROJECT}")
print(
    "Selection objective:",
    WANDB_OBJECTIVE_METRIC,
    WANDB_OBJECTIVE_GOAL,
)

if USE_EXISTING_SWEEP:
    print("Existing sweep ID:", EXISTING_SWEEP_ID)


# %% [markdown]
# ## W&B mode selection


# %% [markdown]
# ## 2. Detect columns and sample 300 rows per class


# %% [cell 7]

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


# %% [cell 8]

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
# ## 3. Stratified train/model-validation/threshold-calibration/test split


# %% [cell 10]

def stratified_four_way_split(
    dataframe: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    calibration_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    total_ratio = (
        train_ratio
        + validation_ratio
        + calibration_ratio
        + test_ratio
    )

    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            "Train, validation, calibration, and test ratios "
            "must sum to 1."
        )

    train_split, remaining = train_test_split(
        dataframe,
        test_size=1.0 - train_ratio,
        random_state=seed,
        stratify=dataframe["label"],
    )

    remaining_ratio = (
        validation_ratio
        + calibration_ratio
        + test_ratio
    )

    validation_share = (
        validation_ratio / remaining_ratio
    )

    validation_split, calibration_and_test = train_test_split(
        remaining,
        train_size=validation_share,
        random_state=seed,
        stratify=remaining["label"],
    )

    calibration_share = (
        calibration_ratio
        / (calibration_ratio + test_ratio)
    )

    calibration_split, test_split = train_test_split(
        calibration_and_test,
        train_size=calibration_share,
        random_state=seed,
        stratify=calibration_and_test["label"],
    )

    return (
        train_split.reset_index(drop=True),
        validation_split.reset_index(drop=True),
        calibration_split.reset_index(drop=True),
        test_split.reset_index(drop=True),
    )


(
    train_df,
    validation_df,
    calibration_df,
    test_df,
) = stratified_four_way_split(
    sampled_df,
    train_ratio=TRAIN_RATIO,
    validation_ratio=VALIDATION_RATIO,
    calibration_ratio=CALIBRATION_RATIO,
    test_ratio=TEST_RATIO,
    seed=SEED,
)

for split_name, split_dataframe in {
    "train": train_df,
    "model_validation": validation_df,
    "threshold_calibration": calibration_df,
    "held_out_test": test_df,
}.items():
    print(
        f"\n{split_name}: "
        f"{len(split_dataframe)}"
    )
    print(
        split_dataframe[
            "label_name"
        ].value_counts().sort_index()
    )

train_df.to_csv(
    OUTPUT_DIR / "train_sample.csv",
    index=False,
)
validation_df.to_csv(
    OUTPUT_DIR
    / "model_validation_sample.csv",
    index=False,
)
calibration_df.to_csv(
    OUTPUT_DIR
    / "threshold_calibration_sample.csv",
    index=False,
)
test_df.to_csv(
    OUTPUT_DIR / "test_sample.csv",
    index=False,
)


# %% [markdown]
# ## 4. Build Hugging Face datasets and tokenize


# %% [cell 12]

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

dataset_dict = DatasetDict({
    "train": Dataset.from_pandas(
        train_df[
            ["sample_id", "text", "label"]
        ],
        preserve_index=False,
    ),
    "validation": Dataset.from_pandas(
        validation_df[
            ["sample_id", "text", "label"]
        ],
        preserve_index=False,
    ),
    "calibration": Dataset.from_pandas(
        calibration_df[
            ["sample_id", "text", "label"]
        ],
        preserve_index=False,
    ),
    "test": Dataset.from_pandas(
        test_df[
            ["sample_id", "text", "label"]
        ],
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

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

print(tokenized_datasets)


# %% [markdown]
# ## 5. Define DistilBERT training helpers


# %% [cell 14]

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
# ## 6. W&B Macro-F1 hyperparameter sweep


# %% [cell 16]

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


if WANDB_SWEEP_MODE != "disabled":
    import wandb

    os.environ["WANDB_MODE"] = WANDB_MODE
    try:
        from google.colab import userdata
        wandb_api_key = userdata.get("WANDB_API_KEY")
    except Exception:
        wandb_api_key = None

    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        wandb.login()


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


# %% [cell 17]

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


# %% [cell 18]

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
# ## 7. Train the final model after loading `BEST_HYPERPARAMETERS`


# %% [cell 20]

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
# ## 8. Standard held-out test performance


# %% [cell 22]

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
# ## Advanced confidence-threshold analysis


# %% [cell 24]

from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import beta as beta_distribution


ADVANCED_THRESHOLD_DIR = (
    OUTPUT_DIR
    / "advanced_confidence_threshold"
)
ADVANCED_THRESHOLD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ANALYSIS_MODEL_NAME = MODEL_NAME

# Deterministic standard inference for MSP analysis.
trainer.model.eval()


def extract_logits(
    prediction_output,
) -> np.ndarray:
    logits = prediction_output.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    # FP16/BF16 model outputs are converted to FP32 before softmax.
    return np.asarray(
        logits,
        dtype=np.float32,
    )


def predict_split_logits(
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    output = trainer.predict(
        tokenized_datasets[split_name]
    )

    logits = extract_logits(output)
    labels = np.asarray(
        output.label_ids,
        dtype=np.int64,
    )

    return logits, labels


calibration_logits, calibration_labels = (
    predict_split_logits(
        "calibration"
    )
)

test_logits, test_labels = (
    predict_split_logits(
        "test"
    )
)

print(
    "Calibration logits:",
    calibration_logits.shape,
)
print(
    "Test logits:",
    test_logits.shape,
)


# %% [markdown]
# ### Temperature scaling on the dedicated calibration split


# %% [cell 26]

def softmax_fp32(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(
            "temperature must be positive."
        )

    scaled_logits = (
        np.asarray(
            logits,
            dtype=np.float32,
        )
        / np.float32(temperature)
    )

    return softmax(
        scaled_logits,
        axis=1,
    ).astype(np.float32)


def negative_log_likelihood_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float = 1.0,
) -> float:
    scaled_logits = (
        np.asarray(
            logits,
            dtype=np.float64,
        )
        / float(temperature)
    )

    log_probabilities = (
        scaled_logits
        - logsumexp(
            scaled_logits,
            axis=1,
            keepdims=True,
        )
    )

    return float(
        -np.mean(
            log_probabilities[
                np.arange(len(labels)),
                labels,
            ]
        )
    )


def fit_temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    before_nll = (
        negative_log_likelihood_from_logits(
            logits,
            labels,
            temperature=1.0,
        )
    )

    result = minimize_scalar(
        lambda log_temperature: (
            negative_log_likelihood_from_logits(
                logits,
                labels,
                temperature=float(
                    np.exp(log_temperature)
                ),
            )
        ),
        bounds=(-3.0, 3.0),
        method="bounded",
        options={
            "xatol": 1e-6,
        },
    )

    temperature = float(
        np.exp(result.x)
    )

    after_nll = (
        negative_log_likelihood_from_logits(
            logits,
            labels,
            temperature=temperature,
        )
    )

    return {
        "temperature": temperature,
        "optimization_success": bool(
            result.success
        ),
        "optimization_message": str(
            result.message
        ),
        "nll_before": before_nll,
        "nll_after": after_nll,
    }


temperature_result = (
    fit_temperature_scaling(
        calibration_logits,
        calibration_labels,
    )
)

SELECTED_TEMPERATURE = (
    temperature_result["temperature"]
)

print(
    "Temperature scaling result:"
)
print(
    json.dumps(
        temperature_result,
        indent=2,
    )
)

with open(
    ADVANCED_THRESHOLD_DIR
    / "temperature_scaling.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        temperature_result,
        file,
        indent=2,
    )


# %% [markdown]
# ### FP32 probabilities and confidence-score tables


# %% [cell 28]

def probability_margin(
    probabilities: np.ndarray,
) -> np.ndarray:
    sorted_probabilities = np.sort(
        probabilities,
        axis=1,
    )

    return (
        sorted_probabilities[:, -1]
        - sorted_probabilities[:, -2]
    )


def normalized_entropy_certainty(
    probabilities: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    entropy = -np.sum(
        clipped * np.log(clipped),
        axis=1,
    )

    maximum_entropy = np.log(
        probabilities.shape[1]
    )

    return (
        1.0
        - entropy / maximum_entropy
    )


def compute_token_lengths(
    texts: Iterable[str],
) -> np.ndarray:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )

    return np.asarray([
        len(input_ids)
        for input_ids
        in encoded["input_ids"]
    ])


def build_advanced_prediction_dataframe(
    original_dataframe: pd.DataFrame,
    logits: np.ndarray,
    temperature: float,
) -> pd.DataFrame:
    raw_probabilities = softmax_fp32(
        logits,
        temperature=1.0,
    )

    calibrated_probabilities = softmax_fp32(
        logits,
        temperature=temperature,
    )

    raw_predictions = np.argmax(
        raw_probabilities,
        axis=1,
    ).astype(int)

    calibrated_predictions = np.argmax(
        calibrated_probabilities,
        axis=1,
    ).astype(int)

    result = original_dataframe[
        [
            "sample_id",
            "text",
            "label",
            "label_name",
        ]
    ].copy().reset_index(drop=True)

    result["predicted_label"] = (
        raw_predictions
    )
    result["predicted_label_name"] = (
        result[
            "predicted_label"
        ].map(ID_TO_CLASS)
    )
    result["calibrated_predicted_label"] = (
        calibrated_predictions
    )

    result["raw_msp"] = np.max(
        raw_probabilities,
        axis=1,
    )
    result["calibrated_msp"] = np.max(
        calibrated_probabilities,
        axis=1,
    )
    result["raw_entropy_certainty"] = (
        normalized_entropy_certainty(
            raw_probabilities
        )
    )
    result["raw_margin"] = (
        probability_margin(
            raw_probabilities
        )
    )

    for class_id, class_name in (
        ID_TO_CLASS.items()
    ):
        normalized_name = (
            class_name.lower()
        )

        result[
            f"raw_prob_{normalized_name}"
        ] = raw_probabilities[
            :,
            class_id,
        ]

        result[
            f"calibrated_prob_{normalized_name}"
        ] = calibrated_probabilities[
            :,
            class_id,
        ]

    result["phase1_correct"] = (
        result["predicted_label"]
        == result["label"]
    )

    token_lengths = compute_token_lengths(
        result["text"].tolist()
    )

    result["token_length"] = (
        token_lengths
    )
    result["was_truncated"] = (
        token_lengths > MAX_LENGTH
    )

    return result


calibration_predictions = (
    build_advanced_prediction_dataframe(
        calibration_df,
        calibration_logits,
        SELECTED_TEMPERATURE,
    )
)

test_predictions = (
    build_advanced_prediction_dataframe(
        test_df,
        test_logits,
        SELECTED_TEMPERATURE,
    )
)

calibration_predictions.to_csv(
    ADVANCED_THRESHOLD_DIR
    / "calibration_predictions.csv",
    index=False,
)

test_predictions.to_csv(
    ADVANCED_THRESHOLD_DIR
    / "test_predictions.csv",
    index=False,
)

print(
    "Calibration accuracy:",
    calibration_predictions[
        "phase1_correct"
    ].mean(),
)
print(
    "Test accuracy:",
    test_predictions[
        "phase1_correct"
    ].mean(),
)

display(
    calibration_predictions.head()
)


# %% [markdown]
# ### Calibration metrics: ECE, adaptive ECE, Brier score, and NLL


# %% [cell 30]

def multiclass_brier_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    one_hot = np.eye(
        probabilities.shape[1],
        dtype=np.float64,
    )[labels]

    return float(
        np.mean(
            np.sum(
                (
                    probabilities
                    - one_hot
                ) ** 2,
                axis=1,
            )
        )
    )


def reliability_bins_equal_width(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    predictions = np.argmax(
        probabilities,
        axis=1,
    )
    confidence = np.max(
        probabilities,
        axis=1,
    )
    correctness = (
        predictions == labels
    ).astype(float)

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    rows = []

    for bin_index in range(n_bins):
        lower = edges[bin_index]
        upper = edges[bin_index + 1]

        if bin_index == n_bins - 1:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        count = int(mask.sum())

        rows.append({
            "bin_index": bin_index,
            "lower": lower,
            "upper": upper,
            "count": count,
            "mean_confidence": (
                float(
                    confidence[mask].mean()
                )
                if count > 0
                else np.nan
            ),
            "accuracy": (
                float(
                    correctness[mask].mean()
                )
                if count > 0
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
) -> float:
    bins = (
        reliability_bins_equal_width(
            probabilities,
            labels,
            n_bins,
        )
    )

    total = len(labels)

    return float(
        np.nansum(
            (
                bins["count"] / total
            )
            * np.abs(
                bins["accuracy"]
                - bins["mean_confidence"]
            )
        )
    )


def adaptive_expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
) -> float:
    predictions = np.argmax(
        probabilities,
        axis=1,
    )
    confidence = np.max(
        probabilities,
        axis=1,
    )
    correctness = (
        predictions == labels
    ).astype(float)

    order = np.argsort(
        confidence
    )

    groups = np.array_split(
        order,
        n_bins,
    )

    total = len(labels)
    adaptive_ece = 0.0

    for group in groups:
        if len(group) == 0:
            continue

        group_confidence = float(
            confidence[group].mean()
        )
        group_accuracy = float(
            correctness[group].mean()
        )

        adaptive_ece += (
            len(group) / total
        ) * abs(
            group_accuracy
            - group_confidence
        )

    return float(adaptive_ece)


def calibration_metric_summary(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
) -> pd.DataFrame:
    rows = []

    for confidence_name, candidate_temperature in [
        ("raw_msp", 1.0),
        (
            "temperature_scaled_msp",
            temperature,
        ),
    ]:
        probabilities = softmax_fp32(
            logits,
            candidate_temperature,
        )

        row = {
            "confidence_method": (
                confidence_name
            ),
            "temperature": float(
                candidate_temperature
            ),
            "nll": (
                negative_log_likelihood_from_logits(
                    logits,
                    labels,
                    candidate_temperature,
                )
            ),
            "brier": (
                multiclass_brier_score(
                    probabilities,
                    labels,
                )
            ),
            "adaptive_ece_15": (
                adaptive_expected_calibration_error(
                    probabilities,
                    labels,
                    PRIMARY_ECE_BINS,
                )
            ),
        }

        for n_bins in ECE_BIN_COUNTS:
            row[
                f"ece_equal_width_{n_bins}"
            ] = expected_calibration_error(
                probabilities,
                labels,
                n_bins,
            )

        rows.append(row)

    return pd.DataFrame(rows)


calibration_quality_table = (
    calibration_metric_summary(
        calibration_logits,
        calibration_labels,
        SELECTED_TEMPERATURE,
    )
)

calibration_quality_table.to_csv(
    ADVANCED_THRESHOLD_DIR
    / "calibration_quality_metrics.csv",
    index=False,
)

display(
    calibration_quality_table
)


# %% [markdown]
# ### Exact threshold candidates and selective-risk control


# %% [cell 32]

def one_sided_binomial_upper_bound(
    error_count: int,
    accepted_count: int,
    delta: float,
) -> float:
    if accepted_count <= 0:
        return np.nan

    if error_count >= accepted_count:
        return 1.0

    return float(
        beta_distribution.ppf(
            1.0 - delta,
            error_count + 1,
            accepted_count - error_count,
        )
    )


def build_exact_threshold_candidates(
    scores: Iterable[float],
) -> np.ndarray:
    unique_scores = np.sort(
        np.unique(
            np.asarray(
                list(scores),
                dtype=np.float64,
            )
        )
    )

    if len(unique_scores) == 0:
        raise ValueError(
            "No confidence scores were supplied."
        )

    if len(unique_scores) == 1:
        return np.asarray([
            np.nextafter(
                unique_scores[0],
                -np.inf,
            ),
            np.nextafter(
                unique_scores[0],
                np.inf,
            ),
        ])

    midpoints = (
        unique_scores[:-1]
        + unique_scores[1:]
    ) / 2.0

    return np.unique(
        np.concatenate([
            [
                np.nextafter(
                    unique_scores[0],
                    -np.inf,
                )
            ],
            midpoints,
            [
                np.nextafter(
                    unique_scores[-1],
                    np.inf,
                )
            ],
        ])
    )


def calculate_selective_metrics(
    prediction_dataframe: pd.DataFrame,
    score_column: str,
    threshold: float,
    delta: float,
) -> Dict[str, Any]:
    accepted = (
        prediction_dataframe[
            score_column
        ].to_numpy()
        >= threshold
    )
    routed = ~accepted

    correct = (
        prediction_dataframe[
            "phase1_correct"
        ].to_numpy(dtype=bool)
    )
    errors = ~correct

    total_count = len(
        prediction_dataframe
    )
    accepted_count = int(
        accepted.sum()
    )
    routed_count = int(
        routed.sum()
    )
    total_errors = int(
        errors.sum()
    )
    accepted_errors = int(
        (accepted & errors).sum()
    )
    routed_errors = int(
        (routed & errors).sum()
    )

    empirical_risk = (
        accepted_errors / accepted_count
        if accepted_count > 0
        else np.nan
    )

    risk_upper_bound = (
        one_sided_binomial_upper_bound(
            accepted_errors,
            accepted_count,
            delta,
        )
    )

    routing_precision = (
        routed_errors / routed_count
        if routed_count > 0
        else np.nan
    )

    error_capture_rate = (
        routed_errors / total_errors
        if total_errors > 0
        else np.nan
    )

    true_labels = (
        prediction_dataframe[
            "label"
        ].to_numpy(dtype=int)
    )
    predicted_labels = (
        prediction_dataframe[
            "predicted_label"
        ].to_numpy(dtype=int)
    )

    accepted_depression = (
        accepted
        & (true_labels == 0)
    )

    accepted_depression_count = int(
        accepted_depression.sum()
    )

    accepted_depression_false_negatives = int(
        (
            accepted_depression
            & (predicted_labels != 0)
        ).sum()
    )

    depression_false_negative_risk = (
        accepted_depression_false_negatives
        / accepted_depression_count
        if accepted_depression_count > 0
        else np.nan
    )

    return {
        "score_column": score_column,
        "tau": float(threshold),
        "n_total": total_count,
        "n_accepted": accepted_count,
        "n_routed": routed_count,
        "coverage": (
            accepted_count / total_count
            if total_count > 0
            else np.nan
        ),
        "routing_rate": (
            routed_count / total_count
            if total_count > 0
            else np.nan
        ),
        "accepted_accuracy": (
            1.0 - empirical_risk
            if accepted_count > 0
            else np.nan
        ),
        "selective_risk": (
            empirical_risk
        ),
        "selective_risk_upper_bound": (
            risk_upper_bound
        ),
        "phase1_errors": total_errors,
        "accepted_phase1_errors": (
            accepted_errors
        ),
        "routed_phase1_errors": (
            routed_errors
        ),
        "error_capture_rate": (
            error_capture_rate
        ),
        "routing_precision": (
            routing_precision
        ),
        "accepted_depression_count": (
            accepted_depression_count
        ),
        "accepted_depression_false_negatives": (
            accepted_depression_false_negatives
        ),
        "accepted_depression_false_negative_risk": (
            depression_false_negative_risk
        ),
    }


def threshold_sweep(
    prediction_dataframe: pd.DataFrame,
    score_column: str,
    delta: float,
) -> pd.DataFrame:
    candidates = (
        build_exact_threshold_candidates(
            prediction_dataframe[
                score_column
            ].to_numpy()
        )
    )

    return pd.DataFrame([
        calculate_selective_metrics(
            prediction_dataframe,
            score_column,
            threshold,
            delta,
        )
        for threshold in candidates
    ])


def minimum_accepted_samples(
    sample_count: int,
) -> int:
    return max(
        int(MIN_ACCEPTED_COUNT),
        int(
            np.ceil(
                MIN_ACCEPTED_FRACTION
                * sample_count
            )
        ),
    )


def select_threshold(
    sweep_dataframe: pd.DataFrame,
    alpha: float,
    risk_control_method: str,
    allow_fallback: bool,
) -> Dict[str, Any]:
    if risk_control_method not in {
        "empirical",
        "upper_bound",
    }:
        raise ValueError(
            "risk_control_method must be "
            "'empirical' or 'upper_bound'."
        )

    risk_column = (
        "selective_risk"
        if risk_control_method
        == "empirical"
        else "selective_risk_upper_bound"
    )

    minimum_count = (
        minimum_accepted_samples(
            int(
                sweep_dataframe[
                    "n_total"
                ].iloc[0]
            )
        )
    )

    valid = sweep_dataframe[
        sweep_dataframe[
            risk_column
        ].notna()
        & (
            sweep_dataframe[
                "n_accepted"
            ]
            >= minimum_count
        )
    ].copy()

    feasible = valid[
        valid[risk_column] <= alpha
    ].copy()

    if not feasible.empty:
        # Deterministic tie breaking:
        # 1. maximum coverage
        # 2. minimum controlled risk
        # 3. minimum empirical risk
        # 4. lower threshold
        selected = (
            feasible.sort_values(
                by=[
                    "coverage",
                    risk_column,
                    "selective_risk",
                    "tau",
                ],
                ascending=[
                    False,
                    True,
                    True,
                    True,
                ],
            ).iloc[0]
        )

        return {
            "selected_tau": float(
                selected["tau"]
            ),
            "selection_status": (
                "risk_constraint_satisfied"
            ),
            "risk_constraint_satisfied": True,
            "risk_control_method": (
                risk_control_method
            ),
            "risk_column": risk_column,
            "alpha": float(alpha),
            "minimum_accepted_samples": (
                minimum_count
            ),
            "selected_metrics": (
                selected.to_dict()
            ),
        }

    if not allow_fallback:
        return {
            "selected_tau": None,
            "selection_status": (
                "risk_constraint_infeasible"
            ),
            "risk_constraint_satisfied": False,
            "risk_control_method": (
                risk_control_method
            ),
            "risk_column": risk_column,
            "alpha": float(alpha),
            "minimum_accepted_samples": (
                minimum_count
            ),
            "selected_metrics": None,
        }

    if valid.empty:
        raise ValueError(
            "No threshold has enough accepted "
            "calibration samples."
        )

    selected = (
        valid.sort_values(
            by=[
                risk_column,
                "selective_risk",
                "coverage",
                "tau",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
        ).iloc[0]
    )

    return {
        "selected_tau": float(
            selected["tau"]
        ),
        "selection_status": (
            "fallback_minimum_observed_risk"
        ),
        "risk_constraint_satisfied": False,
        "risk_control_method": (
            risk_control_method
        ),
        "risk_column": risk_column,
        "alpha": float(alpha),
        "minimum_accepted_samples": (
            minimum_count
        ),
        "selected_metrics": (
            selected.to_dict()
        ),
    }


def aurc_and_eaurc(
    prediction_dataframe: pd.DataFrame,
    score_column: str,
) -> Dict[str, float]:
    scores = prediction_dataframe[
        score_column
    ].to_numpy(dtype=float)

    errors = (
        ~prediction_dataframe[
            "phase1_correct"
        ].to_numpy(dtype=bool)
    ).astype(float)

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    ordered_errors = errors[order]
    counts = np.arange(
        1,
        len(errors) + 1,
    )

    risk_curve = (
        np.cumsum(
            ordered_errors
        )
        / counts
    )
    coverage_curve = (
        counts / len(errors)
    )

    aurc = float(
        np.trapz(
            np.concatenate([
                [0.0],
                risk_curve,
            ]),
            np.concatenate([
                [0.0],
                coverage_curve,
            ]),
        )
    )

    oracle_errors = np.sort(
        errors
    )

    oracle_risk_curve = (
        np.cumsum(
            oracle_errors
        )
        / counts
    )

    oracle_aurc = float(
        np.trapz(
            np.concatenate([
                [0.0],
                oracle_risk_curve,
            ]),
            np.concatenate([
                [0.0],
                coverage_curve,
            ]),
        )
    )

    return {
        "aurc": aurc,
        "oracle_aurc": oracle_aurc,
        "eaurc": (
            aurc - oracle_aurc
        ),
    }


# %% [markdown]
# ### Confidence-score ablation and model-specific threshold selection


# %% [cell 34]

analysis_rows = []
analysis_objects = {}

for (
    confidence_method,
    score_column,
) in CONFIDENCE_METHODS.items():
    calibration_sweep = threshold_sweep(
        calibration_predictions,
        score_column,
        RISK_CONFIDENCE_DELTA,
    )

    selection = select_threshold(
        calibration_sweep,
        alpha=TARGET_SELECTIVE_RISK,
        risk_control_method=(
            RISK_CONTROL_METHOD
        ),
        allow_fallback=(
            ALLOW_EXPLICIT_RISK_FALLBACK
        ),
    )

    selected_tau = (
        selection["selected_tau"]
    )

    if selected_tau is None:
        test_metrics_for_method = None
    else:
        test_metrics_for_method = (
            calculate_selective_metrics(
                test_predictions,
                score_column,
                selected_tau,
                RISK_CONFIDENCE_DELTA,
            )
        )

    calibration_ranking = (
        aurc_and_eaurc(
            calibration_predictions,
            score_column,
        )
    )

    test_ranking = aurc_and_eaurc(
        test_predictions,
        score_column,
    )

    summary_row = {
        "confidence_method": (
            confidence_method
        ),
        "score_column": score_column,
        "selected_tau": selected_tau,
        "selection_status": (
            selection[
                "selection_status"
            ]
        ),
        "risk_constraint_satisfied": (
            selection[
                "risk_constraint_satisfied"
            ]
        ),
        "calibration_aurc": (
            calibration_ranking[
                "aurc"
            ]
        ),
        "calibration_eaurc": (
            calibration_ranking[
                "eaurc"
            ]
        ),
        "test_aurc": (
            test_ranking["aurc"]
        ),
        "test_eaurc": (
            test_ranking["eaurc"]
        ),
    }

    if test_metrics_for_method:
        summary_row.update({
            f"test_{key}": value
            for key, value
            in test_metrics_for_method.items()
            if key not in {
                "score_column",
            }
        })

    analysis_rows.append(
        summary_row
    )

    analysis_objects[
        confidence_method
    ] = {
        "score_column": score_column,
        "calibration_sweep": (
            calibration_sweep
        ),
        "selection": selection,
        "test_metrics": (
            test_metrics_for_method
        ),
        "calibration_ranking": (
            calibration_ranking
        ),
        "test_ranking": (
            test_ranking
        ),
    }

    calibration_sweep.to_csv(
        ADVANCED_THRESHOLD_DIR
        / (
            f"{confidence_method}_"
            "calibration_threshold_sweep.csv"
        ),
        index=False,
    )


confidence_method_comparison = (
    pd.DataFrame(
        analysis_rows
    )
)

confidence_method_comparison.to_csv(
    ADVANCED_THRESHOLD_DIR
    / "confidence_method_comparison.csv",
    index=False,
)

display(
    confidence_method_comparison
)


if (
    PRIMARY_CONFIDENCE_METHOD
    not in analysis_objects
):
    raise KeyError(
        "PRIMARY_CONFIDENCE_METHOD was not "
        "found in CONFIDENCE_METHODS."
    )


primary_analysis = (
    analysis_objects[
        PRIMARY_CONFIDENCE_METHOD
    ]
)

PRIMARY_SCORE_COLUMN = (
    primary_analysis[
        "score_column"
    ]
)

PRIMARY_THRESHOLD_SELECTION = (
    primary_analysis[
        "selection"
    ]
)

SELECTED_THRESHOLD = (
    PRIMARY_THRESHOLD_SELECTION[
        "selected_tau"
    ]
)

if SELECTED_THRESHOLD is None:
    raise RuntimeError(
        "The primary confidence method did "
        "not produce a feasible threshold and "
        "fallback was disabled."
    )

print(
    "Primary confidence method:",
    PRIMARY_CONFIDENCE_METHOD,
)
print(
    "Primary score column:",
    PRIMARY_SCORE_COLUMN,
)
print(
    "Selected threshold:",
    SELECTED_THRESHOLD,
)
print(
    "Selection status:",
    PRIMARY_THRESHOLD_SELECTION[
        "selection_status"
    ],
)
print(
    "Risk constraint satisfied:",
    PRIMARY_THRESHOLD_SELECTION[
        "risk_constraint_satisfied"
    ],
)


# %% [markdown]
# ### Target-risk sensitivity and routing-budget analysis


# %% [cell 36]

primary_calibration_sweep = (
    primary_analysis[
        "calibration_sweep"
    ]
)

alpha_rows = []

for alpha in ALPHA_SENSITIVITY_VALUES:
    selection = select_threshold(
        primary_calibration_sweep,
        alpha=alpha,
        risk_control_method=(
            RISK_CONTROL_METHOD
        ),
        allow_fallback=(
            ALLOW_EXPLICIT_RISK_FALLBACK
        ),
    )

    tau = selection[
        "selected_tau"
    ]

    row = {
        "alpha": alpha,
        "selected_tau": tau,
        "selection_status": (
            selection[
                "selection_status"
            ]
        ),
        "risk_constraint_satisfied": (
            selection[
                "risk_constraint_satisfied"
            ]
        ),
    }

    if tau is not None:
        test_metrics_alpha = (
            calculate_selective_metrics(
                test_predictions,
                PRIMARY_SCORE_COLUMN,
                tau,
                RISK_CONFIDENCE_DELTA,
            )
        )
        row.update({
            f"test_{key}": value
            for key, value
            in test_metrics_alpha.items()
            if key != "score_column"
        })

    alpha_rows.append(row)


alpha_sensitivity_table = (
    pd.DataFrame(alpha_rows)
)

alpha_sensitivity_table.to_csv(
    ADVANCED_THRESHOLD_DIR
    / "alpha_sensitivity.csv",
    index=False,
)

display(
    alpha_sensitivity_table
)


def select_budget_threshold(
    sweep_dataframe: pd.DataFrame,
    routing_budget: float,
) -> Optional[pd.Series]:
    eligible = sweep_dataframe[
        sweep_dataframe[
            "routing_rate"
        ]
        <= routing_budget
    ].copy()

    if eligible.empty:
        return None

    return eligible.sort_values(
        by=[
            "error_capture_rate",
            "selective_risk",
            "coverage",
            "tau",
        ],
        ascending=[
            False,
            True,
            False,
            True,
        ],
    ).iloc[0]


budget_rows = []

for budget in ROUTING_BUDGETS:
    selected_budget_row = (
        select_budget_threshold(
            primary_calibration_sweep,
            budget,
        )
    )

    if selected_budget_row is None:
        budget_rows.append({
            "routing_budget": budget,
            "selection_status": (
                "infeasible"
            ),
        })
        continue

    budget_tau = float(
        selected_budget_row["tau"]
    )

    budget_test_metrics = (
        calculate_selective_metrics(
            test_predictions,
            PRIMARY_SCORE_COLUMN,
            budget_tau,
            RISK_CONFIDENCE_DELTA,
        )
    )

    budget_rows.append({
        "routing_budget": budget,
        "selection_status": "selected",
        "selected_tau": budget_tau,
        **{
            f"calibration_{key}": value
            for key, value
            in selected_budget_row.to_dict().items()
        },
        **{
            f"test_{key}": value
            for key, value
            in budget_test_metrics.items()
            if key != "score_column"
        },
    })


routing_budget_table = (
    pd.DataFrame(
        budget_rows
    )
)

routing_budget_table.to_csv(
    ADVANCED_THRESHOLD_DIR
    / "routing_budget_analysis.csv",
    index=False,
)

display(
    routing_budget_table
)


# %% [markdown]
# ### Bootstrap confidence intervals, threshold stability, and high-confidence errors


# %% [cell 38]

def bootstrap_fixed_threshold_metrics(
    prediction_dataframe: pd.DataFrame,
    score_column: str,
    threshold: float,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        seed
    )

    records = []
    sample_count = len(
        prediction_dataframe
    )

    for iteration in range(
        iterations
    ):
        sampled_indices = rng.integers(
            0,
            sample_count,
            size=sample_count,
        )

        sampled_dataframe = (
            prediction_dataframe.iloc[
                sampled_indices
            ].reset_index(drop=True)
        )

        metrics = (
            calculate_selective_metrics(
                sampled_dataframe,
                score_column,
                threshold,
                RISK_CONFIDENCE_DELTA,
            )
        )

        records.append({
            "iteration": iteration,
            **metrics,
        })

    bootstrap_dataframe = (
        pd.DataFrame(records)
    )

    lower_percentile = (
        100.0
        * (1.0 - confidence_level)
        / 2.0
    )
    upper_percentile = (
        100.0 - lower_percentile
    )

    summary_rows = []

    metric_names = [
        "coverage",
        "routing_rate",
        "accepted_accuracy",
        "selective_risk",
        "selective_risk_upper_bound",
        "error_capture_rate",
        "routing_precision",
        "accepted_depression_false_negative_risk",
    ]

    for metric_name in metric_names:
        values = (
            bootstrap_dataframe[
                metric_name
            ]
            .dropna()
            .to_numpy()
        )

        if len(values) == 0:
            continue

        summary_rows.append({
            "metric": metric_name,
            "point_estimate": (
                calculate_selective_metrics(
                    prediction_dataframe,
                    score_column,
                    threshold,
                    RISK_CONFIDENCE_DELTA,
                )[metric_name]
            ),
            "bootstrap_mean": float(
                np.mean(values)
            ),
            "ci_lower": float(
                np.percentile(
                    values,
                    lower_percentile,
                )
            ),
            "ci_upper": float(
                np.percentile(
                    values,
                    upper_percentile,
                )
            ),
            "confidence_level": (
                confidence_level
            ),
        })

    return pd.DataFrame(
        summary_rows
    )


test_bootstrap_ci = (
    bootstrap_fixed_threshold_metrics(
        test_predictions,
        PRIMARY_SCORE_COLUMN,
        SELECTED_THRESHOLD,
        BOOTSTRAP_ITERATIONS,
        BOOTSTRAP_CONFIDENCE_LEVEL,
        SEED,
    )
)

test_bootstrap_ci.to_csv(
    ADVANCED_THRESHOLD_DIR
    / "test_selective_metrics_bootstrap_ci.csv",
    index=False,
)

display(
    test_bootstrap_ci
)


def bootstrap_threshold_stability(
    calibration_dataframe: pd.DataFrame,
    score_column: str,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        seed
    )

    records = []
    sample_count = len(
        calibration_dataframe
    )

    for iteration in range(
        iterations
    ):
        sampled_indices = rng.integers(
            0,
            sample_count,
            size=sample_count,
        )

        sampled = (
            calibration_dataframe.iloc[
                sampled_indices
            ].reset_index(drop=True)
        )

        sampled_sweep = (
            threshold_sweep(
                sampled,
                score_column,
                RISK_CONFIDENCE_DELTA,
            )
        )

        selection = select_threshold(
            sampled_sweep,
            TARGET_SELECTIVE_RISK,
            RISK_CONTROL_METHOD,
            ALLOW_EXPLICIT_RISK_FALLBACK,
        )

        records.append({
            "iteration": iteration,
            "selected_tau": (
                selection[
                    "selected_tau"
                ]
            ),
            "selection_status": (
                selection[
                    "selection_status"
                ]
            ),
            "risk_constraint_satisfied": (
                selection[
                    "risk_constraint_satisfied"
                ]
            ),
        })

    return pd.DataFrame(records)


threshold_stability = (
    bootstrap_threshold_stability(
        calibration_predictions,
        PRIMARY_SCORE_COLUMN,
        THRESHOLD_STABILITY_BOOTSTRAPS,
        SEED,
    )
)

threshold_stability.to_csv(
    ADVANCED_THRESHOLD_DIR
    / "threshold_bootstrap_stability.csv",
    index=False,
)

valid_tau_values = (
    threshold_stability[
        "selected_tau"
    ].dropna()
)

threshold_stability_summary = {
    "bootstrap_iterations": (
        THRESHOLD_STABILITY_BOOTSTRAPS
    ),
    "valid_threshold_count": int(
        valid_tau_values.shape[0]
    ),
    "tau_mean": (
        float(valid_tau_values.mean())
        if len(valid_tau_values)
        else None
    ),
    "tau_std": (
        float(valid_tau_values.std())
        if len(valid_tau_values)
        else None
    ),
    "tau_median": (
        float(valid_tau_values.median())
        if len(valid_tau_values)
        else None
    ),
    "tau_ci_lower": (
        float(
            np.percentile(
                valid_tau_values,
                2.5,
            )
        )
        if len(valid_tau_values)
        else None
    ),
    "tau_ci_upper": (
        float(
            np.percentile(
                valid_tau_values,
                97.5,
            )
        )
        if len(valid_tau_values)
        else None
    ),
    "constraint_satisfaction_rate": (
        float(
            threshold_stability[
                "risk_constraint_satisfied"
            ].mean()
        )
    ),
}

with open(
    ADVANCED_THRESHOLD_DIR
    / "threshold_stability_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        threshold_stability_summary,
        file,
        indent=2,
    )

print(
    json.dumps(
        threshold_stability_summary,
        indent=2,
    )
)


test_with_routing = (
    test_predictions.copy()
)

test_with_routing[
    "selected_threshold"
] = SELECTED_THRESHOLD

test_with_routing[
    "is_routed"
] = (
    test_with_routing[
        PRIMARY_SCORE_COLUMN
    ]
    < SELECTED_THRESHOLD
)

test_with_routing[
    "routing_decision"
] = np.where(
    test_with_routing[
        "is_routed"
    ],
    "Phase2",
    "Phase1_accept",
)

test_with_routing.to_csv(
    ADVANCED_THRESHOLD_DIR
    / "test_predictions_with_routing.csv",
    index=False,
)


high_confidence_errors = (
    test_with_routing[
        (
            ~test_with_routing[
                "phase1_correct"
            ]
        )
        & (
            ~test_with_routing[
                "is_routed"
            ]
        )
    ].copy()
)

high_confidence_errors.to_csv(
    ADVANCED_THRESHOLD_DIR
    / "high_confidence_accepted_errors.csv",
    index=False,
)

print(
    "Accepted high-confidence errors:",
    len(high_confidence_errors),
)

display(
    high_confidence_errors.head(20)
)


# %% [markdown]
# ### Per-class and class-conditional threshold analysis


# %% [cell 40]

def per_class_selective_metrics(
    prediction_dataframe: pd.DataFrame,
    score_column: str,
    threshold: float,
) -> pd.DataFrame:
    accepted = (
        prediction_dataframe[
            score_column
        ] >= threshold
    )
    routed = ~accepted

    rows = []

    for class_id, class_name in (
        ID_TO_CLASS.items()
    ):
        class_mask = (
            prediction_dataframe[
                "label"
            ] == class_id
        )

        accepted_class = (
            accepted & class_mask
        )
        routed_class = (
            routed & class_mask
        )

        accepted_count = int(
            accepted_class.sum()
        )
        routed_count = int(
            routed_class.sum()
        )
        total_count = int(
            class_mask.sum()
        )

        accepted_errors = int(
            (
                accepted_class
                & ~prediction_dataframe[
                    "phase1_correct"
                ]
            ).sum()
        )

        routed_errors = int(
            (
                routed_class
                & ~prediction_dataframe[
                    "phase1_correct"
                ]
            ).sum()
        )

        total_errors = int(
            (
                class_mask
                & ~prediction_dataframe[
                    "phase1_correct"
                ]
            ).sum()
        )

        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "n_total": total_count,
            "n_accepted": accepted_count,
            "n_routed": routed_count,
            "coverage": (
                accepted_count / total_count
                if total_count
                else np.nan
            ),
            "selective_risk": (
                accepted_errors
                / accepted_count
                if accepted_count
                else np.nan
            ),
            "error_capture_rate": (
                routed_errors
                / total_errors
                if total_errors
                else np.nan
            ),
            "routing_precision": (
                routed_errors
                / routed_count
                if routed_count
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


per_class_test_metrics = (
    per_class_selective_metrics(
        test_predictions,
        PRIMARY_SCORE_COLUMN,
        SELECTED_THRESHOLD,
    )
)

per_class_test_metrics.to_csv(
    ADVANCED_THRESHOLD_DIR
    / "per_class_selective_metrics.csv",
    index=False,
)

display(
    per_class_test_metrics
)


def class_conditional_thresholds(
    calibration_dataframe: pd.DataFrame,
    score_column: str,
) -> Dict[int, Dict[str, Any]]:
    result = {}

    for class_id in sorted(
        ID_TO_CLASS
    ):
        class_subset = (
            calibration_dataframe[
                calibration_dataframe[
                    "predicted_label"
                ] == class_id
            ].reset_index(drop=True)
        )

        if len(class_subset) == 0:
            result[class_id] = {
                "selected_tau": (
                    SELECTED_THRESHOLD
                ),
                "selection_status": (
                    "global_threshold_fallback_"
                    "empty_predicted_class"
                ),
            }
            continue

        class_sweep = threshold_sweep(
            class_subset,
            score_column,
            RISK_CONFIDENCE_DELTA,
        )

        selection = select_threshold(
            class_sweep,
            TARGET_SELECTIVE_RISK,
            RISK_CONTROL_METHOD,
            ALLOW_EXPLICIT_RISK_FALLBACK,
        )

        if (
            selection[
                "selected_tau"
            ]
            is None
        ):
            selection[
                "selected_tau"
            ] = SELECTED_THRESHOLD
            selection[
                "selection_status"
            ] = (
                "global_threshold_fallback_"
                "class_infeasible"
            )

        result[class_id] = selection

    return result


if (
    RUN_CLASS_CONDITIONAL_THRESHOLD_ABLATION
):
    class_threshold_results = (
        class_conditional_thresholds(
            calibration_predictions,
            PRIMARY_SCORE_COLUMN,
        )
    )

    class_threshold_values = {
        int(class_id): float(
            details[
                "selected_tau"
            ]
        )
        for class_id, details
        in class_threshold_results.items()
    }

    class_conditional_test = (
        test_predictions.copy()
    )

    applied_thresholds = (
        class_conditional_test[
            "predicted_label"
        ].map(
            class_threshold_values
        )
    )

    class_conditional_test[
        "class_conditional_threshold"
    ] = applied_thresholds

    class_conditional_test[
        "is_routed"
    ] = (
        class_conditional_test[
            PRIMARY_SCORE_COLUMN
        ]
        < applied_thresholds
    )

    class_conditional_test[
        "final_phase1_accepted_correct"
    ] = (
        ~class_conditional_test[
            "is_routed"
        ]
        & class_conditional_test[
            "phase1_correct"
        ]
    )

    class_conditional_summary = {
        "thresholds": (
            class_threshold_values
        ),
        "routing_rate": float(
            class_conditional_test[
                "is_routed"
            ].mean()
        ),
        "routed_phase1_errors": int(
            (
                class_conditional_test[
                    "is_routed"
                ]
                & ~class_conditional_test[
                    "phase1_correct"
                ]
            ).sum()
        ),
        "phase1_errors": int(
            (
                ~class_conditional_test[
                    "phase1_correct"
                ]
            ).sum()
        ),
    }

    with open(
        ADVANCED_THRESHOLD_DIR
        / "class_conditional_thresholds.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "selection_details": (
                    class_threshold_results
                ),
                "test_summary": (
                    class_conditional_summary
                ),
            },
            file,
            indent=2,
            default=str,
        )

    class_conditional_test.to_csv(
        ADVANCED_THRESHOLD_DIR
        / (
            "test_class_conditional_"
            "threshold_ablation.csv"
        ),
        index=False,
    )

    print(
        json.dumps(
            class_conditional_summary,
            indent=2,
        )
    )


# %% [markdown]
# ### Optional cost-sensitive analysis, input-length analysis, and figures


# %% [cell 42]

if RUN_COST_SENSITIVE_ABLATION:
    cost_rows = []

    for _, threshold_row in (
        primary_calibration_sweep.iterrows()
    ):
        threshold = float(
            threshold_row["tau"]
        )

        accepted = (
            calibration_predictions[
                PRIMARY_SCORE_COLUMN
            ] >= threshold
        )

        if accepted.sum() == 0:
            weighted_risk = np.nan
        else:
            true_labels = (
                calibration_predictions.loc[
                    accepted,
                    "label",
                ].to_numpy(dtype=int)
            )
            predicted_labels = (
                calibration_predictions.loc[
                    accepted,
                    "predicted_label",
                ].to_numpy(dtype=int)
            )

            weighted_risk = float(
                np.mean(
                    COST_MATRIX[
                        true_labels,
                        predicted_labels,
                    ]
                )
            )

        cost_rows.append({
            **threshold_row.to_dict(),
            "cost_sensitive_selective_risk": (
                weighted_risk
            ),
        })

    cost_sensitive_table = (
        pd.DataFrame(cost_rows)
    )

    cost_sensitive_table.to_csv(
        ADVANCED_THRESHOLD_DIR
        / "cost_sensitive_threshold_ablation.csv",
        index=False,
    )


def length_group_analysis(
    prediction_dataframe: pd.DataFrame,
    score_column: str,
    threshold: float,
) -> pd.DataFrame:
    dataframe = (
        prediction_dataframe.copy()
    )

    dataframe[
        "length_group"
    ] = pd.cut(
        dataframe[
            "token_length"
        ],
        bins=[
            -np.inf,
            64,
            128,
            MAX_LENGTH,
            np.inf,
        ],
        labels=[
            "0-64",
            "65-128",
            f"129-{MAX_LENGTH}",
            "truncated",
        ],
    )

    rows = []

    for group_name, group in (
        dataframe.groupby(
            "length_group",
            observed=False,
        )
    ):
        if len(group) == 0:
            continue

        metrics = (
            calculate_selective_metrics(
                group.reset_index(
                    drop=True
                ),
                score_column,
                threshold,
                RISK_CONFIDENCE_DELTA,
            )
        )

        rows.append({
            "length_group": str(
                group_name
            ),
            "mean_score": float(
                group[
                    score_column
                ].mean()
            ),
            "accuracy": float(
                group[
                    "phase1_correct"
                ].mean()
            ),
            **metrics,
        })

    return pd.DataFrame(rows)


length_analysis = length_group_analysis(
    test_predictions,
    PRIMARY_SCORE_COLUMN,
    SELECTED_THRESHOLD,
)

length_analysis.to_csv(
    ADVANCED_THRESHOLD_DIR
    / "input_length_threshold_analysis.csv",
    index=False,
)

display(
    length_analysis
)


raw_calibration_probabilities = (
    softmax_fp32(
        calibration_logits,
        1.0,
    )
)

scaled_calibration_probabilities = (
    softmax_fp32(
        calibration_logits,
        SELECTED_TEMPERATURE,
    )
)

raw_reliability = (
    reliability_bins_equal_width(
        raw_calibration_probabilities,
        calibration_labels,
        PRIMARY_ECE_BINS,
    )
)

scaled_reliability = (
    reliability_bins_equal_width(
        scaled_calibration_probabilities,
        calibration_labels,
        PRIMARY_ECE_BINS,
    )
)

plt.figure(figsize=(7, 5))
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration",
)
plt.plot(
    raw_reliability[
        "mean_confidence"
    ],
    raw_reliability["accuracy"],
    marker="o",
    label="Raw MSP",
)
plt.plot(
    scaled_reliability[
        "mean_confidence"
    ],
    scaled_reliability["accuracy"],
    marker="o",
    label="Temperature-scaled MSP",
)
plt.xlabel("Mean confidence")
plt.ylabel("Empirical accuracy")
plt.title(
    "Calibration Reliability Diagram"
)
plt.legend()
plt.tight_layout()
plt.savefig(
    ADVANCED_THRESHOLD_DIR
    / "reliability_diagram.png",
    dpi=200,
)
plt.show()


plt.figure(figsize=(7, 5))
plt.plot(
    primary_calibration_sweep[
        "coverage"
    ],
    primary_calibration_sweep[
        "selective_risk"
    ],
    label="Empirical risk",
)
plt.plot(
    primary_calibration_sweep[
        "coverage"
    ],
    primary_calibration_sweep[
        "selective_risk_upper_bound"
    ],
    label=(
        "One-sided risk upper bound"
    ),
)
plt.xlabel("Coverage")
plt.ylabel("Selective risk")
plt.title(
    "Calibration Risk–Coverage Curve"
)
plt.legend()
plt.tight_layout()
plt.savefig(
    ADVANCED_THRESHOLD_DIR
    / "risk_coverage_curve.png",
    dpi=200,
)
plt.show()


plt.figure(figsize=(7, 5))
plt.plot(
    primary_calibration_sweep[
        "routing_rate"
    ],
    primary_calibration_sweep[
        "error_capture_rate"
    ],
)
plt.xlabel("Routing rate")
plt.ylabel("Error capture rate")
plt.title(
    "Error Capture vs Routing Cost"
)
plt.tight_layout()
plt.savefig(
    ADVANCED_THRESHOLD_DIR
    / "error_capture_vs_routing_rate.png",
    dpi=200,
)
plt.show()


plt.figure(figsize=(7, 5))
plt.hist(
    [
        calibration_predictions.loc[
            calibration_predictions[
                "phase1_correct"
            ],
            PRIMARY_SCORE_COLUMN,
        ],
        calibration_predictions.loc[
            ~calibration_predictions[
                "phase1_correct"
            ],
            PRIMARY_SCORE_COLUMN,
        ],
    ],
    bins=10,
    alpha=0.7,
    label=[
        "Correct",
        "Incorrect",
    ],
)
plt.axvline(
    SELECTED_THRESHOLD,
    linestyle="--",
    label=(
        "Selected threshold = "
        f"{SELECTED_THRESHOLD:.4f}"
    ),
)
plt.xlabel(
    PRIMARY_SCORE_COLUMN
)
plt.ylabel("Calibration samples")
plt.title(
    "Confidence Distribution"
)
plt.legend()
plt.tight_layout()
plt.savefig(
    ADVANCED_THRESHOLD_DIR
    / "confidence_distribution.png",
    dpi=200,
)
plt.show()


# %% [markdown]
# ### Standard held-out classification report


# %% [cell 44]

print(
    classification_report(
        test_predictions["label"],
        test_predictions[
            "predicted_label"
        ],
        target_names=[
            ID_TO_CLASS[index]
            for index in range(3)
        ],
        digits=4,
        zero_division=0,
    )
)

confusion = confusion_matrix(
    test_predictions["label"],
    test_predictions[
        "predicted_label"
    ],
    labels=[0, 1, 2],
)

confusion_dataframe = pd.DataFrame(
    confusion,
    index=[
        f"true_{ID_TO_CLASS[index]}"
        for index in range(3)
    ],
    columns=[
        f"pred_{ID_TO_CLASS[index]}"
        for index in range(3)
    ],
)

display(
    confusion_dataframe
)

confusion_dataframe.to_csv(
    ADVANCED_THRESHOLD_DIR
    / "test_confusion_matrix.csv"
)


# %% [markdown]
# ### Optional Mixed Emotion and real Phase 2 integration


# %% [cell 46]

def prepare_external_dataframe(
    dataframe: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    prepared = dataframe.copy()

    prepared["text"] = (
        prepared[text_column]
        .astype(str)
        .str.strip()
    )
    prepared["label"] = (
        prepared[label_column]
        .map(normalize_label)
    )

    prepared = prepared[
        prepared["label"].notna()
        & (prepared["text"].str.len() > 0)
    ].copy()

    prepared["label"] = (
        prepared["label"].astype(int)
    )
    prepared["label_name"] = (
        prepared["label"].map(
            ID_TO_CLASS
        )
    )
    prepared["sample_id"] = (
        np.arange(len(prepared))
    )

    return prepared.reset_index(
        drop=True
    )


if MIXED_EMOTION_CSV_PATH:
    mixed_raw = pd.read_csv(
        MIXED_EMOTION_CSV_PATH
    )

    mixed_dataframe = (
        prepare_external_dataframe(
            mixed_raw,
            MIXED_EMOTION_TEXT_COLUMN,
            MIXED_EMOTION_LABEL_COLUMN,
        )
    )

    mixed_dataset = Dataset.from_pandas(
        mixed_dataframe[
            ["sample_id", "text", "label"]
        ],
        preserve_index=False,
    )

    mixed_tokenized = mixed_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=[
            column
            for column in [
                "sample_id",
                "text",
            ]
            if column
            in mixed_dataset.column_names
        ],
    )

    mixed_output = trainer.predict(
        mixed_tokenized
    )

    mixed_logits = extract_logits(
        mixed_output
    )

    mixed_predictions = (
        build_advanced_prediction_dataframe(
            mixed_dataframe,
            mixed_logits,
            SELECTED_TEMPERATURE,
        )
    )

    mixed_metrics = (
        calculate_selective_metrics(
            mixed_predictions,
            PRIMARY_SCORE_COLUMN,
            SELECTED_THRESHOLD,
            RISK_CONFIDENCE_DELTA,
        )
    )

    mixed_predictions[
        "is_routed"
    ] = (
        mixed_predictions[
            PRIMARY_SCORE_COLUMN
        ]
        < SELECTED_THRESHOLD
    )

    mixed_predictions.to_csv(
        ADVANCED_THRESHOLD_DIR
        / "mixed_emotion_fixed_threshold_predictions.csv",
        index=False,
    )

    with open(
        ADVANCED_THRESHOLD_DIR
        / "mixed_emotion_fixed_threshold_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            mixed_metrics,
            file,
            indent=2,
        )

    print(
        "Mixed Emotion evaluation used the "
        "Reddit-selected temperature and threshold."
    )
    print(
        json.dumps(
            mixed_metrics,
            indent=2,
        )
    )

    if (
        MIXED_EMOTION_SCENARIO_COLUMN
        in mixed_raw.columns
    ):
        mixed_predictions[
            MIXED_EMOTION_SCENARIO_COLUMN
        ] = mixed_dataframe[
            MIXED_EMOTION_SCENARIO_COLUMN
        ].to_numpy()

        scenario_rows = []

        for (
            scenario_name,
            scenario_dataframe,
        ) in mixed_predictions.groupby(
            MIXED_EMOTION_SCENARIO_COLUMN
        ):
            metrics = (
                calculate_selective_metrics(
                    scenario_dataframe.reset_index(
                        drop=True
                    ),
                    PRIMARY_SCORE_COLUMN,
                    SELECTED_THRESHOLD,
                    RISK_CONFIDENCE_DELTA,
                )
            )
            scenario_rows.append({
                "scenario_type": (
                    scenario_name
                ),
                **metrics,
            })

        pd.DataFrame(
            scenario_rows
        ).to_csv(
            ADVANCED_THRESHOLD_DIR
            / "mixed_emotion_scenario_metrics.csv",
            index=False,
        )


if PHASE2_PREDICTIONS_PATH:
    phase2_predictions = pd.read_csv(
        PHASE2_PREDICTIONS_PATH
    )

    required_columns = {
        "sample_id",
        "phase2_predicted_label",
    }

    missing_columns = (
        required_columns
        - set(
            phase2_predictions.columns
        )
    )

    if missing_columns:
        raise KeyError(
            "Phase 2 result file is missing: "
            f"{sorted(missing_columns)}"
        )

    end_to_end = (
        test_with_routing.merge(
            phase2_predictions[
                [
                    "sample_id",
                    "phase2_predicted_label",
                ]
            ],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
    )

    missing_routed_outputs = (
        end_to_end[
            "is_routed"
        ]
        & end_to_end[
            "phase2_predicted_label"
        ].isna()
    )

    if missing_routed_outputs.any():
        raise ValueError(
            "Phase 2 predictions are missing "
            "for routed test rows."
        )

    end_to_end[
        "final_prediction"
    ] = np.where(
        end_to_end[
            "is_routed"
        ],
        end_to_end[
            "phase2_predicted_label"
        ],
        end_to_end[
            "predicted_label"
        ],
    ).astype(int)

    end_to_end[
        "final_correct"
    ] = (
        end_to_end[
            "final_prediction"
        ]
        == end_to_end["label"]
    )

    end_to_end[
        "corrected_error"
    ] = (
        end_to_end[
            "is_routed"
        ]
        & ~end_to_end[
            "phase1_correct"
        ]
        & end_to_end[
            "final_correct"
        ]
    )

    end_to_end[
        "introduced_error"
    ] = (
        end_to_end[
            "is_routed"
        ]
        & end_to_end[
            "phase1_correct"
        ]
        & ~end_to_end[
            "final_correct"
        ]
    )

    final_summary = {
        "phase1_accuracy": float(
            end_to_end[
                "phase1_correct"
            ].mean()
        ),
        "final_accuracy": float(
            end_to_end[
                "final_correct"
            ].mean()
        ),
        "corrected_errors": int(
            end_to_end[
                "corrected_error"
            ].sum()
        ),
        "introduced_errors": int(
            end_to_end[
                "introduced_error"
            ].sum()
        ),
        "net_corrections": int(
            end_to_end[
                "corrected_error"
            ].sum()
            - end_to_end[
                "introduced_error"
            ].sum()
        ),
    }

    end_to_end.to_csv(
        ADVANCED_THRESHOLD_DIR
        / "test_end_to_end_phase2_predictions.csv",
        index=False,
    )

    with open(
        ADVANCED_THRESHOLD_DIR
        / "phase2_end_to_end_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_summary,
            file,
            indent=2,
        )

    print(
        json.dumps(
            final_summary,
            indent=2,
        )
    )


# %% [markdown]
# ### Reproducibility and threshold provenance


# %% [cell 48]

primary_test_metrics = (
    calculate_selective_metrics(
        test_predictions,
        PRIMARY_SCORE_COLUMN,
        SELECTED_THRESHOLD,
        RISK_CONFIDENCE_DELTA,
    )
)

provenance = {
    "model": str(
        ANALYSIS_MODEL_NAME
    ),
    "model_display_name": "DistilBERT",
    "sample_size_per_class": (
        SAMPLES_PER_CLASS
    ),
    "split_ratios": {
        "train": TRAIN_RATIO,
        "model_validation": (
            VALIDATION_RATIO
        ),
        "threshold_calibration": (
            CALIBRATION_RATIO
        ),
        "held_out_test": (
            TEST_RATIO
        ),
    },
    "confidence_method": (
        PRIMARY_CONFIDENCE_METHOD
    ),
    "score_column": (
        PRIMARY_SCORE_COLUMN
    ),
    "temperature": (
        SELECTED_TEMPERATURE
    ),
    "alpha": (
        TARGET_SELECTIVE_RISK
    ),
    "delta": (
        RISK_CONFIDENCE_DELTA
    ),
    "risk_control_method": (
        RISK_CONTROL_METHOD
    ),
    "candidate_method": (
        "unique_confidence_midpoints"
    ),
    "boundary_rule": (
        "score >= tau is accepted; "
        "score < tau is routed"
    ),
    "tie_breaking": [
        "maximum coverage",
        "minimum controlled risk",
        "minimum empirical risk",
        "lower threshold",
    ],
    "selected_tau": (
        SELECTED_THRESHOLD
    ),
    "selection_status": (
        PRIMARY_THRESHOLD_SELECTION[
            "selection_status"
        ]
    ),
    "risk_constraint_satisfied": (
        PRIMARY_THRESHOLD_SELECTION[
            "risk_constraint_satisfied"
        ]
    ),
    "calibration_n": int(
        len(calibration_predictions)
    ),
    "test_n": int(
        len(test_predictions)
    ),
    "primary_test_metrics": (
        primary_test_metrics
    ),
    "threshold_stability": (
        threshold_stability_summary
    ),
    "random_seed": SEED,
    "max_length": MAX_LENGTH,
}

with open(
    ADVANCED_THRESHOLD_DIR
    / "threshold_provenance.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        provenance,
        file,
        indent=2,
        default=str,
    )

print(
    "Advanced confidence-threshold "
    "analysis completed."
)
print(
    "Results:",
    ADVANCED_THRESHOLD_DIR.resolve(),
)


# %% [markdown]
# ## Notes for paper-quality reruns
