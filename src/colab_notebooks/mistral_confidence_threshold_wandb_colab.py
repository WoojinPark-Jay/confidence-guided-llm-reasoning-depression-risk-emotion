# This file is generated from the matching Colab notebook for code review/reuse.
# The notebook remains the primary runnable artifact.

# %% [markdown]
# # Mistral QLoRA Colab Training Notebook
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
# # Mistral Reddit Classification + W&B QLoRA Tuning + Confidence Threshold Analysis
# 
# This notebook performs an end-to-end smoke test for three-class Reddit emotion
# classification using a Mistral sequence-classification model.
# 
# Execution order:
# 
# 1. Load the Reddit CSV.
# 2. Sample 300 rows from each class:
#    - Depression
#    - Neutral
#    - Happy
# 3. Create stratified train/validation/test splits.
# 4. Tokenize the sampled texts.
# 5. Run or reuse a W&B hyperparameter sweep.
# 6. Train each sweep trial using Mistral sequence classification with LoRA/QLoRA.
# 7. Select the best **finished** W&B run using validation macro F1.
# 8. Create `BEST_HYPERPARAMETERS`.
# 9. Train a fresh final Mistral classifier using the selected configuration.
# 10. Evaluate the held-out test set.
# 11. Calculate maximum-softmax-probability confidence.
# 12. Select a routing threshold on validation data only.
# 13. Apply the fixed threshold to the held-out test set.
# 14. Save the adapter, predictions, tables, metrics, and figures.
# 
# The unrelated `dair-ai/emotion` dataset is not used.
# 
# ## Reference-model choices
# 
# - Base checkpoint: `pblair-basis/Mistral-7B-v0.1`
# - Task: three-class sequence classification
# - LoRA targets: `q_proj` and `v_proj`
# - Scheduler: constant learning rate
# - Warmup ratio: 0.1
# - Maximum gradient norm: 0.3
# 
# On a CUDA GPU, the notebook uses 4-bit QLoRA by default. On a CPU-only
# runtime, it automatically switches to a tiny Mistral sequence-classification
# checkpoint for code-path testing.

# %%

# Colab/Jupyter dependency installation.
#
# This workflow uses bitsandbytes for 4-bit QLoRA and does not use torchao.
# A preinstalled torchao 0.10.0 can be incompatible with recent Transformers,
# so remove it before importing Transformers.

# %pip uninstall -y torchao
# %pip install -q -U     transformers datasets accelerate peft bitsandbytes     scikit-learn pandas matplotlib scipy wandb sentencepiece packaging

# IMPORTANT:
# When this cell changes or removes an already imported package, restart the
# runtime once and then run the notebook from the beginning.

# %%

import os
import gc
import json
import random
import inspect
import time
import warnings
import sys
import subprocess
import importlib
import importlib.metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from packaging.version import Version


def remove_incompatible_torchao() -> None:
    '''
    Remove an installed torchao version that conflicts with Transformers.

    This notebook uses bitsandbytes, not torchao, for 4-bit QLoRA.
    The check runs before importing torch or transformers.
    '''
    try:
        installed_version = (
            importlib.metadata.version(
                "torchao"
            )
        )
    except importlib.metadata.PackageNotFoundError:
        print(
            "torchao is not installed; "
            "no compatibility action is needed."
        )
        return

    parsed_version = Version(
        installed_version
    )

    # The reported Transformers error requires a version above 0.16.0.
    if parsed_version <= Version("0.16.0"):
        print(
            "Removing incompatible torchao "
            f"{installed_version}. "
            "This workflow uses bitsandbytes "
            "instead."
        )

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "torchao",
        ])

        importlib.invalidate_caches()

        print(
            "Incompatible torchao removed. "
            "In a notebook runtime, restart once "
            "if torchao or transformers had already "
            "been imported earlier."
        )

    else:
        print(
            "Installed torchao version is "
            f"{installed_version}; no removal needed."
        )


remove_incompatible_torchao()

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
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
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

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "BF16 supported:",
        torch.cuda.is_bf16_supported(),
    )

# %% [markdown]
# 
# ## 1. Configuration
# 
# The default repository mode creates a new W&B sweep under the account selected at login:
# 
# ```python
# WANDB_PROJECT = "confidence-guided-mistral-colab"
# EXISTING_SWEEP_ID = ""
# ```
# 
# ### One-epoch smoke-test mode
# 
# The default is:
# 
# ```python
# FORCE_SMOKE_TEST_EPOCHS = 1
# WANDB_EPOCH_VALUES = [1]
# ```
# 
# This forces every sweep trial and the final retraining run to use one epoch,
# even when the existing W&B sweep proposes a larger value.
# 
# To restore normal epoch tuning later:
# 
# ```python
# FORCE_SMOKE_TEST_EPOCHS = None
# WANDB_EPOCH_VALUES = [1, 2, 3]
# DEFAULT_HYPERPARAMETERS["epochs"] = 2
# ```
# 
# When continuing an existing sweep, the W&B server may still display its
# original proposed epoch value. The notebook prints and logs both:
# 
# - `sweep_requested_epochs`: value proposed by the existing sweep;
# - `resolved_epochs`: actual value used for training.
# 
# ### torchao compatibility
# 
# This workflow uses `bitsandbytes` for 4-bit QLoRA. It does not require
# `torchao`. The dependency cell removes an incompatible preinstalled
# `torchao<=0.16.0` before Transformers is imported.

# %%

DATA_URL = (
    "https://media.githubusercontent.com/media/"
    "Branden-Kang/LLaMA-2/main/data/final_preprocessed_df2.csv"
)

TEXT_COLUMN = "title_with_selftext_cleaned"
LABEL_COLUMN = "class_group"

TEXT_COLUMN_CANDIDATES = [
    "title_with_selftext_cleaned",
    "Title_with_selftext_cleaned",
    "title_with_selftext",
    "Title_with_selftext",
    "text",
    "Text",
    "cleaned_text",
    "content",
    "selftext",
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

# "first_balanced": fastest smoke test; stops when all classes are filled.
# "reservoir": reads the whole CSV and is less sensitive to file ordering.
SAMPLING_MODE = "first_balanced"
# Use "reservoir" for paper-quality sampling when running the full dataset.
CSV_CHUNK_SIZE = 5_000

TRAIN_RATIO = 0.75
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.10

# With SAMPLES_PER_CLASS=300 this gives approximately:
# train=675, validation=135, test=90.

# 900 rows -> approximately 675 train / 135 validation / 90 test.
# The reference notebook uses MAX_LEN=512. For this 900-row smoke test,
# 256 is the default to reduce GPU memory and runtime. Set it to 512 later if needed.
MAX_LENGTH = 256

# -------------------------------------------------------------------
# Mistral model / QLoRA
# -------------------------------------------------------------------
MISTRAL_MODEL_NAME = "pblair-basis/Mistral-7B-v0.1"
TINY_MISTRAL_MODEL_NAME = (
    "hf-internal-testing/tiny-random-MistralForSequenceClassification"
)

DEBUG_USE_TINY_MODEL = False
AUTO_USE_TINY_ON_CPU = True

# Full 7B smoke test defaults.
USE_4BIT_QLORA = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True

LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

EARLY_STOPPING_PATIENCE = 2
LR_SCHEDULER_TYPE = "constant"
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 0.3

# -------------------------------------------------------------------
# Fast smoke-test epoch control
# -------------------------------------------------------------------
# 1 means every W&B trial and the final retraining run use exactly one epoch,
# even when an existing server-side sweep proposes epochs=2, 3, 5, etc.
#
# For later experiments:
#   FORCE_SMOKE_TEST_EPOCHS = None
# and then expand WANDB_EPOCH_VALUES and DEFAULT_HYPERPARAMETERS["epochs"].
FORCE_SMOKE_TEST_EPOCHS: Optional[int] = 1

if (
    FORCE_SMOKE_TEST_EPOCHS is not None
    and FORCE_SMOKE_TEST_EPOCHS < 1
):
    raise ValueError(
        "FORCE_SMOKE_TEST_EPOCHS must be "
        "None or an integer of at least 1."
    )

# -------------------------------------------------------------------
# W&B
# -------------------------------------------------------------------
# Modes:
#   "new"               Create a new validation-Macro-F1 sweep.
#   "continue_existing" Add trials to the supplied existing sweep.
#   "reuse_best"        Add no trials; reuse the best finished run.
#   "disabled"          Skip W&B and use DEFAULT_HYPERPARAMETERS.
WANDB_SWEEP_MODE = "new"

WANDB_ENTITY = None
WANDB_PROJECT = "confidence-guided-mistral-colab"
EXISTING_SWEEP_ID = ""

# Leave WANDB_ENTITY as None to use the account/team selected at login.
# To continue a collaborator sweep instead, set for example:
# WANDB_ENTITY = "kangsy413"
# WANDB_PROJECT = "my-mistral-sweep"
# EXISTING_SWEEP_ID = "rjo3737f"
# WANDB_SWEEP_MODE = "continue_existing"

WANDB_SWEEP_NAME = "mistral-reddit-qlora-validation-macro-f1"
WANDB_SWEEP_COUNT = 2
WANDB_MODE = "online"

WANDB_OBJECTIVE_METRIC = "validation_f1_macro"
WANDB_OBJECTIVE_GOAL = "maximize"

# An existing sweep is controlled by its server-side metric. When this is
# False, a mismatch prints a warning but still allows the agent to run.
# Final run selection below still prefers validation macro F1.
REQUIRE_EXISTING_SWEEP_METRIC_MATCH = False

WANDB_API_RETRIES = 12
WANDB_API_RETRY_SECONDS = 5
LOG_FINAL_TRAINING_TO_WANDB = True

# New-sweep search space suitable for a 900-row 7B smoke test.
WANDB_BATCH_SIZE_VALUES = [1, 2]
WANDB_GRADIENT_ACCUMULATION_VALUES = [4, 8]
WANDB_EPOCH_VALUES = [1]
WANDB_WEIGHT_DECAY_VALUES = [1e-2, 1e-3, 1e-4]
WANDB_LORA_RANK_VALUES = [4, 8, 16]
WANDB_LORA_ALPHA_VALUES = [8, 16, 32]
WANDB_LORA_DROPOUT_VALUES = [0.05, 0.10]
WANDB_LEARNING_RATE_MIN = 1e-5
WANDB_LEARNING_RATE_MAX = 2e-4

DEFAULT_HYPERPARAMETERS = {
    "learning_rate": 2e-5,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "epochs": 1,
    "weight_decay": 0.001,
    "lora_rank": 4,
    "lora_alpha": 16,
    "lora_dropout": 0.10,
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

# Exact metric first; legacy aliases are used only when no exact Macro-F1
# run is available in an older sweep.
RUN_SELECTION_METRIC_ALIASES = [
    "validation_f1_macro",
    "eval/f1_macro",
    "eval_f1_macro",
    "eval/f1",
    "eval_f1",
    "eval/f1-score",
    "eval_f1-score",
    "f1",
    "f1-score",
]

# -------------------------------------------------------------------
# Confidence threshold analysis
# -------------------------------------------------------------------
TARGET_SELECTIVE_RISK = 0.05
MIN_ACCEPTED_SAMPLES = 10
REPORT_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]

OUTPUT_DIR = Path(
    "./mistral_reddit_wandb_qlora_threshold_outputs"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SWEEP_RESULTS_PATH = (
    OUTPUT_DIR / "wandb_sweep_results.jsonl"
)

print("Output directory:", OUTPUT_DIR.resolve())
print("W&B mode:", WANDB_SWEEP_MODE)
print(
    "W&B entity/project:",
    f"{WANDB_ENTITY or 'default-account'}/{WANDB_PROJECT}",
)

if USE_EXISTING_SWEEP:
    print("Existing sweep ID:", EXISTING_SWEEP_ID)

if (
    not torch.cuda.is_available()
    and AUTO_USE_TINY_ON_CPU
):
    DEBUG_USE_TINY_MODEL = True

ACTIVE_MODEL_NAME = (
    TINY_MISTRAL_MODEL_NAME
    if DEBUG_USE_TINY_MODEL
    else MISTRAL_MODEL_NAME
)

ACTIVE_USE_4BIT = (
    USE_4BIT_QLORA
    and torch.cuda.is_available()
    and not DEBUG_USE_TINY_MODEL
)

print("Model:", ACTIVE_MODEL_NAME)
print("Tiny debug model:", DEBUG_USE_TINY_MODEL)
print("4-bit QLoRA:", ACTIVE_USE_4BIT)
print(
    "Forced smoke-test epochs:",
    FORCE_SMOKE_TEST_EPOCHS,
)

# %% [markdown]
# ### W&B mode behavior
# 
# `continue_existing` connects to:
# 
# ```text
# kangsy413/my-mistral-sweep/rjo3737f
# ```
# 
# The parameter space and optimization objective of an existing sweep remain
# controlled by the W&B server. The local configuration does not overwrite them.
# 
# The final best-run selector:
# 
# 1. excludes interrupted, failed, crashed, killed, and running runs;
# 2. first looks for `validation_f1_macro`;
# 3. falls back to legacy F1 keys only when no exact Macro-F1 result exists.
# 
# When the existing sweep is loss-based, use `WANDB_SWEEP_MODE="new"` for a
# strict Macro-F1 Bayesian sweep.

# %% [markdown]
# ## 2. Load and sample the Reddit dataset

# %%

CANONICAL_CLASS_TO_ID = {
    "Depression": 0,
    "Neutral": 1,
    "Happy": 2,
}

ID_TO_CLASS = {
    value: key
    for key, value in CANONICAL_CLASS_TO_ID.items()
}


def detect_column(
    columns: Iterable[str],
    preferred: Optional[str],
    candidates: List[str],
) -> str:
    columns = list(columns)

    if preferred is not None:
        if preferred not in columns:
            raise KeyError(
                f"Configured column {preferred!r} was not found. "
                f"Available columns: {columns}"
            )
        return preferred

    for candidate in candidates:
        if candidate in columns:
            return candidate

    lower_to_original = {
        str(column).lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[
                candidate.lower()
            ]

    raise KeyError(
        "Could not automatically detect a required column. "
        f"Available columns: {columns}"
    )


def normalize_label(value) -> Optional[int]:
    if pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        integer = int(value)
        return integer if integer in ID_TO_CLASS else None

    if (
        isinstance(value, (float, np.floating))
        and float(value).is_integer()
    ):
        integer = int(value)
        return integer if integer in ID_TO_CLASS else None

    normalized = (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if normalized in {
        "0",
        "depression",
        "depressed",
        "depression_group",
    }:
        return 0

    if normalized in {
        "1",
        "neutral",
        "neutral_group",
    }:
        return 1

    if normalized in {
        "2",
        "happy",
        "happiness",
        "happy_group",
        "positive",
    }:
        return 2

    if "depress" in normalized:
        return 0
    if "neutral" in normalized:
        return 1
    if "happy" in normalized:
        return 2

    return None


preview_df = pd.read_csv(
    DATA_URL,
    nrows=20,
    low_memory=False,
)

detected_text_column = detect_column(
    preview_df.columns,
    TEXT_COLUMN,
    TEXT_COLUMN_CANDIDATES,
)

detected_label_column = detect_column(
    preview_df.columns,
    LABEL_COLUMN,
    LABEL_COLUMN_CANDIDATES,
)

print("Detected text column :", detected_text_column)
print("Detected label column:", detected_label_column)
print("Available columns     :", list(preview_df.columns))

print("\nRaw label examples:")
print(
    preview_df[
        detected_label_column
    ].value_counts(
        dropna=False
    ).head(10)
)

# %%

def sample_balanced_from_csv(
    csv_url: str,
    text_column: str,
    label_column: str,
    samples_per_class: int,
    chunksize: int,
    mode: str,
    seed: int,
) -> pd.DataFrame:
    if mode not in {
        "first_balanced",
        "reservoir",
    }:
        raise ValueError(
            "mode must be 'first_balanced' or 'reservoir'."
        )

    rng = random.Random(seed)
    target_ids = [0, 1, 2]

    reservoirs: Dict[int, List[dict]] = {
        class_id: []
        for class_id in target_ids
    }

    seen_counts: Dict[int, int] = {
        class_id: 0
        for class_id in target_ids
    }

    for chunk_index, chunk in enumerate(
        pd.read_csv(
            csv_url,
            usecols=[
                text_column,
                label_column,
            ],
            chunksize=chunksize,
            low_memory=False,
        ),
        start=1,
    ):
        chunk = chunk.dropna(
            subset=[
                text_column,
                label_column,
            ]
        ).copy()

        chunk["label"] = chunk[
            label_column
        ].map(normalize_label)

        chunk = chunk[
            chunk["label"].isin(target_ids)
        ].copy()

        chunk["text"] = (
            chunk[text_column]
            .astype(str)
            .str.strip()
        )

        chunk = chunk[
            chunk["text"].str.len() > 0
        ]

        if mode == "first_balanced":
            chunk = chunk.sample(
                frac=1.0,
                random_state=seed + chunk_index,
            )

            for class_id in target_ids:
                remaining = (
                    samples_per_class
                    - len(reservoirs[class_id])
                )

                if remaining <= 0:
                    continue

                candidates = chunk[
                    chunk["label"] == class_id
                ][["text", "label"]]

                reservoirs[class_id].extend(
                    candidates
                    .head(remaining)
                    .to_dict("records")
                )

            if all(
                len(reservoirs[class_id])
                >= samples_per_class
                for class_id in target_ids
            ):
                print(
                    "Sampling completed after chunk",
                    chunk_index,
                )
                break

        else:
            for row in chunk[
                ["text", "label"]
            ].to_dict("records"):
                class_id = int(row["label"])
                seen_counts[class_id] += 1

                if (
                    len(reservoirs[class_id])
                    < samples_per_class
                ):
                    reservoirs[class_id].append(row)
                else:
                    replacement_index = rng.randint(
                        0,
                        seen_counts[class_id] - 1,
                    )

                    if (
                        replacement_index
                        < samples_per_class
                    ):
                        reservoirs[class_id][
                            replacement_index
                        ] = row

        if chunk_index % 10 == 0:
            print(
                "Processed chunks:",
                chunk_index,
                {
                    ID_TO_CLASS[key]: len(value)
                    for key, value
                    in reservoirs.items()
                },
            )

    rows = []

    for class_id in target_ids:
        class_rows = reservoirs[class_id]

        if len(class_rows) < samples_per_class:
            raise ValueError(
                f"Only {len(class_rows)} rows were collected for "
                f"{ID_TO_CLASS[class_id]}; "
                f"required {samples_per_class}."
            )

        rows.extend(
            class_rows[:samples_per_class]
        )

    sampled = pd.DataFrame(rows)
    sampled["label"] = (
        sampled["label"].astype(int)
    )
    sampled["label_name"] = (
        sampled["label"].map(ID_TO_CLASS)
    )

    sampled = sampled.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    sampled.insert(
        0,
        "sample_id",
        np.arange(len(sampled)),
    )

    return sampled


sampled_df = sample_balanced_from_csv(
    csv_url=DATA_URL,
    text_column=detected_text_column,
    label_column=detected_label_column,
    samples_per_class=SAMPLES_PER_CLASS,
    chunksize=CSV_CHUNK_SIZE,
    mode=SAMPLING_MODE,
    seed=SEED,
)

print("\nBalanced sample shape:", sampled_df.shape)
print(
    sampled_df[
        "label_name"
    ].value_counts()
)

display(sampled_df.head())

sampled_df.to_csv(
    OUTPUT_DIR
    / "balanced_reddit_sample_900.csv",
    index=False,
)

# %% [markdown]
# ## 3. Stratified train/validation/test split

# %%

def stratified_three_way_split(
    dataframe: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(
        train_ratio
        + validation_ratio
        + test_ratio,
        1.0,
    ):
        raise ValueError(
            "Train, validation, and test ratios must sum to 1."
        )

    train_df, temporary_df = train_test_split(
        dataframe,
        test_size=1.0 - train_ratio,
        random_state=seed,
        stratify=dataframe["label"],
    )

    relative_test_ratio = (
        test_ratio
        / (validation_ratio + test_ratio)
    )

    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temporary_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


train_df, validation_df, test_df = (
    stratified_three_way_split(
        sampled_df,
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )
)

for split_name, split_df in {
    "train": train_df,
    "validation": validation_df,
    "test": test_df,
}.items():
    print(f"\n{split_name}: {len(split_df)}")
    print(
        split_df[
            "label_name"
        ].value_counts().sort_index()
    )

train_df.to_csv(
    OUTPUT_DIR / "train_sample.csv",
    index=False,
)
validation_df.to_csv(
    OUTPUT_DIR / "validation_sample.csv",
    index=False,
)
test_df.to_csv(
    OUTPUT_DIR / "test_sample.csv",
    index=False,
)

# %% [markdown]
# ## 4. Tokenizer and Hugging Face datasets

# %%

tokenizer = AutoTokenizer.from_pretrained(
    ACTIVE_MODEL_NAME,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = (
        tokenizer.eos_token
        or tokenizer.unk_token
    )

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = (
        tokenizer.convert_tokens_to_ids(
            tokenizer.pad_token
        )
    )

tokenizer.padding_side = "right"

raw_datasets = DatasetDict({
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


tokenized_datasets = raw_datasets.map(
    tokenize_batch,
    batched=True,
    remove_columns=[
        "sample_id",
        "text",
    ],
    desc="Tokenizing Reddit sample",
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8
    if torch.cuda.is_available()
    else None,
)

print(tokenized_datasets)
print("Pad token:", tokenizer.pad_token)
print("Pad token ID:", tokenizer.pad_token_id)

# %% [markdown]
# ## 5. Mistral QLoRA model and Trainer helpers

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


def get_compute_dtype() -> torch.dtype:
    if (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16

    if torch.cuda.is_available():
        return torch.float16

    return torch.float32


def find_classification_head_name(
    model,
) -> Optional[str]:
    for candidate in [
        "score",
        "classifier",
        "classification_head",
    ]:
        if hasattr(model, candidate):
            return candidate

    return None


def create_mistral_peft_model(
    hyperparameters: Dict[str, Any],
):
    compute_dtype = get_compute_dtype()

    model_kwargs = {
        "pretrained_model_name_or_path": (
            ACTIVE_MODEL_NAME
        ),
        "num_labels": 3,
        "id2label": id2label,
        "label2id": label2id,
        "trust_remote_code": True,
        "ignore_mismatched_sizes": True,
        "low_cpu_mem_usage": True,
    }

    if ACTIVE_USE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=(
                BNB_4BIT_QUANT_TYPE
            ),
            bnb_4bit_use_double_quant=(
                BNB_4BIT_USE_DOUBLE_QUANT
            ),
            bnb_4bit_compute_dtype=(
                compute_dtype
            ),
        )

        model_kwargs.update({
            "quantization_config": (
                quantization_config
            ),
            "device_map": "auto",
            "torch_dtype": compute_dtype,
        })

    else:
        model_kwargs["torch_dtype"] = (
            compute_dtype
            if torch.cuda.is_available()
            else torch.float32
        )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(**model_kwargs)
    )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )
    model.config.problem_type = (
        "single_label_classification"
    )
    model.config.use_cache = False

    if hasattr(
        model.config,
        "pretraining_tp",
    ):
        model.config.pretraining_tp = 1

    if ACTIVE_USE_4BIT:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    elif (
        torch.cuda.is_available()
        and not DEBUG_USE_TINY_MODEL
    ):
        model.gradient_checkpointing_enable()

    head_name = find_classification_head_name(
        model
    )

    lora_kwargs = {
        "task_type": TaskType.SEQ_CLS,
        "r": int(
            hyperparameters["lora_rank"]
        ),
        "lora_alpha": int(
            hyperparameters["lora_alpha"]
        ),
        "lora_dropout": float(
            hyperparameters["lora_dropout"]
        ),
        "bias": "none",
        "target_modules": (
            LORA_TARGET_MODULES
        ),
    }

    if head_name is not None:
        lora_kwargs["modules_to_save"] = [
            head_name
        ]

    peft_config = LoraConfig(
        **lora_kwargs
    )

    model = get_peft_model(
        model,
        peft_config,
    )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )
    model.config.use_cache = False

    print(
        "LoRA configuration:",
        {
            "rank": (
                hyperparameters[
                    "lora_rank"
                ]
            ),
            "alpha": (
                hyperparameters[
                    "lora_alpha"
                ]
            ),
            "dropout": (
                hyperparameters[
                    "lora_dropout"
                ]
            ),
            "target_modules": (
                LORA_TARGET_MODULES
            ),
            "classification_head": (
                head_name
            ),
            "4bit": ACTIVE_USE_4BIT,
            "compute_dtype": str(
                compute_dtype
            ),
        },
    )

    model.print_trainable_parameters()

    return model


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(
        logits,
        axis=-1,
    )

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
    if SaveStrategy is not None:
        values = {
            member.value
            for member in SaveStrategy
        }

        if "best" in values:
            return "best"

    return "epoch"


class ManualWandbMetricsCallback(
    TrainerCallback
):
    def __init__(self, wandb_run=None):
        self.wandb_run = wandb_run

    def run_is_active(self) -> bool:
        if self.wandb_run is None:
            return False

        return not bool(
            getattr(
                self.wandb_run,
                "_is_finished",
                False,
            )
        )

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
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            ):
                result[key] = float(value)

        return result

    def on_log(
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs,
    ):
        if not self.run_is_active():
            return

        scalar_logs = self.scalar_metrics(
            logs
        )

        if scalar_logs:
            self.wandb_run.log(
                {
                    f"trainer/{key}": value
                    for key, value
                    in scalar_logs.items()
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
        if not self.run_is_active():
            return

        metrics = self.scalar_metrics(
            metrics
        )

        mapping = {
            "eval_loss": "validation_loss",
            "eval_accuracy": (
                "validation_accuracy"
            ),
            "eval_precision_macro": (
                "validation_precision_macro"
            ),
            "eval_recall_macro": (
                "validation_recall_macro"
            ),
            "eval_f1_macro": (
                "validation_f1_macro"
            ),
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
    hyperparameters: Dict[str, Any],
    run_name: Optional[str] = None,
) -> TrainingArguments:
    save_strategy = resolve_save_strategy()

    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and not DEBUG_USE_TINY_MODEL
    )

    use_fp16 = (
        torch.cuda.is_available()
        and not use_bf16
        and not DEBUG_USE_TINY_MODEL
    )

    kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": float(
            hyperparameters[
                "learning_rate"
            ]
        ),
        "per_device_train_batch_size": int(
            hyperparameters[
                "batch_size"
            ]
        ),
        "per_device_eval_batch_size": int(
            hyperparameters[
                "batch_size"
            ]
        ),
        "gradient_accumulation_steps": int(
            hyperparameters[
                "gradient_accumulation_steps"
            ]
        ),
        "num_train_epochs": int(
            hyperparameters["epochs"]
        ),
        "weight_decay": float(
            hyperparameters[
                "weight_decay"
            ]
        ),
        "lr_scheduler_type": (
            LR_SCHEDULER_TYPE
        ),
        "warmup_ratio": WARMUP_RATIO,
        "max_grad_norm": MAX_GRAD_NORM,
        "logging_strategy": "epoch",
        "save_strategy": save_strategy,
        "load_best_model_at_end": True,
        "metric_for_best_model": (
            "f1_macro"
        ),
        "greater_is_better": True,
        "save_total_limit": 1,
        "report_to": [],
        "run_name": run_name,
        "seed": SEED,
        "data_seed": SEED,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gradient_checkpointing": (
            not DEBUG_USE_TINY_MODEL
        ),
        "remove_unused_columns": True,
        "push_to_hub": False,
        "optim": (
            "paged_adamw_8bit"
            if ACTIVE_USE_4BIT
            else "adamw_torch"
        ),
    }

    signature = inspect.signature(
        TrainingArguments.__init__
    )

    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs[
            "evaluation_strategy"
        ] = "epoch"

    if (
        "gradient_checkpointing_kwargs"
        in signature.parameters
        and not DEBUG_USE_TINY_MODEL
    ):
        kwargs[
            "gradient_checkpointing_kwargs"
        ] = {
            "use_reentrant": False
        }

    training_arguments = (
        TrainingArguments(**kwargs)
    )

    effective_batch_size = (
        int(
            hyperparameters[
                "batch_size"
            ]
        )
        * int(
            hyperparameters[
                "gradient_accumulation_steps"
            ]
        )
    )

    print(
        "Training arguments:",
        {
            "learning_rate": (
                training_arguments
                .learning_rate
            ),
            "micro_batch_size": (
                training_arguments
                .per_device_train_batch_size
            ),
            "gradient_accumulation_steps": (
                training_arguments
                .gradient_accumulation_steps
            ),
            "effective_batch_size": (
                effective_batch_size
            ),
            "epochs": (
                training_arguments
                .num_train_epochs
            ),
            "weight_decay": (
                training_arguments
                .weight_decay
            ),
            "optimizer": (
                training_arguments.optim
            ),
            "save_strategy": str(
                training_arguments
                .save_strategy
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
            tokenized_datasets[
                "validation"
            ]
        ),
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "callbacks": callbacks,
    }

    signature = inspect.signature(
        Trainer.__init__
    )

    if "processing_class" in signature.parameters:
        kwargs[
            "processing_class"
        ] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer

    return Trainer(**kwargs)


def release_training_objects(
    *objects,
) -> None:
    for obj in objects:
        del obj

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %% [markdown]
# ## 6. W&B sweep setup

# %%

def build_sweep_path(
    entity: str,
    project: str,
    sweep_id: str,
) -> str:
    clean_id = str(
        sweep_id
    ).strip().strip("/")

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
) -> Tuple[
    Optional[str],
    Optional[str],
]:
    sweep_config = dict(
        sweep.config or {}
    )

    metric_config = (
        sweep_config.get("metric")
        or {}
    )

    if not isinstance(
        metric_config,
        dict,
    ):
        return None, None

    metric_name = (
        metric_config.get("name")
    )
    metric_goal = (
        metric_config.get("goal")
    )

    if metric_goal is not None:
        metric_goal = str(
            metric_goal
        ).lower()

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
        metric_name
        == WANDB_OBJECTIVE_METRIC
        and metric_goal
        == WANDB_OBJECTIVE_GOAL
    )

    if matches:
        print(
            "Existing sweep objective "
            "matches validation macro F1."
        )
        return

    message = (
        "Existing sweep objective differs "
        "from this notebook's preferred objective. "
        f"Existing metric={metric_name!r}, "
        f"goal={metric_goal!r}; "
        f"preferred metric="
        f"{WANDB_OBJECTIVE_METRIC!r}, "
        f"goal={WANDB_OBJECTIVE_GOAL!r}. "
        "The existing W&B server objective "
        "will still control parameter proposals. "
        "Final completed runs will be ranked "
        "by validation F1 when possible."
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
            "name": (
                WANDB_OBJECTIVE_METRIC
            ),
            "goal": (
                WANDB_OBJECTIVE_GOAL
            ),
        },
        "parameters": {
            "batch_size": {
                "values": (
                    WANDB_BATCH_SIZE_VALUES
                ),
            },
            "gradient_accumulation_steps": {
                "values": (
                    WANDB_GRADIENT_ACCUMULATION_VALUES
                ),
            },
            "epochs": {
                "values": (
                    WANDB_EPOCH_VALUES
                ),
            },
            "weight_decay": {
                "values": (
                    WANDB_WEIGHT_DECAY_VALUES
                ),
            },
            "lora_rank": {
                "values": (
                    WANDB_LORA_RANK_VALUES
                ),
            },
            "lora_alpha": {
                "values": (
                    WANDB_LORA_ALPHA_VALUES
                ),
            },
            "lora_dropout": {
                "values": (
                    WANDB_LORA_DROPOUT_VALUES
                ),
            },
            "learning_rate": {
                "distribution": (
                    "log_uniform_values"
                ),
                "min": (
                    WANDB_LEARNING_RATE_MIN
                ),
                "max": (
                    WANDB_LEARNING_RATE_MAX
                ),
            },
        },
    }

    print("New Mistral QLoRA sweep:")
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
        WANDB_ENTITY,
        WANDB_PROJECT,
        EXISTING_SWEEP_ID,
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
        "W&B disabled. "
        "Fixed defaults will be used."
    )

# %% [markdown]
# ## 7. Run W&B QLoRA trials

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


def get_config_value(
    config,
    canonical_name: str,
    aliases: Optional[List[str]] = None,
    default: Any = None,
    required: bool = True,
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

    if not required:
        return default

    raise KeyError(
        f"Missing W&B parameter "
        f"{canonical_name!r}. "
        f"Checked names: "
        f"{[canonical_name, *aliases]}. "
        f"Available keys: "
        f"{sorted(config_dict.keys())}"
    )


def resolve_trial_hyperparameters(
    config,
) -> Dict[str, Any]:
    requested_epochs = int(
        get_config_value(
            config,
            "epochs",
            aliases=[
                "num_train_epochs",
            ],
            default=(
                DEFAULT_HYPERPARAMETERS[
                    "epochs"
                ]
            ),
            required=False,
        )
    )

    effective_epochs = (
        int(FORCE_SMOKE_TEST_EPOCHS)
        if FORCE_SMOKE_TEST_EPOCHS
        is not None
        else requested_epochs
    )

    return {
        "learning_rate": float(
            get_config_value(
                config,
                "learning_rate",
                aliases=["lr"],
            )
        ),
        "batch_size": int(
            get_config_value(
                config,
                "batch_size",
                aliases=[
                    "per_device_train_batch_size",
                    "train_batch_size",
                ],
            )
        ),
        "gradient_accumulation_steps": int(
            get_config_value(
                config,
                "gradient_accumulation_steps",
                aliases=[
                    "grad_accumulation",
                    "gradient_accumulation",
                ],
                default=(
                    DEFAULT_HYPERPARAMETERS[
                        "gradient_accumulation_steps"
                    ]
                ),
                required=False,
            )
        ),
        "epochs": effective_epochs,
        "sweep_requested_epochs": (
            requested_epochs
        ),
        "weight_decay": float(
            get_config_value(
                config,
                "weight_decay",
                default=(
                    DEFAULT_HYPERPARAMETERS[
                        "weight_decay"
                    ]
                ),
                required=False,
            )
        ),
        "lora_rank": int(
            get_config_value(
                config,
                "lora_rank",
                aliases=[
                    "lora_r",
                    "rank",
                    "r",
                ],
                default=(
                    DEFAULT_HYPERPARAMETERS[
                        "lora_rank"
                    ]
                ),
                required=False,
            )
        ),
        "lora_alpha": int(
            get_config_value(
                config,
                "lora_alpha",
                aliases=["alpha"],
                default=(
                    DEFAULT_HYPERPARAMETERS[
                        "lora_alpha"
                    ]
                ),
                required=False,
            )
        ),
        "lora_dropout": float(
            get_config_value(
                config,
                "lora_dropout",
                aliases=["dropout"],
                default=(
                    DEFAULT_HYPERPARAMETERS[
                        "lora_dropout"
                    ]
                ),
                required=False,
            )
        ),
    }


def run_wandb_trial() -> None:
    import wandb

    with wandb.init() as run:
        hyperparameters = (
            resolve_trial_hyperparameters(
                run.config
            )
        )

        # Add normalized values without overwriting
        # server-locked sweep keys.
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
                "resolved_gradient_accumulation_steps": (
                    hyperparameters[
                        "gradient_accumulation_steps"
                    ]
                ),
                "sweep_requested_epochs": (
                    hyperparameters[
                        "sweep_requested_epochs"
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
                "resolved_lora_rank": (
                    hyperparameters[
                        "lora_rank"
                    ]
                ),
                "resolved_lora_alpha": (
                    hyperparameters[
                        "lora_alpha"
                    ]
                ),
                "resolved_lora_dropout": (
                    hyperparameters[
                        "lora_dropout"
                    ]
                ),
                "model_name": (
                    ACTIVE_MODEL_NAME
                ),
                "samples_per_class": (
                    SAMPLES_PER_CLASS
                ),
                "max_length": MAX_LENGTH,
                "four_bit_qlora": (
                    ACTIVE_USE_4BIT
                ),
                "seed": SEED,
            },
            allow_val_change=True,
        )

        print("Resolved trial hyperparameters:")
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

        model = create_mistral_peft_model(
            hyperparameters
        )

        training_args = (
            create_training_arguments(
                output_dir=trial_output_dir,
                hyperparameters=(
                    hyperparameters
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

            # The best validation-Macro-F1
            # checkpoint is restored here.
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

            completion_metrics = {
                "validation_loss": eval_loss,
                "validation_f1_macro": (
                    eval_f1
                ),
                "validation_accuracy": (
                    eval_accuracy
                ),
                "trial_completed": 1,
            }

            run.log(completion_metrics)
            run.summary.update(
                completion_metrics
            )

            record = {
                "wandb_run_id": run.id,
                "wandb_run_name": (
                    run.name
                ),
                **hyperparameters,
                "eval_loss": eval_loss,
                "eval_f1_macro": eval_f1,
                "eval_accuracy": (
                    eval_accuracy
                ),
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

            print("Completed W&B trial:")
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
    count: int,
) -> None:
    import wandb

    try:
        wandb.agent(
            sweep_path,
            function=run_wandb_trial,
            count=count,
        )

    except KeyboardInterrupt:
        print(
            "Sweep interrupted manually. "
            "Only finished runs will be "
            "eligible in the next step."
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
        "Continuing existing sweep:",
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
        "No new W&B trials will run. "
        "Best finished run will be loaded from:",
        ACTIVE_SWEEP_PATH,
    )

else:
    ACTIVE_SWEEP_PATH = None

# %% [markdown]
# 
# ## 8. Select the best finished W&B run
# 
# `BEST_HYPERPARAMETERS` is created in this section and is not referenced by final
# model training until the following section.

# %%

def first_available_metric(
    run,
    metric_names: List[str],
) -> Tuple[
    Optional[str],
    Optional[float],
]:
    if str(run.state).lower() != "finished":
        return None, None

    for metric_name in metric_names:
        value = run.summary.get(
            metric_name
        )

        if value is None:
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue

        if np.isfinite(numeric_value):
            return (
                metric_name,
                numeric_value,
            )

    return None, None


def normalized_value(
    config: Dict[str, Any],
    names: List[str],
    default: Any = None,
    required: bool = True,
):
    for name in names:
        if (
            name in config
            and config[name] is not None
        ):
            return config[name]

    if not required:
        return default

    raise KeyError(
        f"Best W&B run is missing "
        f"{names[0]!r}. "
        f"Checked aliases: {names}. "
        f"Available keys: "
        f"{sorted(config.keys())}"
    )


def normalize_best_hyperparameters(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "learning_rate": float(
            normalized_value(
                config,
                [
                    "learning_rate",
                    "lr",
                    "resolved_learning_rate",
                ],
            )
        ),
        "batch_size": int(
            normalized_value(
                config,
                [
                    "batch_size",
                    "per_device_train_batch_size",
                    "train_batch_size",
                    "resolved_batch_size",
                ],
            )
        ),
        "gradient_accumulation_steps": int(
            normalized_value(
                config,
                [
                    "gradient_accumulation_steps",
                    "grad_accumulation",
                    "gradient_accumulation",
                    "resolved_gradient_accumulation_steps",
                ],
                default=(
                    DEFAULT_HYPERPARAMETERS[
                        "gradient_accumulation_steps"
                    ]
                ),
                required=False,
            )
        ),
        "epochs": (
            int(FORCE_SMOKE_TEST_EPOCHS)
            if FORCE_SMOKE_TEST_EPOCHS
            is not None
            else int(
                normalized_value(
                    config,
                    [
                        "epochs",
                        "num_train_epochs",
                        "resolved_epochs",
                    ],
                    default=(
                        DEFAULT_HYPERPARAMETERS[
                            "epochs"
                        ]
                    ),
                    required=False,
                )
            )
        ),
        "weight_decay": float(
            normalized_value(
                config,
                [
                    "weight_decay",
                    "resolved_weight_decay",
                ],
                default=(
                    DEFAULT_HYPERPARAMETERS[
                        "weight_decay"
                    ]
                ),
                required=False,
            )
        ),
        "lora_rank": int(
            normalized_value(
                config,
                [
                    "lora_rank",
                    "lora_r",
                    "rank",
                    "r",
                    "resolved_lora_rank",
                ],
                default=(
                    DEFAULT_HYPERPARAMETERS[
                        "lora_rank"
                    ]
                ),
                required=False,
            )
        ),
        "lora_alpha": int(
            normalized_value(
                config,
                [
                    "lora_alpha",
                    "alpha",
                    "resolved_lora_alpha",
                ],
                default=(
                    DEFAULT_HYPERPARAMETERS[
                        "lora_alpha"
                    ]
                ),
                required=False,
            )
        ),
        "lora_dropout": float(
            normalized_value(
                config,
                [
                    "lora_dropout",
                    "dropout",
                    "resolved_lora_dropout",
                ],
                default=(
                    DEFAULT_HYPERPARAMETERS[
                        "lora_dropout"
                    ]
                ),
                required=False,
            )
        ),
    }


def rank_finished_runs(
    sweep,
) -> Tuple[
    Any,
    str,
    float,
    pd.DataFrame,
]:
    records = []
    exact_candidates = []
    fallback_candidates = []

    for run in sweep.runs:
        metric_name, metric_value = (
            first_available_metric(
                run,
                RUN_SELECTION_METRIC_ALIASES,
            )
        )

        record = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "run_url": run.url,
        }

        records.append(record)

        if metric_value is None:
            continue

        candidate = (
            run,
            metric_name,
            metric_value,
        )

        if (
            metric_name
            == WANDB_OBJECTIVE_METRIC
        ):
            exact_candidates.append(
                candidate
            )
        else:
            fallback_candidates.append(
                candidate
            )

    runs_df = pd.DataFrame(records)

    candidates = (
        exact_candidates
        if exact_candidates
        else fallback_candidates
    )

    if not candidates:
        raise ValueError(
            "No finished W&B run contains "
            "a usable F1 metric. "
            "Run at least one complete trial "
            "or set WANDB_SWEEP_MODE='disabled'."
        )

    candidates.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    return (
        candidates[0][0],
        candidates[0][1],
        candidates[0][2],
        runs_df,
    )


def load_best_finished_run(
    sweep_path: str,
):
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
                metric_name,
                metric_value,
                runs_df,
            ) = rank_finished_runs(
                sweep
            )

            best_hyperparameters = (
                normalize_best_hyperparameters(
                    dict(best_run.config)
                )
            )

            metadata = {
                "sweep_path": sweep_path,
                "sweep_name": sweep.name,
                "sweep_state": sweep.state,
                "sweep_url": sweep.url,
                "best_run_id": best_run.id,
                "best_run_name": (
                    best_run.name
                ),
                "best_run_url": best_run.url,
                "selection_metric_used": (
                    metric_name
                ),
                "selection_metric_value": (
                    metric_value
                ),
                "best_run_config": dict(
                    best_run.config
                ),
                "best_run_summary": dict(
                    best_run.summary
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
        "Mistral run with an F1 metric."
    ) from last_error


if WANDB_SWEEP_MODE in {
    "new",
    "continue_existing",
    "reuse_best",
}:
    (
        BEST_HYPERPARAMETERS,
        BEST_WANDB_RUN_METADATA,
        WANDB_RUN_CANDIDATES,
    ) = load_best_finished_run(
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


required_keys = {
    "learning_rate",
    "batch_size",
    "gradient_accumulation_steps",
    "epochs",
    "weight_decay",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
}

missing_keys = (
    required_keys
    - set(BEST_HYPERPARAMETERS)
)

if missing_keys:
    raise KeyError(
        "BEST_HYPERPARAMETERS is missing: "
        f"{sorted(missing_keys)}"
    )

print("BEST_HYPERPARAMETERS loaded:")
print(
    json.dumps(
        BEST_HYPERPARAMETERS,
        indent=2,
    )
)

if BEST_WANDB_RUN_METADATA:
    print(
        "Metric used:",
        BEST_WANDB_RUN_METADATA[
            "selection_metric_used"
        ],
    )
    print(
        "Metric value:",
        BEST_WANDB_RUN_METADATA[
            "selection_metric_value"
        ],
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
# ## 9. Train the final Mistral classifier
# 
# A fresh base model is loaded and adapted using the selected QLoRA
# hyperparameters. The final W&B run remains active through held-out test
# evaluation and is closed immediately afterward.

# %%

if "BEST_HYPERPARAMETERS" not in globals():
    raise RuntimeError(
        "BEST_HYPERPARAMETERS is not defined. "
        "Run the W&B trial and best-run "
        "selection sections first."
    )

if (
    WANDB_SWEEP_MODE != "disabled"
    and "wandb" in globals()
    and hasattr(wandb, "teardown")
):
    # Clean agent state before starting a normal final run.
    wandb.teardown()


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
            "final-mistral-training-"
            "best-f1-config"
        ),
        job_type="final-training",
        config={
            **BEST_HYPERPARAMETERS,
            "model_name": ACTIVE_MODEL_NAME,
            "samples_per_class": (
                SAMPLES_PER_CLASS
            ),
            "max_length": MAX_LENGTH,
            "four_bit_qlora": (
                ACTIVE_USE_4BIT
            ),
            "selection_metric": (
                WANDB_OBJECTIVE_METRIC
            ),
        },
        reinit=True,
    )


final_model = create_mistral_peft_model(
    BEST_HYPERPARAMETERS
)

final_training_args = (
    create_training_arguments(
        output_dir=(
            OUTPUT_DIR
            / "final_model_training"
        ),
        hyperparameters=(
            BEST_HYPERPARAMETERS
        ),
        run_name=(
            "final-mistral-training-"
            "best-f1-config"
        ),
    )
)

trainer = create_trainer(
    final_model,
    final_training_args,
    wandb_run=final_run,
)

train_result = trainer.train()

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
    validation_payload = {
        "final_validation_loss": float(
            validation_metrics[
                "eval_loss"
            ]
        ),
        "final_validation_f1_macro": float(
            validation_metrics[
                "eval_f1_macro"
            ]
        ),
        "final_validation_accuracy": float(
            validation_metrics[
                "eval_accuracy"
            ]
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

    final_run.log(
        validation_payload
    )
    final_run.summary.update(
        validation_payload
    )

# Save the PEFT adapter and classification head.
trainer.save_model(
    OUTPUT_DIR
    / "best_mistral_qlora_adapter"
)
tokenizer.save_pretrained(
    OUTPUT_DIR
    / "best_mistral_qlora_adapter"
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
# ## 10. Held-out test evaluation

# %%

test_metrics = trainer.evaluate(
    eval_dataset=(
        tokenized_datasets["test"]
    ),
    metric_key_prefix="test",
)

print("Held-out test metrics:")
print(
    json.dumps(
        test_metrics,
        indent=2,
    )
)

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


if final_run is not None:
    numeric_test_metrics = {
        key: float(value)
        for key, value in test_metrics.items()
        if isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        )
    }

    # The callback already logs test-prefixed metrics.
    # Update the run summary and then close the run.
    final_run.summary.update(
        numeric_test_metrics
    )

    trainer.remove_callback(
        ManualWandbMetricsCallback
    )

    final_run.finish()
    final_run = None

    if hasattr(wandb, "teardown"):
        wandb.teardown()

print(
    "Held-out test evaluation completed. "
    "The final W&B run is closed."
)

# %% [markdown]
# 
# ## 11. MSP confidence prediction tables
# 
# For each row:
# 
# - predicted class: class with the largest softmax probability;
# - confidence: maximum softmax probability;
# - Phase 1 correctness: whether the Mistral classifier prediction matches the label.

# %%

def build_prediction_dataframe(
    trainer: Trainer,
    tokenized_split: Dataset,
    original_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    prediction_output = trainer.predict(
        tokenized_split
    )

    logits = (
        prediction_output.predictions
    )

    if isinstance(logits, tuple):
        logits = logits[0]

    probabilities = softmax(
        logits,
        axis=1,
    )

    predicted_labels = np.argmax(
        probabilities,
        axis=1,
    )

    confidence = np.max(
        probabilities,
        axis=1,
    )

    result = original_dataframe[
        [
            "sample_id",
            "text",
            "label",
            "label_name",
        ]
    ].copy()

    result["predicted_label"] = (
        predicted_labels.astype(int)
    )
    result["predicted_label_name"] = (
        result[
            "predicted_label"
        ].map(ID_TO_CLASS)
    )
    result["confidence"] = confidence
    result["prob_depression"] = (
        probabilities[:, 0]
    )
    result["prob_neutral"] = (
        probabilities[:, 1]
    )
    result["prob_happy"] = (
        probabilities[:, 2]
    )
    result["phase1_correct"] = (
        result["predicted_label"]
        == result["label"]
    )

    return result


validation_predictions = (
    build_prediction_dataframe(
        trainer,
        tokenized_datasets[
            "validation"
        ],
        validation_df,
    )
)

test_predictions = (
    build_prediction_dataframe(
        trainer,
        tokenized_datasets["test"],
        test_df,
    )
)

validation_predictions.to_csv(
    OUTPUT_DIR
    / "validation_predictions.csv",
    index=False,
)

test_predictions.to_csv(
    OUTPUT_DIR
    / "test_predictions.csv",
    index=False,
)

print(
    "Validation accuracy:",
    validation_predictions[
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
    validation_predictions.head()
)

# %% [markdown]
# ## 12. Validation-only risk–coverage threshold selection

# %%

def calculate_threshold_metrics(
    predictions_dataframe: pd.DataFrame,
    threshold: float,
) -> Dict[str, float]:
    accepted = (
        predictions_dataframe[
            "confidence"
        ]
        >= threshold
    )
    routed = ~accepted

    errors = (
        ~predictions_dataframe[
            "phase1_correct"
        ]
    )

    total_count = len(
        predictions_dataframe
    )
    accepted_count = int(
        accepted.sum()
    )
    routed_count = int(
        routed.sum()
    )
    error_count = int(
        errors.sum()
    )

    routed_error_count = int(
        (routed & errors).sum()
    )

    coverage = (
        accepted_count / total_count
        if total_count
        else np.nan
    )

    routing_rate = (
        routed_count / total_count
        if total_count
        else np.nan
    )

    if accepted_count > 0:
        accepted_accuracy = float(
            predictions_dataframe.loc[
                accepted,
                "phase1_correct",
            ].mean()
        )
        selective_risk = (
            1.0 - accepted_accuracy
        )
    else:
        accepted_accuracy = np.nan
        selective_risk = np.nan

    error_capture_rate = (
        routed_error_count
        / error_count
        if error_count > 0
        else np.nan
    )

    return {
        "tau": float(threshold),
        "n_total": total_count,
        "n_accepted": accepted_count,
        "n_routed": routed_count,
        "coverage": coverage,
        "routing_rate": routing_rate,
        "accepted_accuracy": (
            accepted_accuracy
        ),
        "selective_risk": (
            selective_risk
        ),
        "phase1_errors": error_count,
        "routed_phase1_errors": (
            routed_error_count
        ),
        "error_capture_rate": (
            error_capture_rate
        ),
    }


def threshold_sweep(
    predictions_dataframe: pd.DataFrame,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    return pd.DataFrame([
        calculate_threshold_metrics(
            predictions_dataframe,
            float(threshold),
        )
        for threshold in thresholds
    ])


def build_exact_threshold_candidates(
    predictions_dataframe: pd.DataFrame,
) -> np.ndarray:
    confidence_values = (
        predictions_dataframe[
            "confidence"
        ]
        .astype(float)
        .to_numpy()
    )

    grid_values = np.arange(
        0.50,
        1.00,
        0.01,
    )

    candidates = np.unique(
        np.concatenate([
            confidence_values,
            grid_values,
            np.asarray(
                REPORT_THRESHOLDS,
                dtype=float,
            ),
        ])
    )

    candidates = candidates[
        (candidates >= 0.0)
        & (candidates <= 1.0)
    ]

    return np.sort(candidates)


validation_threshold_candidates = (
    build_exact_threshold_candidates(
        validation_predictions
    )
)

validation_sweep = threshold_sweep(
    validation_predictions,
    validation_threshold_candidates,
)

display(validation_sweep.head())

# %%

def select_threshold_by_risk_coverage(
    sweep_dataframe: pd.DataFrame,
    target_selective_risk: float,
    minimum_accepted_samples: int,
) -> Tuple[
    float,
    str,
    pd.Series,
]:
    eligible = sweep_dataframe[
        sweep_dataframe[
            "selective_risk"
        ].notna()
        & (
            sweep_dataframe[
                "selective_risk"
            ]
            <= target_selective_risk
        )
        & (
            sweep_dataframe[
                "n_accepted"
            ]
            >= minimum_accepted_samples
        )
    ].copy()

    if not eligible.empty:
        selected_row = (
            eligible.sort_values(
                by=[
                    "coverage",
                    "tau",
                ],
                ascending=[
                    False,
                    True,
                ],
            ).iloc[0]
        )

        status = (
            "risk_constraint_satisfied"
        )

    else:
        valid_rows = sweep_dataframe[
            sweep_dataframe[
                "selective_risk"
            ].notna()
            & (
                sweep_dataframe[
                    "n_accepted"
                ]
                >= minimum_accepted_samples
            )
        ].copy()

        if valid_rows.empty:
            raise ValueError(
                "No threshold has enough "
                "accepted validation samples."
            )

        selected_row = (
            valid_rows.sort_values(
                by=[
                    "selective_risk",
                    "coverage",
                    "tau",
                ],
                ascending=[
                    True,
                    False,
                    True,
                ],
            ).iloc[0]
        )

        status = (
            "fallback_minimum_observed_risk"
        )

        print(
            "WARNING: No threshold satisfied "
            "the configured selective-risk "
            "constraint. The lowest-risk "
            "validation threshold was used "
            "as an explicit fallback."
        )

    return (
        float(selected_row["tau"]),
        status,
        selected_row,
    )


(
    selected_tau,
    threshold_selection_status,
    selected_validation_threshold_row,
) = select_threshold_by_risk_coverage(
    validation_sweep,
    target_selective_risk=(
        TARGET_SELECTIVE_RISK
    ),
    minimum_accepted_samples=(
        MIN_ACCEPTED_SAMPLES
    ),
)

print("Selected threshold:", selected_tau)
print(
    "Selection status:",
    threshold_selection_status,
)

print("\nSelected validation metrics:")
display(
    selected_validation_threshold_row
    .to_frame("value")
)

threshold_metadata = {
    "selected_tau": selected_tau,
    "selection_status": (
        threshold_selection_status
    ),
    "target_selective_risk": (
        TARGET_SELECTIVE_RISK
    ),
    "minimum_accepted_samples": (
        MIN_ACCEPTED_SAMPLES
    ),
    "selected_validation_metrics": (
        selected_validation_threshold_row
        .to_dict()
    ),
}

with open(
    OUTPUT_DIR
    / "selected_threshold.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        threshold_metadata,
        file,
        indent=2,
        default=str,
    )

# %% [markdown]
# ## 13. Apply the fixed validation-selected threshold to test data

# %%

def apply_routing(
    predictions_dataframe: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    result = (
        predictions_dataframe.copy()
    )

    result["selected_tau"] = (
        threshold
    )

    result["is_routed"] = (
        result["confidence"]
        < threshold
    )

    result["routing_decision"] = (
        np.where(
            result["is_routed"],
            "Phase2",
            "Phase1_accept",
        )
    )

    return result


test_routed = apply_routing(
    test_predictions,
    selected_tau,
)

test_routing_metrics = (
    calculate_threshold_metrics(
        test_predictions,
        selected_tau,
    )
)

print(
    "Fixed threshold applied to "
    "held-out test set:",
    selected_tau,
)

print(
    json.dumps(
        test_routing_metrics,
        indent=2,
    )
)

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
    ].sort_values(
        "confidence"
    )
)

test_routed.to_csv(
    OUTPUT_DIR
    / "test_predictions_with_routing.csv",
    index=False,
)

with open(
    OUTPUT_DIR
    / "test_routing_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        test_routing_metrics,
        file,
        indent=2,
    )

# %% [markdown]
# ## 14. Threshold sensitivity tables and figures

# %%

validation_report_table = (
    threshold_sweep(
        validation_predictions,
        REPORT_THRESHOLDS,
    )
)

test_report_table = (
    threshold_sweep(
        test_predictions,
        REPORT_THRESHOLDS,
    )
)

validation_report_table.to_csv(
    OUTPUT_DIR
    / "validation_threshold_sensitivity.csv",
    index=False,
)

test_report_table.to_csv(
    OUTPUT_DIR
    / "test_threshold_sensitivity.csv",
    index=False,
)

print("Validation threshold sensitivity")
display(validation_report_table)

print("Held-out test threshold sensitivity")
display(test_report_table)

# %%

plot_dataframe = (
    validation_sweep.dropna(
        subset=["selective_risk"]
    ).copy()
)

plt.figure(figsize=(7, 5))
plt.plot(
    plot_dataframe["coverage"],
    plot_dataframe["selective_risk"],
)

selected_plot_rows = plot_dataframe[
    np.isclose(
        plot_dataframe["tau"],
        selected_tau,
    )
]

if not selected_plot_rows.empty:
    selected_plot_row = (
        selected_plot_rows.iloc[0]
    )

    plt.scatter(
        [
            selected_plot_row[
                "coverage"
            ]
        ],
        [
            selected_plot_row[
                "selective_risk"
            ]
        ],
        s=100,
        label=(
            f"selected tau="
            f"{selected_tau:.4f}"
        ),
    )
    plt.legend()

plt.xlabel("Coverage")
plt.ylabel("Selective risk")
plt.title(
    "Mistral Validation "
    "Risk–Coverage Curve"
)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "validation_risk_coverage_curve.png",
    dpi=200,
)

plt.show()

# %%

correct_confidence = (
    validation_predictions.loc[
        validation_predictions[
            "phase1_correct"
        ],
        "confidence",
    ]
)

incorrect_confidence = (
    validation_predictions.loc[
        ~validation_predictions[
            "phase1_correct"
        ],
        "confidence",
    ]
)

plt.figure(figsize=(7, 5))

plt.hist(
    [
        correct_confidence,
        incorrect_confidence,
    ],
    bins=10,
    alpha=0.7,
    label=[
        "Correct",
        "Incorrect",
    ],
)

plt.axvline(
    selected_tau,
    linestyle="--",
    label=(
        f"selected tau="
        f"{selected_tau:.4f}"
    ),
)

plt.xlabel(
    "Maximum softmax probability"
)
plt.ylabel(
    "Number of validation samples"
)
plt.title(
    "Mistral Validation "
    "Confidence Distribution"
)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "validation_confidence_distribution.png",
    dpi=200,
)

plt.show()

# %% [markdown]
# ## 15. Classification report and confusion matrix

# %%

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

display(confusion_dataframe)

confusion_dataframe.to_csv(
    OUTPUT_DIR
    / "test_confusion_matrix.csv"
)

# %% [markdown]
# 
# ## 16. Optional Phase 2 integration
# 
# This notebook evaluates the Mistral classifier and its routing decisions. A true
# two-phase final accuracy requires Phase 2 predictions for rows whose
# `is_routed=True`.
# 
# Create a CSV with:
# 
# - `sample_id`
# - `phase2_predicted_label`
# 
# Then uncomment the next cell.

# %%

# PHASE2_PREDICTIONS_PATH = (
#     "./phase2_predictions.csv"
# )
#
# phase2_predictions = pd.read_csv(
#     PHASE2_PREDICTIONS_PATH
# )
#
# required_columns = {
#     "sample_id",
#     "phase2_predicted_label",
# }
#
# missing_columns = (
#     required_columns
#     - set(phase2_predictions.columns)
# )
#
# if missing_columns:
#     raise KeyError(
#         f"Missing columns: "
#         f"{sorted(missing_columns)}"
#     )
#
# end_to_end_predictions = (
#     test_routed.merge(
#         phase2_predictions[
#             [
#                 "sample_id",
#                 "phase2_predicted_label",
#             ]
#         ],
#         on="sample_id",
#         how="left",
#         validate="one_to_one",
#     )
# )
#
# missing_phase2 = (
#     end_to_end_predictions[
#         "is_routed"
#     ]
#     & end_to_end_predictions[
#         "phase2_predicted_label"
#     ].isna()
# )
#
# if missing_phase2.any():
#     raise ValueError(
#         "Phase 2 predictions are missing "
#         "for routed rows."
#     )
#
# end_to_end_predictions[
#     "final_prediction"
# ] = np.where(
#     end_to_end_predictions[
#         "is_routed"
#     ],
#     end_to_end_predictions[
#         "phase2_predicted_label"
#     ],
#     end_to_end_predictions[
#         "predicted_label"
#     ],
# )
#
# end_to_end_predictions[
#     "final_correct"
# ] = (
#     end_to_end_predictions[
#         "final_prediction"
#     ]
#     == end_to_end_predictions["label"]
# )
#
# print(
#     "Mistral classifier accuracy:",
#     end_to_end_predictions[
#         "phase1_correct"
#     ].mean(),
# )
#
# print(
#     "Final two-phase accuracy:",
#     end_to_end_predictions[
#         "final_correct"
#     ].mean(),
# )
#
# end_to_end_predictions.to_csv(
#     OUTPUT_DIR
#     / "test_end_to_end_predictions.csv",
#     index=False,
# )

# %% [markdown]
# 
# ## 17. Interpretation and limitations
# 
# - Each W&B trial is a separate Mistral fine-tuning run.
# - `FORCE_SMOKE_TEST_EPOCHS=1` makes every trial and final retraining run use one epoch.
# - To tune epochs later, set `FORCE_SMOKE_TEST_EPOCHS=None` and expand `WANDB_EPOCH_VALUES`.
# - An existing sweep can display a larger requested epoch value, but `resolved_epochs` is the actual value used.
# - The 900-row experiment is a smoke test, not the final paper experiment.
# - The held-out test set is never used for hyperparameter selection.
# - The confidence threshold is selected on validation data and fixed before test evaluation.
# - MSP is a confidence proxy, not a calibrated uncertainty estimate.
# - The QLoRA implementation uses `bitsandbytes`; `torchao` is not required.
# - Only finished W&B runs are eligible for best-run selection.
# - The final saved model is a PEFT/LoRA adapter plus the sequence-classification head.
# - The default `MAX_LENGTH=256` reduces runtime; set it to 512 for a closer reference reproduction.

