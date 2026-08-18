#!/usr/bin/env python3
"""Reproduce paired end-to-end tests reported in the manuscript.

The analysis compares correctness for the same example before and after Phase 2.
It reports exact two-sided McNemar tests, paired bootstrap confidence intervals
for the accuracy change, and Holm-adjusted p-values across four comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260813
BOOTSTRAP_REPLICATES = 50_000


def exact_mcnemar_p(introduced: int, corrected: int) -> float:
    """Return the two-sided exact binomial p-value for discordant pairs."""
    discordant = introduced + corrected
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(introduced, corrected) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_bootstrap_ci(
    phase1_correct: np.ndarray,
    end_to_end_correct: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Return a percentile CI for paired accuracy change in percentage points."""
    introduced = int(np.sum(phase1_correct & ~end_to_end_correct))
    corrected = int(np.sum(~phase1_correct & end_to_end_correct))
    sample_size = len(phase1_correct)
    unchanged = sample_size - introduced - corrected
    draws = rng.multinomial(
        sample_size,
        [introduced / sample_size, unchanged / sample_size, corrected / sample_size],
        size=BOOTSTRAP_REPLICATES,
    )
    changes = (draws[:, 2] - draws[:, 0]) / sample_size * 100.0
    lower, upper = np.percentile(changes, [2.5, 97.5])
    return float(lower), float(upper)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, p_values[index] * (total - rank)))
        adjusted[index] = running_max
    return adjusted


def balanced_reddit_view(frame: pd.DataFrame) -> pd.DataFrame:
    """Match the manuscript protocol of exactly 4,000 held-out rows per class."""
    required = {"label_str", "example_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Reddit results are missing columns: {sorted(missing)}")
    return (
        frame.sort_values(["label_str", "example_id"])
        .groupby("label_str", sort=False, group_keys=False)
        .head(4_000)
        .reset_index(drop=True)
    )


def build_pairs(args: argparse.Namespace):
    reddit = balanced_reddit_view(pd.read_csv(args.reddit_end_to_end))
    required_reddit = {
        "is_correct_phase1",
        "is_correct_llama2_e2e",
        "is_correct_llama3_e2e",
    }
    missing_reddit = required_reddit - set(reddit.columns)
    if missing_reddit:
        raise ValueError(f"Reddit results are missing columns: {sorted(missing_reddit)}")

    mixed_phase1 = pd.read_csv(args.mixed_phase1)
    mixed_phase2 = pd.read_csv(args.mixed_phase2)
    required_phase1 = {"example_id", "target_label", "phase1_label"}
    required_phase2 = {"example_id", "LLaMA2_final_label", "LLaMA3_final_label"}
    if missing := required_phase1 - set(mixed_phase1.columns):
        raise ValueError(f"Mixed Phase 1 results are missing columns: {sorted(missing)}")
    if missing := required_phase2 - set(mixed_phase2.columns):
        raise ValueError(f"Mixed Phase 2 results are missing columns: {sorted(missing)}")

    mixed = mixed_phase1.merge(
        mixed_phase2[list(required_phase2)],
        on="example_id",
        how="left",
        validate="one_to_one",
    )
    for column in ("LLaMA2_final_label", "LLaMA3_final_label"):
        mixed[column] = mixed[column].fillna(mixed["phase1_label"])

    return [
        (
            "Reddit held-out",
            "Llama 2 CoT",
            reddit["is_correct_phase1"].to_numpy(dtype=bool),
            reddit["is_correct_llama2_e2e"].to_numpy(dtype=bool),
        ),
        (
            "Reddit held-out",
            "Llama 3 SELF-DISCOVER",
            reddit["is_correct_phase1"].to_numpy(dtype=bool),
            reddit["is_correct_llama3_e2e"].to_numpy(dtype=bool),
        ),
        (
            "Mixed Emotion",
            "Llama 2 CoT",
            (mixed["target_label"] == mixed["phase1_label"]).to_numpy(),
            (mixed["target_label"] == mixed["LLaMA2_final_label"]).to_numpy(),
        ),
        (
            "Mixed Emotion",
            "Llama 3 SELF-DISCOVER",
            (mixed["target_label"] == mixed["phase1_label"]).to_numpy(),
            (mixed["target_label"] == mixed["LLaMA3_final_label"]).to_numpy(),
        ),
    ]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--reddit-end-to-end", type=Path, required=True)
    parser.add_argument("--mixed-phase1", type=Path, required=True)
    parser.add_argument("--mixed-phase2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=root / "reports/statistics")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    rows = []
    for dataset, reasoner, phase1, end_to_end in build_pairs(args):
        introduced = int(np.sum(phase1 & ~end_to_end))
        corrected = int(np.sum(~phase1 & end_to_end))
        lower, upper = paired_bootstrap_ci(phase1, end_to_end, rng)
        rows.append(
            {
                "dataset": dataset,
                "reasoner": reasoner,
                "n": len(phase1),
                "phase1_accuracy": float(np.mean(phase1)),
                "end_to_end_accuracy": float(np.mean(end_to_end)),
                "accuracy_change_pp": float(
                    (np.mean(end_to_end) - np.mean(phase1)) * 100.0
                ),
                "corrected": corrected,
                "introduced": introduced,
                "net_corrections": corrected - introduced,
                "bootstrap_95_ci_lower_pp": lower,
                "bootstrap_95_ci_upper_pp": upper,
                "mcnemar_exact_p": exact_mcnemar_p(introduced, corrected),
            }
        )

    results = pd.DataFrame(rows)
    results["holm_adjusted_p"] = holm_adjust(results["mcnemar_exact_p"].tolist())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "paired_end_to_end_statistics.csv", index=False)
    (args.output_dir / "paired_end_to_end_statistics.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "multiple_testing": (
                    "Holm adjustment across four prespecified paired comparisons"
                ),
                "results": results.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
