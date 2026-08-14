#!/usr/bin/env python3
"""Audit accepted Phase 1 errors on the 12,000-row Reddit held-out protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONFIDENCE_BINS = [0.70, 0.80, 0.90, 0.95, 0.98, 1.0000001]
CONFIDENCE_LABELS = ["0.70-0.80", "0.80-0.90", "0.90-0.95", "0.95-0.98", "0.98-1.00"]

CASE_AUDIT = {
    "RED_041439": {
        "error_type": "acute-distress cue missed",
        "publication_excerpt": (
            "suicidal again ... the past two weeks were going pretty okay"
        ),
        "audit_interpretation": (
            "The present-state title explicitly expresses acute distress, but a short statement "
            "about the preceding two weeks appears to dominate the Happy prediction."
        ),
    },
    "RED_000879": {
        "error_type": "severe negative-state cue missed",
        "publication_excerpt": (
            "blackmailed ... alcoholic headed to prison ... loss of custody"
        ),
        "audit_interpretation": (
            "Coercion, alcohol dependence, imprisonment, family separation, and custody loss "
            "provide no textual support for the Happy prediction."
        ),
    },
    "RED_026909": {
        "error_type": "technical context suppresses an affective cue",
        "publication_excerpt": (
            "learning Java ... never expected to grasp it quickly ... kinda proud"
        ),
        "audit_interpretation": (
            "The classifier gives more weight to the technical programming context than to the "
            "terminal accomplishment cue supporting the Happy proxy label."
        ),
    },
    "RED_103680": {
        "error_type": "product details suppress explicit excitement",
        "publication_excerpt": (
            "saved money on an EEPROM programmer ... new project ... so excited"
        ),
        "audit_interpretation": (
            "Pricing and hardware details dominate the representation despite explicit "
            "excitement and a positive evaluation of the new project."
        ),
    },
    "RED_010855": {
        "error_type": "mental-health topic-term shortcut",
        "publication_excerpt": (
            "can a computer diagnose a mental health disorder ... online test diagnose"
        ),
        "audit_interpretation": (
            "The post is informational, but repeated disorder and diagnosis terms trigger a "
            "high-confidence Depression prediction."
        ),
    },
    "RED_000438": {
        "error_type": "affect inferred from a factual question",
        "publication_excerpt": (
            "high-pitched tone coming from tea ... what is going on ... is it safe"
        ),
        "audit_interpretation": (
            "The post asks for a physical explanation and safety information and contains no "
            "explicit positive affect supporting the Happy prediction."
        ),
    },
}


def balanced_reddit_view(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "example_id",
        "text",
        "label_str",
        "phase1_label",
        "phase1_confidence",
        "phase1_raw_confidence",
        "phase1_accepted",
        "phase1_routed",
        "routing_threshold",
        "temperature",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    return (
        frame.sort_values(["label_str", "example_id"])
        .groupby("label_str", sort=False, group_keys=False)
        .head(4_000)
        .reset_index(drop=True)
    )


def transition_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["label_str", "phase1_label"], as_index=False)
        .agg(
            error_count=("example_id", "size"),
            mean_confidence=("phase1_confidence", "mean"),
            median_confidence=("phase1_confidence", "median"),
            median_word_count=("text_word_count", "median"),
        )
        .rename(columns={"label_str": "reference_proxy_label"})
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-predictions", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "reports/high_confidence_accepted_error_audit",
    )
    args = parser.parse_args()

    frame = balanced_reddit_view(pd.read_csv(args.phase1_predictions))
    frame["is_correct"] = frame["label_str"].eq(frame["phase1_label"])
    frame["text_word_count"] = frame["text"].fillna("").str.split().str.len()
    frame["threshold_margin"] = frame["phase1_confidence"] - frame["routing_threshold"]

    accepted = frame[frame["phase1_accepted"].astype(bool)].copy()
    routed = frame[frame["phase1_routed"].astype(bool)].copy()
    accepted_errors = accepted[~accepted["is_correct"]].copy()
    routed_errors = routed[~routed["is_correct"]].copy()

    depression_accepted = accepted[accepted["label_str"].eq("Depression")]
    depression_false_negatives = accepted_errors[
        accepted_errors["label_str"].eq("Depression")
        & ~accepted_errors["phase1_label"].eq("Depression")
    ]

    summary = {
        "held_out_rows": int(len(frame)),
        "routing_threshold": float(frame["routing_threshold"].iloc[0]),
        "temperature": float(frame["temperature"].iloc[0]),
        "accepted_count": int(len(accepted)),
        "routed_count": int(len(routed)),
        "total_phase1_errors": int((~frame["is_correct"]).sum()),
        "accepted_errors": int(len(accepted_errors)),
        "accepted_selective_risk": float((~accepted["is_correct"]).mean()),
        "routed_errors": int(len(routed_errors)),
        "error_capture_rate": float(len(routed_errors) / (~frame["is_correct"]).sum()),
        "routing_precision": float((~routed["is_correct"]).mean()),
        "accepted_depression_count": int(len(depression_accepted)),
        "accepted_depression_false_negatives": int(len(depression_false_negatives)),
        "accepted_depression_false_negative_risk": float(
            len(depression_false_negatives) / len(depression_accepted)
        ),
        "accepted_errors_at_or_above_0_98": int(
            (accepted_errors["phase1_confidence"] >= 0.98).sum()
        ),
        "depression_false_negatives_at_or_above_0_98": int(
            (depression_false_negatives["phase1_confidence"] >= 0.98).sum()
        ),
    }

    by_class = (
        accepted.groupby("label_str", as_index=False)
        .agg(
            accepted_count=("example_id", "size"),
            accepted_errors=("is_correct", lambda x: int((~x).sum())),
            accepted_risk=("is_correct", lambda x: float((~x).mean())),
        )
        .rename(columns={"label_str": "reference_proxy_label"})
    )

    accepted["confidence_band"] = pd.cut(
        accepted["phase1_confidence"],
        bins=CONFIDENCE_BINS,
        labels=CONFIDENCE_LABELS,
        right=False,
    )
    confidence_bands = (
        accepted.groupby("confidence_band", observed=False, as_index=False)
        .agg(
            accepted_count=("example_id", "size"),
            accepted_errors=("is_correct", lambda x: int((~x).sum())),
            accepted_error_rate=("is_correct", lambda x: float((~x).mean())),
        )
    )

    comparison = pd.DataFrame(
        [
            {
                "subset": "accepted",
                "count": len(accepted),
                "errors": (~accepted["is_correct"]).sum(),
                "accuracy": accepted["is_correct"].mean(),
                "median_confidence": accepted["phase1_confidence"].median(),
                "median_word_count": accepted["text_word_count"].median(),
            },
            {
                "subset": "routed",
                "count": len(routed),
                "errors": (~routed["is_correct"]).sum(),
                "accuracy": routed["is_correct"].mean(),
                "median_confidence": routed["phase1_confidence"].median(),
                "median_word_count": routed["text_word_count"].median(),
            },
        ]
    )

    case_rows = []
    indexed = accepted_errors.set_index("example_id")
    for example_id, audit in CASE_AUDIT.items():
        if example_id not in indexed.index:
            raise ValueError(f"Representative case not found among accepted errors: {example_id}")
        row = indexed.loc[example_id]
        case_rows.append(
            {
                "example_id": example_id,
                "reference_proxy_label": row["label_str"],
                "phase1_label": row["phase1_label"],
                "calibrated_confidence": row["phase1_confidence"],
                "routing_threshold": row["routing_threshold"],
                "threshold_margin": row["threshold_margin"],
                "text_word_count": row["text_word_count"],
                **audit,
            }
        )
    cases = pd.DataFrame(case_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(args.output_dir / "accepted_error_summary.csv", index=False)
    by_class.to_csv(args.output_dir / "accepted_error_by_class.csv", index=False)
    transition_table(accepted_errors).to_csv(
        args.output_dir / "accepted_error_transitions.csv", index=False
    )
    confidence_bands.to_csv(
        args.output_dir / "accepted_error_confidence_bands.csv", index=False
    )
    comparison.to_csv(
        args.output_dir / "accepted_vs_routed_error_comparison.csv", index=False
    )
    cases.to_csv(args.output_dir / "representative_accepted_error_cases.csv", index=False)
    (args.output_dir / "accepted_error_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(pd.DataFrame([summary]).to_string(index=False))
    print("\nAccepted error transitions:\n", transition_table(accepted_errors).to_string(index=False))
    print("\nConfidence bands:\n", confidence_bands.to_string(index=False))
    print("\nRepresentative cases:\n", cases.to_string(index=False))


if __name__ == "__main__":
    main()
