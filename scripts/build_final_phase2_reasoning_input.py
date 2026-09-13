#!/usr/bin/env python3
"""Attach original Reddit title/body text to a frozen Phase 1 prediction export."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE_URL = (
    "https://media.githubusercontent.com/media/"
    "WoojinPark-Jay/confidence-guided-llm-reasoning-depression-risk-emotion/"
    "refs/heads/main/data/final_preprocessed_whole_df.csv"
)
LABELS = ("Depression", "Neutral", "Happy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-predictions", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--expected-total", type=int, default=12_000)
    parser.add_argument("--expected-routed", type=int, default=218)
    parser.add_argument("--max-characters", type=int, default=6_000)
    parser.add_argument("--head-characters", type=int, default=3_500)
    parser.add_argument("--tail-characters", type=int, default=2_500)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Unrecognized Boolean value: {value!r}")


def normalize_text_for_match(value) -> str:
    value = html.unescape(str(value or "")).lower()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def normalize_label(value) -> str:
    text = str(value or "").strip().lower()
    for label in LABELS:
        if label.lower() in text:
            return label
    raise ValueError(f"Unrecognized label: {value!r}")


def normalize_source_label(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    for label in LABELS:
        if label.lower() in text:
            return label
    return ""


def minimally_sanitize_original_text(title, selftext) -> str:
    title = html.unescape("" if pd.isna(title) else str(title)).strip()
    body = html.unescape("" if pd.isna(selftext) else str(selftext)).strip()
    if body.lower() in {"[deleted]", "[removed]", "nan", "none"}:
        body = ""
    combined = title if not body else f"{title}\n\n{body}"
    combined = re.sub(r"https?://\S+|www\.\S+", " [URL] ", combined, flags=re.I)
    combined = re.sub(r"(?<!\w)(?:/u/|u/)[A-Za-z0-9_-]+", "[USER]", combined)
    combined = combined.replace("\r\n", "\n").replace("\r", "\n")
    combined = re.sub(r"[ \t]+", " ", combined)
    return re.sub(r"\n{3,}", "\n\n", combined).strip()


def limit_reasoning_text(value: str, args: argparse.Namespace) -> tuple[str, bool]:
    if len(value) <= args.max_characters:
        return value, False
    marker = "\n\n[Middle omitted to fit model context; beginning and ending preserved.]\n\n"
    available = args.max_characters - len(marker)
    if available <= 0:
        raise ValueError("max-characters must be longer than the omission marker")
    head_characters = min(args.head_characters, available)
    tail_characters = min(args.tail_characters, available - head_characters)
    limited = value[:head_characters] + marker + value[-tail_characters:]
    return limited, True


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def iter_original_source(source_url: str):
    """Stream the large multiline CSV with the standard parser, not pandas C engine."""
    raise_csv_field_limit()
    with urllib.request.urlopen(source_url) as response:
        text_stream = io.TextIOWrapper(response, encoding="utf-8", errors="replace", newline="")
        reader = csv.reader(text_stream)
        header = next(reader)
        required = ["title", "selftext", "title_with_selftext_cleaned", "class_group"]
        missing = [name for name in required if name not in header]
        if missing:
            raise ValueError(f"Source CSV is missing columns: {missing}")
        indexes = {name: header.index(name) for name in required}
        max_index = max(indexes.values())
        for row_number, row in enumerate(reader, start=2):
            if len(row) <= max_index:
                continue
            yield row_number, {
                name: row[index]
                for name, index in indexes.items()
            }


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.phase1_predictions)
    required = {
        "example_id", "text", "label_str", "phase1_label", "phase1_confidence",
        "phase1_routed", "routing_threshold", "temperature",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Phase 1 prediction file is missing: {sorted(missing)}")
    if len(predictions) != args.expected_total:
        raise ValueError(f"Expected {args.expected_total} rows, found {len(predictions)}")
    if predictions["example_id"].isna().any() or predictions["example_id"].duplicated().any():
        raise ValueError("example_id must be complete and unique")

    predictions["example_id"] = predictions["example_id"].astype(str)
    predictions["phase1_routed"] = predictions["phase1_routed"].map(normalize_bool)
    predictions["target_label"] = predictions["label_str"].map(normalize_label)
    routed = predictions[predictions["phase1_routed"]].copy()
    if len(routed) != args.expected_routed:
        raise ValueError(f"Expected {args.expected_routed} routed rows, found {len(routed)}")

    routed["_match_text"] = routed["text"].map(normalize_text_for_match)
    routed["_match_label"] = routed["target_label"].map(normalize_label)
    key_to_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
    for _, row in routed.iterrows():
        key_to_ids[(row["_match_text"], row["_match_label"])].append(row["example_id"])

    target_keys = set(key_to_ids)
    source_matches: dict[tuple[str, str], dict] = {}
    source_variants: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    source_counts: dict[tuple[str, str], int] = defaultdict(int)
    print(f"Streaming source for {len(routed)} routed rows from {args.source_url}")
    last_reported = 0
    for row_number, row in iter_original_source(args.source_url):
        key = (
            normalize_text_for_match(row["title_with_selftext_cleaned"]),
            normalize_source_label(row["class_group"]),
        )
        if key in target_keys:
            title = row["title"]
            body = row["selftext"]
            source_counts[key] += 1
            source_variants[key].add((title, body))
            source_matches.setdefault(key, {"title": title, "selftext": body})
        if row_number - last_reported >= args.chunk_size:
            print(f"rows {row_number:,}: matched keys {len(source_matches)}/{len(target_keys)}")
            last_reported = row_number

    ambiguous = [key for key, variants in source_variants.items() if len(variants) > 1]
    if ambiguous:
        raise ValueError(f"{len(ambiguous)} normalized keys map to conflicting original texts")

    mapping_rows = []
    for key, example_ids in key_to_ids.items():
        matched = source_matches.get(key)
        if matched is None:
            for example_id in example_ids:
                mapping_rows.append({"example_id": example_id, "original_match_status": "unmatched"})
            continue
        original = minimally_sanitize_original_text(matched["title"], matched["selftext"])
        limited, truncated = limit_reasoning_text(original, args)
        for example_id in example_ids:
            mapping_rows.append(
                {
                    "example_id": example_id,
                    "phase2_original_text": limited,
                    "phase2_original_character_count": len(original),
                    "phase2_input_was_truncated": truncated,
                    "original_source_candidate_count": source_counts[key],
                    "original_match_status": "normalized_exact",
                    "phase2_input_policy_version": "original-title-selftext-minimal-sanitize-v1",
                }
            )

    mapping = pd.DataFrame(mapping_rows)
    unmatched = mapping[mapping["original_match_status"] != "normalized_exact"]
    if not unmatched.empty:
        raise ValueError(
            f"Original-text linkage failed for {len(unmatched)} routed rows: "
            f"{unmatched['example_id'].head(10).tolist()}"
        )
    if len(mapping) != len(routed) or mapping["example_id"].duplicated().any():
        raise ValueError("Original-text mapping is not one-to-one with routed predictions")

    output = predictions.merge(mapping, on="example_id", how="left", validate="one_to_one")
    routed_output = output[output["phase1_routed"]]
    if routed_output["phase2_original_text"].isna().any():
        raise ValueError("At least one routed row has no original reasoning text")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    manifest = {
        "source_phase1_predictions": str(args.phase1_predictions),
        "source_phase1_predictions_sha256": sha256_file(args.phase1_predictions),
        "source_reddit_csv": args.source_url,
        "output_csv": str(args.output_csv),
        "output_csv_sha256": sha256_file(args.output_csv),
        "total_rows": len(output),
        "routed_rows": len(routed_output),
        "matched_routed_rows": int(routed_output["phase2_original_text"].notna().sum()),
        "truncated_routed_rows": int(
            routed_output["phase2_input_was_truncated"].astype("boolean").fillna(False).sum()
        ),
        "unique_example_ids": int(output["example_id"].nunique()),
        "temperature_values": sorted(output["temperature"].dropna().astype(float).unique().tolist()),
        "routing_threshold_values": sorted(
            output["routing_threshold"].dropna().astype(float).unique().tolist()
        ),
        "class_counts": output["target_label"].value_counts().sort_index().to_dict(),
        "phase2_input_policy_version": "original-title-selftext-minimal-sanitize-v1",
        "author_and_subreddit_fields_loaded": False,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
