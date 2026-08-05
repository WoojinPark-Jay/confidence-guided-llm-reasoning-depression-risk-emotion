# Mixed Emotion Stress-Test Dataset

This folder contains controlled synthetic mixed-emotion stress-test datasets for supplementary evaluation.

## Current recommended dataset

The current recommended version is v2.3:

- `mixed_emotion_stress_test_v2_3_300.csv`
- `mixed_emotion_stress_test_v2_3_300.jsonl`
- `mixed_emotion_stress_test_v2_3_300.xlsx`

v2.3 keeps the same 300-example structure as v2.2, but strengthens the final emotional trajectory and final takeaway cues in the generated texts. This version is intended for the trajectory-aware Phase 2 prompt experiment.

## Previous dataset

The previous v2.2 dataset is retained for traceability and comparison:

- `mixed_emotion_stress_test_v2_2_300.csv`
- `mixed_emotion_stress_test_v2_2_300.jsonl`
- `mixed_emotion_stress_test_v2_2_300.xlsx`

## Dataset design

For v2.3:

- Total examples: 300
- Class balance: Depression 100, Neutral 100, Happy 100
- Scenario distribution: 60 examples per scenario type
- Scenario types: blended emotion co-occurrence, positive-to-distress shift, distress-to-recovery shift, neutral framing with subtle affect, and conflicting cues with a dominant trajectory
- Intended use: supplementary robustness/stress-test evaluation only
- Not intended for: Phase 1 training, hyperparameter tuning, threshold selection, clinical validation, or diagnostic claims
- Generation model: GPT-5 Codex, 2026-08-05
- Prompt version: mixed-emotion-stress-test-v2.3

## Main v2.3 change

v2.3 makes the final emotional trajectory more explicit:

- Happy examples end with clearer relief, warmth, connection, cautious hope, or positive resolution.
- Depression examples end with clearer unresolved sadness, heaviness, withdrawal, emptiness, or emotional burden.
- Neutral examples remain measured, descriptive, observational, or balanced without a clear Happy- or Depression-oriented final takeaway.

This change is intended to reduce ambiguity caused by overly subtle endings in v2.2 while preserving the mixed-emotion stress-test design.

## Regeneration

To regenerate v2.3 from the project root:

```bash
python scripts/generate_mixed_emotion_dataset_v2_3.py
```

The earlier v2.2 generator remains available:

```bash
python scripts/generate_mixed_emotion_dataset.py
```

## Manuscript support

- `appendix_mixed_emotion_dataset_protocol.md`: appendix-style dataset protocol text for the current v2.3 design.
