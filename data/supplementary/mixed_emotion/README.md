# Mixed Emotion Stress-Test Dataset v2

This folder contains a controlled synthetic mixed-emotion stress-test dataset for supplementary evaluation.

- Total examples: 300
- Class balance: {'Depression': 100, 'Neutral': 100, 'Happy': 100}
- Scenario distribution: {'blended_emotion_cooccurrence': 60, 'positive_to_distress_shift': 60, 'distress_to_recovery_shift': 60, 'neutral_with_subtle_affect': 60, 'conflicting_cues_dominant_trajectory': 60}
- Intended use: supplementary robustness/stress-test evaluation only
- Not intended for: Phase 1 training, hyperparameter tuning, threshold selection, clinical validation, or diagnostic claims
- Generation model: GPT-5 Codex, 2026-07-23
- Prompt version: mixed-emotion-stress-test-v2.2

Files:
- `mixed_emotion_stress_test_v2_2_300.csv`: tabular CSV dataset
- `mixed_emotion_stress_test_v2_2_300.jsonl`: JSONL version
- `mixed_emotion_stress_test_v2_2_300.xlsx`: spreadsheet version for quick inspection
- `appendix_mixed_emotion_dataset_protocol.md`: manuscript-ready appendix protocol text
