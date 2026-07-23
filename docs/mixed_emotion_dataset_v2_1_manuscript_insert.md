# Mixed Emotion Dataset v2.1: Manuscript-Ready Insert

## Recommended Scale

For this study, the supplementary Mixed Emotion Dataset should be reported as a controlled synthetic stress-test set rather than a primary benchmark. A scale of 300 examples, with 100 examples per class, is appropriate for this purpose. This size is large enough to provide more stable class-wise stress-test results than the original 90-example set, while still remaining transparent and manually auditable. If the dataset is later used to support stronger generalization claims, the recommended next step would be to expand it to at least 600 examples, with 200 examples per class, and add independent human review.

## Main-Text Replacement: Supplementary Mixed Emotion Dataset

In addition to the primary Reddit dataset, we constructed a supplementary synthetic Mixed Emotion Dataset to evaluate model behavior under emotionally ambiguous conditions. The dataset contains 300 controlled examples, with 100 examples each for the Depression, Neutral, and Happy proxy emotion classes. It was designed as a targeted stress-test set for cases in which positive, neutral, and depression-related cues co-occur or shift across the text. The dataset was not used for Phase 1 model training, hyperparameter tuning, or confidence-threshold selection.

Each example was generated according to a controlled prompt protocol that required mixed or shifting emotional cues, a dominant overall emotional trajectory, and a brief rationale for the assigned proxy label. Five scenario types were included: blended emotion co-occurrence, positive-to-distress shift, distress-to-recovery shift, neutral framing with subtle affect, and conflicting cues with a dominant trajectory. Each scenario type contributed 60 examples, resulting in balanced coverage across both class labels and ambiguity types.

The target label was assigned according to the dominant emotional trajectory of the text rather than isolated affective phrases. Depression examples were assigned when sadness, emptiness, withdrawal, hopelessness, or persistent emotional burden dominated the post. Happy examples were assigned when the overall trajectory ended in relief, gratitude, connection, accomplishment, or cautious optimism. Neutral examples were assigned when emotional cues were present but the post remained primarily descriptive, balanced, or informational. Because the dataset is synthetic, it should be interpreted only as a supplementary robustness probe and not as evidence of clinical validity or real-world prevalence.

## Main-Text Table

Table X. Composition of the supplementary Mixed Emotion Dataset

| Category | Number of examples | Description |
|---|---:|---|
| Depression | 100 | Mixed or shifting posts in which depression-related affect dominates the overall trajectory |
| Neutral | 100 | Mixed or shifting posts with a primarily descriptive, balanced, or informational dominant tone |
| Happy | 100 | Mixed or shifting posts in which positive affect, relief, gratitude, or hope dominates the overall trajectory |
| Total | 300 | Controlled synthetic stress-test examples |

Table Y. Scenario-type distribution in the supplementary Mixed Emotion Dataset

| Scenario type | Number of examples | Purpose |
|---|---:|---|
| Blended emotion co-occurrence | 60 | Tests cases where positive, neutral, and depression-related cues appear together |
| Positive-to-distress shift | 60 | Tests cases where initially positive cues are outweighed by later distress |
| Distress-to-recovery shift | 60 | Tests cases where negative cues shift toward recovery, relief, or hope |
| Neutral framing with subtle affect | 60 | Tests cases where emotion is present but framed in a measured or informational way |
| Conflicting cues with dominant trajectory | 60 | Tests whether the model follows the overall trajectory rather than isolated phrases |
| Total | 300 | Balanced scenario coverage |

## Appendix Text

Appendix A.X. Synthetic Mixed Emotion Dataset Generation Protocol

The supplementary Mixed Emotion Dataset was constructed as a controlled synthetic stress-test set for evaluating model behavior under emotionally ambiguous conditions. The dataset was not used for model training, hyperparameter tuning, or confidence-threshold selection. Instead, it was used only for supplementary evaluation of cases in which positive, neutral, and depression-related cues co-occur or shift across the text.

Generation prompt:

Generate short social-media-style English posts for a three-class proxy emotion classification stress test. Each example must contain emotionally mixed or shifting cues while remaining realistic, non-diagnostic, and free of personally identifying information. Use one of the target labels: Depression, Neutral, or Happy. The target label must reflect the dominant overall emotional trajectory of the post, not isolated phrases. Generate examples across the following scenario types: blended emotion co-occurrence, positive-to-distress shift, distress-to-recovery shift, neutral framing with subtle affect, and conflicting cues with a dominant trajectory. For each example, return the following fields: example_id, target_label, scenario_type, primary_context, dominant_trajectory, text, brief_label_rationale, positive_cue, neutral_cue, depression_related_cue, intended_use, used_for_training, used_for_threshold_selection, prompt_version, generation_model, generated_date, and clinical_disclaimer.

Labeling rules:

Depression examples were assigned when the dominant emotional trajectory centered on sadness, emptiness, withdrawal, hopelessness, or persistent emotional burden, even if brief positive or neutral details were present. Happy examples were assigned when the dominant emotional trajectory ended in relief, gratitude, connection, accomplishment, or cautious optimism, even if stress or sadness appeared earlier. Neutral examples were assigned when emotional cues were present but the post remained primarily descriptive, balanced, or informational, without a clearly dominant positive or depression-related trajectory.

Quality and exclusion criteria:

Examples were excluded from the intended design space if they contained explicit clinical diagnosis claims, treatment recommendations, self-harm instructions, personally identifying information, off-topic content, or insufficient emotional ambiguity. Because the dataset is synthetic and limited in scale, it should be interpreted only as a controlled robustness probe. It is not a substitute for expert-annotated or naturally occurring mixed-emotion data.

## Output Files

- Dataset CSV: `/private/tmp/paper_text_only_output/mixed_emotion_dataset_v2/mixed_emotion_stress_test_v2_1_300.csv`
- Dataset JSONL: `/private/tmp/paper_text_only_output/mixed_emotion_dataset_v2/mixed_emotion_stress_test_v2_1_300.jsonl`
- Appendix protocol: `/private/tmp/paper_text_only_output/mixed_emotion_dataset_v2/appendix_mixed_emotion_dataset_protocol.md`
- Generation script: `/Users/woojinpark/Documents/헬스케어 논문/mixed_emotion_dataset_generator.py`
