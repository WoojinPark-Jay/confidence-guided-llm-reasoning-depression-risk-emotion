Appendix A.X. Synthetic Mixed Emotion Dataset Generation Protocol

The supplementary Mixed Emotion Dataset was constructed as a controlled synthetic stress-test set for evaluating model behavior under emotionally ambiguous conditions. The dataset was not used for model training, hyperparameter tuning, or confidence-threshold selection. Instead, it was used only for supplementary evaluation of cases in which positive, neutral, and depression-related cues co-occur or shift across the text.

Generation Prompt

Generate short social-media-style English posts for a three-class proxy emotion classification stress test. Each example must contain emotionally mixed or shifting cues while remaining realistic, non-diagnostic, and free of personally identifying information. Use one of the target labels: Depression, Neutral, or Happy. The target label must reflect the dominant overall emotional trajectory of the post, not isolated phrases. Generate examples across the following scenario types: blended emotion co-occurrence, positive-to-distress shift, distress-to-recovery shift, neutral framing with subtle affect, and conflicting cues with a dominant trajectory. For each example, return the following fields: example_id, target_label, scenario_type, primary_context, dominant_trajectory, text, brief_label_rationale, positive_cue, neutral_cue, depression_related_cue, intended_use, used_for_training, used_for_threshold_selection, prompt_version, generation_model, generated_date, and clinical_disclaimer.

Labeling Rules

Depression examples were assigned when the dominant emotional trajectory centered on sadness, emptiness, withdrawal, hopelessness, or persistent emotional burden, even if brief positive or neutral details were present. Happy examples were assigned when the dominant emotional trajectory ended in relief, gratitude, connection, accomplishment, or cautious optimism, even if stress or sadness appeared earlier. Neutral examples were assigned when emotional cues were present but the post remained primarily descriptive, balanced, or informational, without a clearly dominant positive or depression-related trajectory.

Quality and Exclusion Criteria

Examples were excluded from the intended design space if they contained explicit clinical diagnosis claims, treatment recommendations, self-harm instructions, personally identifying information, off-topic content, or insufficient emotional ambiguity. Because the dataset is synthetic and limited in scale, it should be interpreted only as a controlled robustness probe. It is not a substitute for expert-annotated or naturally occurring mixed-emotion data.
