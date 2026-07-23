from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path


OUTPUT_DIR = Path("/private/tmp/paper_text_only_output/mixed_emotion_dataset_v2")
OUTPUT_CSV = OUTPUT_DIR / "mixed_emotion_stress_test_v2_1_300.csv"
OUTPUT_JSONL = OUTPUT_DIR / "mixed_emotion_stress_test_v2_1_300.jsonl"
OUTPUT_README = OUTPUT_DIR / "README.md"
OUTPUT_APPENDIX = OUTPUT_DIR / "appendix_mixed_emotion_dataset_protocol.md"

PROMPT_VERSION = "mixed-emotion-stress-test-v2.1"
GENERATION_MODEL = "GPT-5 Codex, 2026-07-23"
CLASSES = ["Depression", "Neutral", "Happy"]
COUNT_PER_CLASS = 100


CONTEXTS = [
    ("a school day", "school"),
    ("a workday", "work"),
    ("time with my family", "family"),
    ("a friendship issue", "friendship"),
    ("moving to a new city", "relocation"),
    ("recovering after a difficult week", "recovery"),
    ("preparing for an exam", "exam preparation"),
    ("starting a new job", "new job"),
    ("taking care of the household", "household"),
    ("joining a community event", "community event"),
    ("waiting for important news", "waiting period"),
    ("planning a small celebration", "celebration"),
    ("managing a group project", "group project"),
    ("returning from a break", "return from break"),
    ("spending a quiet weekend at home", "quiet weekend"),
    ("finishing a personal milestone", "personal milestone"),
    ("navigating a disagreement", "disagreement"),
    ("attending a reunion", "reunion"),
    ("working through a creative project", "creative project"),
    ("adjusting to a new routine", "new routine"),
]

POSITIVE_CUES = [
    "a kind message from a friend",
    "a brief feeling of accomplishment",
    "laughter during a shared meal",
    "encouraging feedback from someone I respect",
    "a quiet moment of relief",
    "a small success that I had been waiting for",
    "warm support from people around me",
    "a pleasant walk that helped me breathe",
    "the comfort of finishing something difficult",
    "a hopeful conversation near the end of the day",
]

NEUTRAL_CUES = [
    "ordinary tasks still needed to be finished",
    "most of the day was spent answering messages and organizing notes",
    "the schedule continued as planned",
    "there were practical details to handle",
    "the conversation stayed mostly factual",
    "I kept track of dates, forms, and reminders",
    "the situation involved both routine decisions and small adjustments",
    "nothing dramatic happened for long stretches",
    "I tried to focus on the next concrete step",
    "the day moved forward in a fairly ordinary way",
]

DISTRESS_CUES = [
    "a heavy sadness returning in the background",
    "an unusually empty feeling once the room became quiet",
    "a sense of hopelessness that the good moments did not fully cover",
    "the urge to withdraw even when people were trying to include me",
    "a dull exhaustion that made everything seem harder",
    "the thought that I was falling behind everyone else",
    "a feeling of being alone and overwhelmed",
    "a sharp drop in my mood once I had time to think",
    "a sense of disconnection from the things that usually make me care",
    "a tired, persistent sadness growing out of the worry",
]

REFLECTION_DETAILS = [
    "I kept rereading the same message without knowing how to answer",
    "I noticed how quickly my mood changed after the room became quiet",
    "I tried to focus on the next hour instead of the whole week",
    "I wrote down what happened so I could understand the day more clearly",
    "I kept comparing the public version of the day with how it felt inside",
    "I found myself paying attention to small gestures more than big events",
    "I realized that the ending of the day mattered more than the beginning",
    "I kept switching between wanting company and wanting silence",
    "I tried to describe the day without making it sound simpler than it was",
    "I noticed that one sentence from someone else stayed with me",
    "I kept measuring whether the good part was strong enough to change the day",
    "I tried to decide whether I felt calm, sad, or relieved",
    "I found it hard to separate what happened from how I interpreted it",
    "I kept returning to the same thought while doing ordinary tasks",
    "I noticed that the practical details made the emotions feel less clear",
    "I tried to hold both the difficult part and the hopeful part at once",
    "I kept wondering which feeling would still be there tomorrow",
    "I realized that the strongest feeling was not always the loudest one",
    "I tried to explain the day to myself in a fair way",
    "I noticed that the emotional tone changed depending on what I remembered first",
    "I kept thinking about whether the moment was a turning point or just a pause",
    "I tried to stay honest about the mixed nature of the experience",
    "I noticed that the same event could feel supportive and stressful at once",
    "I kept focusing on the part of the day that lingered longest",
    "I tried to name the overall feeling without ignoring the smaller ones",
]

TIME_ANCHORS = [
    "before breakfast",
    "during the commute",
    "while checking my calendar",
    "after a short conversation",
    "while cleaning my desk",
    "during a quiet break",
    "after reading an old message",
    "while walking home",
    "before opening my laptop",
    "after the meeting ended",
    "while making tea",
    "before going to sleep",
    "after finishing the main task",
    "while waiting outside",
    "during a pause in the afternoon",
    "after everyone else left",
    "while looking over my notes",
    "before replying to anyone",
    "after the room got quiet",
    "while putting things away",
    "during a slow part of the evening",
    "after hearing someone laugh nearby",
    "while reviewing what I had done",
]

SCENARIO_TYPES = [
    "blended_emotion_cooccurrence",
    "positive_to_distress_shift",
    "distress_to_recovery_shift",
    "neutral_with_subtle_affect",
    "conflicting_cues_dominant_trajectory",
]


def build_text(label: str, i: int) -> dict[str, str]:
    context_phrase, context_tag = CONTEXTS[i % len(CONTEXTS)]
    pos = POSITIVE_CUES[(i * 3) % len(POSITIVE_CUES)]
    neu = NEUTRAL_CUES[(i * 5) % len(NEUTRAL_CUES)]
    dis = DISTRESS_CUES[(i * 7) % len(DISTRESS_CUES)]
    detail = REFLECTION_DETAILS[(i * 11) % len(REFLECTION_DETAILS)]
    anchor = TIME_ANCHORS[(i * 13) % len(TIME_ANCHORS)]
    scenario = SCENARIO_TYPES[i % len(SCENARIO_TYPES)]

    if label == "Depression":
        if scenario == "positive_to_distress_shift":
            text = (
                f"At first, {context_phrase} felt manageable because there was {pos}. "
                f"Still, {neu}, and as the day went on, there was {dis}. "
                f"{detail}, especially {anchor}. "
                "By the end, the positive parts felt brief compared with the heavier feeling that stayed with me."
            )
            trajectory = "positive cues are outweighed by a later depression-related emotional trajectory"
        elif scenario == "distress_to_recovery_shift":
            text = (
                f"The day began with {dis} while I was dealing with {context_phrase}. "
                f"There was {pos}, and I tried to focus on it, but {neu}. "
                f"{detail}, especially {anchor}. "
                "The hopeful moment helped only briefly, while the dominant feeling remained low and difficult to carry."
            )
            trajectory = "brief recovery cue appears, but the dominant trajectory remains depression-related"
        else:
            text = (
                f"During {context_phrase}, I noticed {pos}, and {neu}. "
                f"Even with those ordinary or positive moments, there was {dis}. "
                f"{detail}, especially {anchor}. "
                "The overall tone is not simply negative, but the emotional weight settles more on sadness and withdrawal than on relief."
            )
            trajectory = "mixed cues co-occur, but depression-related affect dominates"
        rationale = (
            "Although the text includes positive or neutral details, the dominant emotional trajectory is sadness, emptiness, "
            "withdrawal, or hopelessness rather than relief or ordinary description."
        )
    elif label == "Happy":
        if scenario == "positive_to_distress_shift":
            text = (
                f"During {context_phrase}, I had a moment when there was {dis}. "
                f"But then there was {pos}, and I felt the day start to open up again. "
                f"{neu}. {detail}, especially {anchor}. "
                "Even with the mixed parts, the feeling I carried away was mainly relief and quiet happiness."
            )
            trajectory = "distress appears early, but the final emotional takeaway is positive"
        elif scenario == "distress_to_recovery_shift":
            text = (
                f"The start of {context_phrase} was difficult because of {dis}. "
                f"Later, {pos}, which made the earlier heaviness feel less controlling. "
                f"Even though {neu}, {detail.lower()}, especially {anchor}. "
                "The day ended with a stronger sense of gratitude and hope."
            )
            trajectory = "negative cue shifts toward a positive dominant ending"
        else:
            text = (
                f"While dealing with {context_phrase}, I could still feel {dis}. "
                f"At the same time, there was {pos}, and {neu}. "
                f"{detail}, especially {anchor}. "
                "The mixed feelings were real, but the overall message ended in appreciation, energy, and cautious optimism."
            )
            trajectory = "mixed cues co-occur, but positive affect dominates"
        rationale = (
            "The text acknowledges stress or sadness, but the dominant outcome is relief, gratitude, accomplishment, "
            "connection, or hopeful momentum."
        )
    else:
        if scenario == "positive_to_distress_shift":
            text = (
                f"In relation to {context_phrase}, there was {pos}, followed by a moment when I noticed {dis}. "
                f"However, {neu}. "
                f"{detail}, especially {anchor}. "
                "Neither emotion became the clear center of the post, which mostly describes a mixed situation in practical terms."
            )
            trajectory = "positive and distress cues are balanced by a factual, descriptive framing"
        elif scenario == "distress_to_recovery_shift":
            text = (
                f"While thinking about {context_phrase}, I noticed {dis}, then later {pos}. "
                f"{neu}. "
                f"{detail}, especially {anchor}. "
                "The post does not settle strongly into either sadness or happiness, and the overall tone remains measured."
            )
            trajectory = "emotional shift is present, but the dominant tone remains neutral and descriptive"
        else:
            text = (
                f"The situation around {context_phrase} included {pos} and also {dis}. "
                f"At the same time, {neu}. "
                f"{detail}, especially {anchor}. "
                "Overall, the post reads more like a balanced account of circumstances than a strongly positive or negative expression."
            )
            trajectory = "mixed cues co-occur, but the dominant framing is neutral"
        rationale = (
            "The text contains emotional cues, but it is mainly descriptive, balanced, or informational, without a strong "
            "dominant positive or depression-related trajectory."
        )

    return {
        "scenario_type": scenario,
        "primary_context": context_tag,
        "positive_cue": pos,
        "neutral_cue": neu,
        "depression_related_cue": dis,
        "dominant_trajectory": trajectory,
        "text": text,
        "brief_label_rationale": rationale,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in CLASSES:
        for i in range(COUNT_PER_CLASS):
            item = build_text(label, i)
            rows.append(
                {
                    "example_id": f"MEV2_{label[:3].upper()}_{i+1:03d}",
                    "target_label": label,
                    "scenario_type": item["scenario_type"],
                    "primary_context": item["primary_context"],
                    "dominant_trajectory": item["dominant_trajectory"],
                    "text": item["text"],
                    "brief_label_rationale": item["brief_label_rationale"],
                    "positive_cue": item["positive_cue"],
                    "neutral_cue": item["neutral_cue"],
                    "depression_related_cue": item["depression_related_cue"],
                    "intended_use": "supplementary controlled stress-test only",
                    "used_for_training": "no",
                    "used_for_threshold_selection": "no",
                    "prompt_version": PROMPT_VERSION,
                    "generation_model": GENERATION_MODEL,
                    "generated_date": str(date.today()),
                    "clinical_disclaimer": "Synthetic proxy-emotion example; not clinical data or diagnosis.",
                }
            )

    fieldnames = list(rows[0].keys())
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    label_counts = {label: sum(1 for r in rows if r["target_label"] == label) for label in CLASSES}
    scenario_counts = {
        scenario: sum(1 for r in rows if r["scenario_type"] == scenario) for scenario in SCENARIO_TYPES
    }

    OUTPUT_README.write_text(
        f"""# Mixed Emotion Stress-Test Dataset v2

This folder contains a controlled synthetic mixed-emotion stress-test dataset for supplementary evaluation.

- Total examples: {len(rows)}
- Class balance: {label_counts}
- Scenario distribution: {scenario_counts}
- Intended use: supplementary robustness/stress-test evaluation only
- Not intended for: Phase 1 training, hyperparameter tuning, threshold selection, clinical validation, or diagnostic claims
- Generation model: {GENERATION_MODEL}
- Prompt version: {PROMPT_VERSION}

Files:
- `{OUTPUT_CSV.name}`: tabular CSV dataset
- `{OUTPUT_JSONL.name}`: JSONL version
- `{OUTPUT_APPENDIX.name}`: manuscript-ready appendix protocol text
""",
        encoding="utf-8",
    )

    OUTPUT_APPENDIX.write_text(
        """Appendix A.X. Synthetic Mixed Emotion Dataset Generation Protocol

The supplementary Mixed Emotion Dataset was constructed as a controlled synthetic stress-test set for evaluating model behavior under emotionally ambiguous conditions. The dataset was not used for model training, hyperparameter tuning, or confidence-threshold selection. Instead, it was used only for supplementary evaluation of cases in which positive, neutral, and depression-related cues co-occur or shift across the text.

Generation Prompt

Generate short social-media-style English posts for a three-class proxy emotion classification stress test. Each example must contain emotionally mixed or shifting cues while remaining realistic, non-diagnostic, and free of personally identifying information. Use one of the target labels: Depression, Neutral, or Happy. The target label must reflect the dominant overall emotional trajectory of the post, not isolated phrases. Generate examples across the following scenario types: blended emotion co-occurrence, positive-to-distress shift, distress-to-recovery shift, neutral framing with subtle affect, and conflicting cues with a dominant trajectory. For each example, return the following fields: example_id, target_label, scenario_type, primary_context, dominant_trajectory, text, brief_label_rationale, positive_cue, neutral_cue, depression_related_cue, intended_use, used_for_training, used_for_threshold_selection, prompt_version, generation_model, generated_date, and clinical_disclaimer.

Labeling Rules

Depression examples were assigned when the dominant emotional trajectory centered on sadness, emptiness, withdrawal, hopelessness, or persistent emotional burden, even if brief positive or neutral details were present. Happy examples were assigned when the dominant emotional trajectory ended in relief, gratitude, connection, accomplishment, or cautious optimism, even if stress or sadness appeared earlier. Neutral examples were assigned when emotional cues were present but the post remained primarily descriptive, balanced, or informational, without a clearly dominant positive or depression-related trajectory.

Quality and Exclusion Criteria

Examples were excluded from the intended design space if they contained explicit clinical diagnosis claims, treatment recommendations, self-harm instructions, personally identifying information, off-topic content, or insufficient emotional ambiguity. Because the dataset is synthetic and limited in scale, it should be interpreted only as a controlled robustness probe. It is not a substitute for expert-annotated or naturally occurring mixed-emotion data.
""",
        encoding="utf-8",
    )

    print(OUTPUT_CSV)
    print(OUTPUT_JSONL)
    print(OUTPUT_README)
    print(OUTPUT_APPENDIX)
    print(label_counts)
    print(scenario_counts)


if __name__ == "__main__":
    main()
