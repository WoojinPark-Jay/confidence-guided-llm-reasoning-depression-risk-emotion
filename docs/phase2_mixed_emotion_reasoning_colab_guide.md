# Phase 2 Mixed Emotion Reasoning Colab Guide

This guide explains how to run the Phase 2 reasoning notebook for the supplementary Mixed Emotion Dataset.

Notebook:

```text
notebooks/colab/14_phase2_mixed_emotion_reasoning_trajectory_prompt_colab.ipynb
```

## Purpose

The notebook runs the current final trajectory-aware Phase 2 reasoning prompts on the 300-example Mixed Emotion Dataset. It is intended to make the input, model, raw reasoning output, final label extraction, result saving, and evaluation steps clear for collaborators.

It currently supports:

- Llama 2 with Chain-of-Thought prompting, aligned with Appendix Table A2.
- Llama 3 with SELF-DISCOVER prompting, aligned with Appendix Table A3.
- Final label extraction for both models.
- CSV outputs for each model, a combined output table, and a summary evaluation file.

## Dataset Loading

In Colab, the dataset is loaded directly from GitHub:

```python
url = (
    "https://raw.githubusercontent.com/WoojinPark-Jay/"
    "confidence-guided-llm-reasoning-depression-risk-emotion/"
    "refs/heads/feature/phase2-mixed-emotion-reasoning-colab/data/supplementary/mixed_emotion/"
    "mixed_emotion_stress_test_v2_3_300.csv"
)

df = pd.read_csv(url)
```

The current trajectory-aware prompt notebook uses the v2.3 dataset, which keeps 300 examples but clarifies final emotional trajectory cues. The dataset contains 300 examples:

- Depression: 100
- Neutral: 100
- Happy: 100

The main input columns are:

- `example_id`
- `text`
- `target_label`

## Default Run Configuration

The notebook is configured to run all 300 examples by default:

```python
MAX_ROWS = 300
```

For a quick smoke test, temporarily set:

```python
MAX_ROWS = 3
```

The model names follow the earlier reasoning notebooks:

```python
LLAMA2_MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"
LLAMA3_MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"
```

## Phase 1 Label Placeholder

The Phase 2 prompts expect an AI-generated label as input. If Phase 1 routed predictions are not available yet, the notebook uses the reference label as a temporary placeholder:

```python
PHASE1_LABEL_MODE = "target_as_placeholder"
```

When Phase 1 predictions are available, add a `prediction` column and switch to:

```python
PHASE1_LABEL_MODE = "prediction_column"
```

This later mode is the correct end-to-end setting after confidence-threshold routing is finalized.

## Llama 2 CoT Output

The Llama 2 section produces:

- `LLaMA2_1`: dominant emotion analysis
- `LLaMA2_2`: comparison with the AI-generated label
- `LLaMA2_3`: final decision text ending with `Final label: Depression`, `Final label: Neutral`, or `Final label: Happy`
- `LLaMA2_final_label`: final label parsed from `Final label:` in `LLaMA2_3`

Example:

```text
LLaMA2_3 = The text includes earlier distress, but the final emotional trajectory shows relief and positive resolution. Final label: Happy
LLaMA2_final_label = Happy
```

## Llama 3 SELF-DISCOVER Output

The Llama 3 section produces:

- `LLaMA3_SELECT`: selected reasoning modules
- `LLaMA3_ADAPT`: adapted reasoning modules
- `LLaMA3_IMPLEMENT`: implemented reasoning structure
- `LLaMA3_Answer`: structured answer with final label
- `LLaMA3_final_label`: parsed final label

The answer is instructed to end with one of:

```text
Final label: Depression
Final label: Neutral
Final label: Happy
```

## SELF-DISCOVER Runtime Mode

The default mode is closest to the original SELF-DISCOVER notebook:

```python
SELF_DISCOVER_STRUCTURE_MODE = "per_sample"
```

This generates SELECT, ADAPT, IMPLEMENT, and final answer outputs for every example. It is slower because 300 examples require many generation calls.

For a faster technical smoke test, use:

```python
SELF_DISCOVER_STRUCTURE_MODE = "fixed"
```

The fixed mode uses a reusable paper-safe reasoning structure and is not the preferred final experimental mode.

## Output Files

The notebook saves outputs under:

```text
outputs_phase2_reasoning/
```

Expected files:

```text
mixed_emotion_llama2_cot_final_label_results.csv
mixed_emotion_llama3_self_discover_results.csv
mixed_emotion_phase2_reasoning_summary.csv
mixed_emotion_phase2_reasoning_combined_outputs.csv
```

The combined output table is the easiest file to inspect first. It includes both models' raw outputs and final labels.

## Colab Runtime Notes

Use a GPU runtime. L4, A100, or T4 can be used depending on availability.

The first setup cell installs `bitsandbytes>=0.46.1`, which is required for 4-bit loading. If the notebook still raises a bitsandbytes import error after installation, restart the Colab runtime once and rerun from the top.

Running this notebook while another Colab notebook is running does not cause code-level conflicts if each notebook has its own runtime. The runtimes do not share variables, memory, or local output folders. However, Google Colab may limit simultaneous GPU sessions or total usage quota for the same account.

If Hugging Face access is required, add a Colab secret named:

```text
HF_TOKEN
```

The notebook reads it automatically when available.

## Current Role in the Project

This notebook is a Phase 2 reasoning-only workflow. It can be used now to verify the Mixed Emotion Dataset reasoning behavior on all 300 examples. Later, after Phase 1 confidence routing is finalized, the same structure should be applied only to routed low-confidence examples and evaluated with correction counts, introduced errors, net corrections, and final two-phase accuracy.

## Resume behavior

The Phase 2 Colab notebook uses row-level checkpointing. Each completed example is appended immediately to a model-specific CSV. By default, the notebook mounts Google Drive and writes outputs under `MyDrive/confidence_guided_llm_reasoning/outputs_phase2_reasoning/`, so checkpoints can survive Colab runtime resets. If the Colab runtime disconnects, rerun the notebook. It will load the existing result CSV, skip completed `example_id` values, and continue with only the unfinished rows. Manual batching is therefore optional rather than required.

Main output files:

- `outputs_phase2_reasoning/mixed_emotion_llama2_cot_final_label_results.csv`
- `outputs_phase2_reasoning/mixed_emotion_llama3_self_discover_results.csv`

For long runs, keep the Colab tab open until at least one row has completed and the CSV appears in Google Drive. If `USE_GOOGLE_DRIVE_OUTPUT = False`, outputs are only stored in the temporary Colab runtime and may disappear when the runtime is reset.

## Trajectory-aware prompt variant

The current recommended final mixed-emotion notebook is `notebooks/colab/14_phase2_mixed_emotion_reasoning_trajectory_prompt_colab.ipynb`. It keeps the same model, dataset, Google Drive checkpointing, row-level append/resume behavior, and evaluation structure as the main Phase 2 notebook, and strengthens the prompt policy for blended or emotionally shifting texts.

Detailed experiment rationale and comparison criteria are documented in `docs/phase2_trajectory_prompt_experiment_plan_ko.md`.

The variant uses separate output files so results do not overwrite the main prompt run:

- `mixed_emotion_llama2_cot_trajectory_prompt_v2_3_final_label_results.csv`
- `mixed_emotion_llama3_self_discover_trajectory_prompt_v2_3_results.csv`

