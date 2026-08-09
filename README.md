# Confidence-Guided LLM Reasoning for Depression-Risk Emotion Classification

This repository organizes the code and data workflow for a Reddit-based depression-risk-related proxy emotion classification project.

The project studies a confidence-guided two-phase framework for research-oriented proxy emotion classification in Reddit text. The codebase is organized step by step so the data preparation, preprocessing, modeling, and final end-to-end evaluation workflows can be reproduced.

## Pipeline Stages

### Stage 0. Subreddit Source Data Preparation

Notebook:

- `notebooks/00_reddit_subreddit_data_preparation.ipynb`

Script:

- `src/prepare_subreddit_data.py`

Purpose:

- Load subreddit-level Reddit archive dumps or existing raw parquet files.
- Convert `created_utc` timestamps.
- Select common post-level columns.
- Save subreddit-level prepared parquet files under `data/01_subreddit_preparation/`.

This stage consolidates the older per-subreddit notebooks such as `reddit_data_preprocessing_depression.ipynb`, `reddit_data_preprocessing_AnxietyDepression.ipynb`, and related notebooks.

### Stage 1. Whole Reddit Dataset Preprocessing

Notebook:

- `notebooks/01_reddit_data_preprocessing.ipynb`

Script:

- `src/preprocess_reddit.py`

Purpose:

- Load 11 subreddit-level preprocessed parquet files.
- Build the three proxy classes: Depression, Neutral, and Happy.
- Concatenate title and body text.
- Clean and normalize text.
- Compute TextBlob polarity scores.
- Apply sentiment-aware filtering.
- Apply class balancing.
- Export final modeling data and summary tables.

### Stage 2. Modeling

Local sample notebooks:

- `notebooks/local/02_distilbert_classification_sample_fine_tuning.ipynb`
- `notebooks/local/03_llama_classification_sample_fine_tuning.ipynb`
- `notebooks/local/04_mistral_classification_sample_fine_tuning.ipynb`

Colab large-run notebooks:

- `notebooks/colab/10_distilbert_confidence_threshold_wandb_colab.ipynb`
- `notebooks/colab/11_llama2_confidence_threshold_wandb_colab.ipynb`
- `notebooks/colab/12_mistral_confidence_threshold_wandb_colab.ipynb`

Final end-to-end Colab notebooks:

- `notebooks/colab/final/01_distilbert_phase1_training_final_colab.ipynb`
- `notebooks/colab/final/02_llm_phase2_reasoning_final_colab.ipynb`
- `notebooks/colab/final/03_mixed_emotion_end_to_end_orchestration_final_colab.ipynb`
- `notebooks/colab/final/04_reddit_test_routed_phase2_end_to_end_final_colab.ipynb`

Final workflow guide:

- `docs/final_end_to_end_workflow_ko.md`

Recommended final execution order:

1. `notebooks/colab/final/01_distilbert_phase1_training_final_colab.ipynb`
2. `notebooks/colab/final/02_llm_phase2_reasoning_final_colab.ipynb`
3. `notebooks/colab/final/03_mixed_emotion_end_to_end_orchestration_final_colab.ipynb`
4. `notebooks/colab/final/04_reddit_test_routed_phase2_end_to_end_final_colab.ipynb`

The first notebook trains and calibrates the DistilBERT Phase 1 model, saves the best model and threshold outputs, runs advanced confidence-threshold analysis, and runs Phase 1 inference on the 300-example Mixed Emotion stress-test set. The second notebook reads the saved Phase 1 Mixed Emotion predictions and applies Llama 2 CoT and Llama 3 SELF-DISCOVER only to routed Mixed Emotion rows while saving row-level resumable outputs, Phase 2 classification reports, label counts, and confusion matrices. The third notebook does not retrain models or rerun LLM inference; it merges the saved Mixed Emotion Phase 1 and Phase 2 outputs and generates paper-ready metrics, tables, figures, error examples, visual review displays, and zip exports. The fourth notebook uses the Reddit held-out test predictions from Final 01, sends only low-confidence routed Reddit test rows to Llama reasoning, and reconstructs the full Reddit held-out end-to-end result.

Script helpers:

- `src/modeling_data.py`
- `src/colab_notebooks/*.py`

Purpose:

- DistilBERT baseline and full fine-tuning workflow.
- Llama 2 / Mistral QLoRA first-stage classifier workflows.
- Load the final preprocessed dataset from `data/02_preprocessing_outputs/`.
- Sample a configurable number of records per class.
- Start with a small `SAMPLES_PER_CLASS` value for smoke testing.
- Increase the value later, for example to 20000 or 40000, for larger runs.
- Export sampled train/validation/test files under `data/03_modeling_inputs/`.
- Report both standard metrics and direct prediction counts, for example `Correct predictions: 267 / 300`.
- For the advanced Colab workflow, run W&B macro-F1 sweeps, final training, temperature scaling, risk-coverage threshold selection, held-out test evaluation, and confidence/error analysis.
- For the final paper workflow, use only the three notebooks under `notebooks/colab/final/`. These preserve the older exploratory notebooks while providing a cleaner end-to-end path from DistilBERT Phase 1 training to LLM Phase 2 reasoning and paper-ready Mixed Emotion evaluation outputs.
- The final DistilBERT notebook also exports paper-defense confidence analysis artifacts, including calibration metrics, reliability diagrams, risk-coverage curves, score ablations, bootstrap confidence intervals, threshold stability, per-class selective risk, high-confidence errors, and threshold provenance metadata.
- The final LLM reasoning notebook exports standalone Phase 2 evaluation artifacts, including classification reports, predicted-label distributions, parse-failure files, confusion matrix CSV files, and confusion matrix PNG files for both Llama 2 and Llama 3 when available.
- The final orchestration notebook exports the complete paper-ready Mixed Emotion result package and displays a final review section with metrics, routing coverage, correction counts, label distributions, classification reports, confusion matrix tables, confusion matrix images, and representative error rows.
- The final Reddit test notebook exports the primary held-out test two-phase result package, including routed-row Llama outputs, Reddit end-to-end metrics, correction analysis, routing coverage, confusion matrices, error examples, and a paper-ready workbook.
- Final Colab outputs are written to Google Drive under `/content/drive/MyDrive/confidence_guided_llm_reasoning/outputs_final/` so long-running training and reasoning results are not lost when a runtime ends.

Current local sample default:

- `SAMPLES_PER_CLASS = 1000`
- `TRAIN_RATIO = 0.75`
- `VALIDATION_RATIO = 0.15`
- `TEST_RATIO = 0.10`
- 1000 Depression, 1000 Neutral, and 1000 Happy records
- 3000 total records before splitting
- 2250 train records, 450 validation records, and 300 test records
- These values are defined near the top of each modeling notebook so the run size and split ratios can be changed in one place.

Current advanced Colab default:

- `SAMPLES_PER_CLASS = 300`
- `TRAIN_RATIO = 0.70`
- `VALIDATION_RATIO = 0.10`
- `CALIBRATION_RATIO = 0.10`
- `TEST_RATIO = 0.10`
- 300 Depression, 300 Neutral, and 300 Happy records
- 900 total records before splitting
- approximately 630 train records, 90 validation records, 90 calibration records, and 90 held-out test records
- The validation split is used for W&B tuning and checkpoint selection.
- The calibration split is used for temperature scaling and routing-threshold selection.
- The held-out test split is used only for final evaluation.

Local CPU/Mac execution note:

- The Llama and Mistral notebooks keep the original full-model workflow for GPU runs.
- When CUDA is not available, they automatically switch to tiny debug checkpoints so the notebook flow can be verified locally.
- To run the full Llama or Mistral checkpoints, use a CUDA GPU environment and disable the local tiny-model fallback only after confirming the environment can load the full model.


### Stage 3. Supplementary Mixed Emotion Stress-Test Dataset

Dataset files:

- `data/supplementary/mixed_emotion/mixed_emotion_stress_test_v2_3_300.csv`
- `data/supplementary/mixed_emotion/mixed_emotion_stress_test_v2_3_300.xlsx`
- `data/supplementary/mixed_emotion/mixed_emotion_stress_test_v2_3_300.jsonl`

Script:

- `scripts/generate_mixed_emotion_dataset.py`

Manuscript support:

- `docs/mixed_emotion_dataset_v2_2_manuscript_insert.md`
- `data/supplementary/mixed_emotion/appendix_mixed_emotion_dataset_protocol.md`

Purpose:

- Provide a controlled synthetic stress-test set for emotionally ambiguous examples.
- Include 300 examples, balanced across Depression, Neutral, and Happy proxy emotion labels.
- Include five ambiguity scenario types: blended emotion co-occurrence, positive-to-distress shift, distress-to-recovery shift, neutral framing with subtle affect, and conflicting cues with a dominant trajectory.
- Use this dataset only for supplementary robustness evaluation, not for Phase 1 training, hyperparameter tuning, or confidence-threshold selection.
- v2.3 clarifies final emotional trajectory and final takeaway cues while preserving the original class and scenario balance.

## Project Layout

```text
confidence-guided-selective-llm-reasoning/
  data/
    00_raw_reddit_archives/           # local archive dumps or raw parquet files; ignored by Git
    01_subreddit_preparation/         # subreddit-level prepared parquet files; ignored by Git
    02_preprocessing_outputs/         # final preprocessing CSV outputs; ignored by Git
    03_modeling_inputs/               # sampled train/validation/test CSVs; ignored by Git
    supplementary/
      mixed_emotion/                  # 300-example synthetic mixed-emotion stress-test dataset
  notebooks/
    00_reddit_subreddit_data_preparation.ipynb
    01_reddit_data_preprocessing.ipynb
    local/
      02_distilbert_classification_sample_fine_tuning.ipynb
      03_llama_classification_sample_fine_tuning.ipynb
      04_mistral_classification_sample_fine_tuning.ipynb
    colab/
      10_distilbert_confidence_threshold_wandb_colab.ipynb
      11_llama2_confidence_threshold_wandb_colab.ipynb
      12_mistral_confidence_threshold_wandb_colab.ipynb
      13_phase2_mixed_emotion_reasoning_colab.ipynb
      14_phase2_mixed_emotion_reasoning_trajectory_prompt_colab.ipynb
      final/
        01_distilbert_phase1_training_final_colab.ipynb
        02_llm_phase2_reasoning_final_colab.ipynb
        03_mixed_emotion_end_to_end_orchestration_final_colab.ipynb
        04_reddit_test_routed_phase2_end_to_end_final_colab.ipynb
  src/
    prepare_subreddit_data.py
    preprocess_reddit.py
    modeling_data.py
    colab_notebooks/
      distilbert_confidence_threshold_wandb_colab.py
      llama2_confidence_threshold_wandb_colab.py
      mistral_confidence_threshold_wandb_colab.py
  scripts/
    generate_mixed_emotion_dataset.py
    generate_mixed_emotion_dataset_v2_3.py
  docs/
    final_end_to_end_workflow_ko.md
  reports/figures/
  requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The preprocessing script downloads required NLTK resources on first run if they are missing.

## Run Stage 0

From the project root:

```bash
python src/prepare_subreddit_data.py
```

Expected outputs:

- `data/01_subreddit_preparation/*_preprocessed.parquet`
- `data/01_subreddit_preparation/subreddit_preparation_summary.csv`

## Run Stage 1

From the project root:

```bash
python src/preprocess_reddit.py
```

Expected outputs:

- `data/02_preprocessing_outputs/final_preprocessed_whole_df.csv`
- `data/02_preprocessing_outputs/final_preprocessed_df.csv`
- `data/02_preprocessing_outputs/class_counts_before_filtering.csv`
- `data/02_preprocessing_outputs/class_counts_after_filtering.csv`
- `data/02_preprocessing_outputs/final_class_distribution.csv`
- `data/02_preprocessing_outputs/filtering_summary.csv`

## Run Stage 2

Open one of the model notebooks and start with:

```python
SAMPLES_PER_CLASS = 1000
```

The modeling notebooks use this value to sample the same number of records from each class. For example, `1000` creates a 3000-record modeling set before train/validation/test splitting.

Default split:

- Train: 75%, or 2250 records when `SAMPLES_PER_CLASS = 1000`
- Validation: 15%, or 450 records when `SAMPLES_PER_CLASS = 1000`
- Test: 10%, or 300 records when `SAMPLES_PER_CLASS = 1000`

Each model evaluation prints the metric dictionary, the direct number of correct predictions, the number of incorrect predictions, and a normalized confusion matrix.

Example evaluation output:

```text
Correct predictions: 267 / 300 (89.00%)
Incorrect predictions: 33 / 300
```

For DistilBERT, the hidden-state visualization section uses `FEATURE_EXTRACTION_SAMPLE_SIZE = 300` by default to keep local exploratory analysis faster. This limit applies to the feature-extraction/visualization step, not to the main fine-tuning split.

The Llama and Mistral notebooks use the explicit train/validation/test CSV splits generated from these ratios. They do not re-split the train set after the CSV split, so the final notebook training/evaluation counts remain aligned with the values above.

Generated modeling inputs:

- `data/03_modeling_inputs/sample_1000_per_class/train_dataset.csv`
- `data/03_modeling_inputs/sample_1000_per_class/validation_dataset.csv`
- `data/03_modeling_inputs/sample_1000_per_class/test_dataset.csv`

## Run Stage 2 In Colab

Use the notebooks under `notebooks/colab/` for larger GPU training runs. These are the advanced Colab versions of the first-stage modeling workflow.

Recommended Colab order:

1. Open the target notebook from GitHub in Colab.
2. Run the dependency installation cell, then restart the runtime once if Colab asks.
3. Set `SAMPLES_PER_CLASS` near the top of the notebook. The default is `300` per class for a smoke test; increase it to values such as `1000`, `20000`, or `40000` for larger runs.
4. Keep `WANDB_SWEEP_MODE = "new"` for an independent sweep. Use `continue_existing` or `reuse_best` only when `WANDB_ENTITY`, `WANDB_PROJECT`, and `WANDB_SWEEP_ID` point to an accessible collaborator sweep.
5. Do not paste W&B tokens into notebook source code. Add `WANDB_API_KEY` in Colab Secrets or enter it only when the secure `wandb.login()` prompt appears.
6. If needed, set `WANDB_ENTITY`, `WANDB_PROJECT`, and `WANDB_SWEEP_ID` through Colab environment variables or edit the configuration cell for the current run only.

The Colab notebooks save W&B sweep results, predictions, metrics, threshold tables, figures, and final model outputs into their local Colab output directories. These generated outputs are ignored by Git and should be shared through W&B artifacts, cloud storage, or another agreed research storage location.

Detailed Colab workflow notes are available in:

- `docs/advanced_colab_training_workflow.md`


## Run Stage 3 / Inspect Mixed Emotion Dataset

The supplementary Mixed Emotion Dataset v2.3 is committed because it is small and intended to support reproducible stress-test evaluation. The previous v2.2 files are retained for traceability.

Open the spreadsheet version directly:

```text
data/supplementary/mixed_emotion/mixed_emotion_stress_test_v2_3_300.xlsx
```

Or regenerate the dataset from the project root:

```bash
python scripts/generate_mixed_emotion_dataset_v2_3.py
```

Dataset design summary:

- Depression: 100 examples
- Neutral: 100 examples
- Happy: 100 examples
- Total: 300 examples
- Scenario types: 5
- Examples per scenario type: 60
- Intended use: supplementary controlled stress-test only
- Not used for training or threshold selection

## Run Stage 4 / Phase 2 Reasoning on Mixed Emotion Dataset

Colab notebooks:

- Final recommended workflow: `notebooks/colab/14_phase2_mixed_emotion_reasoning_trajectory_prompt_colab.ipynb`
- Base/legacy comparison workflow: `notebooks/colab/13_phase2_mixed_emotion_reasoning_colab.ipynb`

Purpose:

- Load the 300-example Mixed Emotion Dataset directly from the GitHub raw CSV URL.
- Run Appendix B-aligned Llama 2 Chain-of-Thought prompting on all 300 examples.
- Run Appendix C-aligned Llama 3 SELF-DISCOVER prompting on all 300 examples.
- Preserve the raw reasoning outputs and add final-label columns for evaluation.
- Save model-specific outputs, a combined output table, and a summary evaluation CSV.
- Use the trajectory-aware Phase 2 prompt as the current final mixed-emotion reasoning workflow; the base prompt remains available only for comparison.

Default Phase 2 reasoning configuration:

- Dataset: `data/supplementary/mixed_emotion/mixed_emotion_stress_test_v2_3_300.csv`
- Default rows: `MAX_ROWS = 300`
- Llama 2 model: `NousResearch/Llama-2-7b-chat-hf`
- Llama 3 model: `NousResearch/Meta-Llama-3-8B-Instruct`
- Llama 2 final label: parsed from `Final label: Depression/Neutral/Happy` in `LLaMA2_3`; percentage breakdown is no longer used as the final decision rule
- Llama 3 final label: parsed from `Final label: Depression/Neutral/Happy` in `LLaMA3_Answer`
- `prompts.py` upload is not required because the SELF-DISCOVER prompt templates are embedded in the notebook.

Current final Phase 2 mixed-emotion workflow:

- `14_phase2_mixed_emotion_reasoning_trajectory_prompt_colab.ipynb` is the current recommended notebook for Phase 2 mixed-emotion reasoning.
- It keeps the same dataset, models, checkpoint/resume logic, and evaluation structure as the base Phase 2 notebook.
- It adds explicit guidance for blended or emotionally shifting texts so the model prioritizes final emotional trajectory and does not default to Neutral solely because multiple emotional cues are present.
- Llama 2 now writes a direct `Final label:` decision in `LLaMA2_3`; the earlier percentage-breakdown decision rule is no longer used for final labeling.
- Separate output filenames are used to avoid overwriting base prompt or earlier percentage-breakdown runs.

Current Phase 2 notebook assumption:

- This notebook can be run before Phase 1 routed predictions are available.
- In that mode, `target_label` is used as a temporary placeholder for the AI-generated label in the prompt.
- After Phase 1 threshold routing is available, set `PHASE1_LABEL_MODE = "prediction_column"` and provide a `prediction` column for the routed examples.

Colab execution note:

- Use a GPU runtime, preferably L4, A100, or T4.
- The notebook is independent from the Stage 2 training notebooks. Running it in a separate Colab runtime does not share variables, memory, or local files with another currently running notebook.
- Two notebooks can run at the same time if Colab grants separate runtimes, but Google account GPU quota or session limits may still interrupt one of them.

Detailed Phase 2 reasoning notes are available in:

- `docs/phase2_mixed_emotion_reasoning_colab_guide.md`
- `docs/phase2_trajectory_prompt_experiment_plan_ko.md`

Colab dependency note:

- The Phase 2 reasoning notebook requires `bitsandbytes>=0.46.1` for 4-bit model loading.
- Run the first setup cell before loading the models.
- If Colab imported `transformers` or `bitsandbytes` before the install finished, restart the runtime once and rerun from the top.

## Important Git Note

The local parquet, zst, and generated CSV files are large. They are intentionally ignored by Git.

For GitHub collaboration, use Git LFS, cloud storage, or an agreed shared data location rather than committing large data directly.

## Current Data Status

The local working copy currently contains the available Reddit archive/raw files and the subreddit-level preprocessed parquet files needed to run the notebooks. These data files are not pushed to GitHub.

Collaborators should place shared data files in the paths documented in `data/README.md`.
