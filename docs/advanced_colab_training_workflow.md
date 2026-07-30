# Advanced Colab Training Workflow

This document summarizes the current Colab-oriented first-stage modeling notebooks. These notebooks are the advanced versions intended for GPU-based training, W&B hyperparameter sweeps, temperature scaling, and confidence-threshold analysis.

## Canonical Files

| Model | Colab notebook | Script mirror |
|---|---|---|
| DistilBERT | `notebooks/colab/10_distilbert_confidence_threshold_wandb_colab.ipynb` | `src/colab_notebooks/distilbert_confidence_threshold_wandb_colab.py` |
| Llama 2 QLoRA | `notebooks/colab/11_llama2_confidence_threshold_wandb_colab.ipynb` | `src/colab_notebooks/llama2_confidence_threshold_wandb_colab.py` |
| Mistral QLoRA | `notebooks/colab/12_mistral_confidence_threshold_wandb_colab.ipynb` | `src/colab_notebooks/mistral_confidence_threshold_wandb_colab.py` |

The notebooks are the primary runnable artifacts in Colab. The `.py` files are script mirrors for review, diffing, and future pipeline refactoring.

## Default Smoke-Test Configuration

The advanced Colab notebooks use the following default split configuration:

| Parameter | Default |
|---|---:|
| `SAMPLES_PER_CLASS` | 300 |
| `TRAIN_RATIO` | 0.70 |
| `VALIDATION_RATIO` | 0.10 |
| `CALIBRATION_RATIO` | 0.10 |
| `TEST_RATIO` | 0.10 |

With three classes, `SAMPLES_PER_CLASS = 300` gives 900 total examples before splitting. The expected split is approximately 630 train, 90 validation, 90 calibration, and 90 held-out test examples.

For larger experiments, increase `SAMPLES_PER_CLASS` to values such as 1000, 20000, or 40000, depending on GPU memory and runtime budget.

## Split Semantics

| Split | Use |
|---|---|
| Train | Model fine-tuning |
| Validation | W&B sweep objective, model selection, early stopping, best checkpoint selection |
| Calibration | Temperature scaling and confidence-threshold selection |
| Held-out test | Final model evaluation only |

This separation is important for the manuscript because the confidence threshold should not be selected on the held-out test set.

## W&B Configuration

No W&B API key is committed in this repository.

Recommended options:

- Put `WANDB_API_KEY` in Colab Secrets, or enter the key only when `wandb.login()` prompts for it.
- Keep `WANDB_SWEEP_MODE = "new"` for an independent run.
- Use `WANDB_SWEEP_MODE = "continue_existing"` or `"reuse_best"` only when the collaborator has explicitly shared the W&B entity, project, and sweep ID.

The notebooks read these optional environment variables:

| Environment variable | Meaning |
|---|---|
| `WANDB_ENTITY` | W&B user or team. If unset, W&B uses the account selected at login. |
| `WANDB_PROJECT` | W&B project name. Each notebook has a safe default project name. |
| `WANDB_SWEEP_ID` | Existing sweep ID for `continue_existing` or `reuse_best` modes. |

## Advanced Confidence Analysis

The current Colab notebooks extend the earlier training workflow with:

- W&B macro-F1 sweep support
- final training with selected best hyperparameters
- held-out test evaluation
- temperature scaling using the calibration split
- confidence-threshold sweep
- risk-coverage analysis
- selective-risk and routing-rate summary
- prediction-level exports
- confusion matrix and error analysis
- token-length and confidence-distribution diagnostics

These outputs are intended to support the manuscript sections on:

- Phase 1 model performance
- confidence-guided routing
- risk-coverage threshold selection
- calibration analysis
- high-confidence and low-confidence error analysis
- selective end-to-end evaluation

## Generated Outputs

The notebooks write generated files into model-specific output directories such as:

- `distilbert_reddit_wandb_f1_threshold_outputs/`
- `llama2_reddit_wandb_qlora_threshold_outputs/`
- `mistral_reddit_wandb_qlora_threshold_outputs/`

Generated outputs, W&B run folders, model checkpoints, and large data files should not be committed to Git. Share them through W&B artifacts, cloud storage, or another agreed research storage location.

## Practical Colab Run Order

1. Open one notebook from GitHub in Colab.
2. Select a GPU runtime.
3. Run dependency installation.
4. Restart runtime if Colab asks.
5. Set `SAMPLES_PER_CLASS`.
6. Confirm `WANDB_SWEEP_MODE`.
7. Run W&B login through Colab Secrets or the secure prompt.
8. Run the notebook top to bottom.
9. Download or share generated metrics and prediction CSVs.
10. Repeat with the next model notebook.

## Manuscript Use

For manuscript updates, the most important files to collect from each model run are:

- final test metrics
- validation and calibration predictions
- threshold sweep table
- selected threshold metadata
- held-out test prediction CSV
- confidence bin summary
- high-confidence error cases

These files are needed to update the Phase 1 performance table, risk-coverage table, confidence calibration discussion, and selective-routing methodology.
