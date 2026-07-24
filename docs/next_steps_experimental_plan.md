# Next Steps for Experimental Validation and Manuscript Upgrade

This document summarizes the recommended next steps for strengthening the confidence-guided two-phase LLM reasoning framework. The goal is to align the codebase, experimental outputs, and manuscript claims before journal submission.

## 1. Current Research Position

The project currently has a clear methodological direction: a lightweight Phase 1 classifier handles high-confidence Reddit posts, while lower-confidence examples are selectively routed to a Phase 2 LLM reasoning module. This design is promising because it connects predictive performance, computational efficiency, and interpretability.

However, the next version of the study should focus on closing the experimental evidence gap. The main priority is to produce reproducible result files that directly support the confidence-guided routing claims, the mixed-emotion robustness claims, and the efficiency claims.

## 2. Priority Order

| Priority | Task | Purpose | Expected Output |
|---|---|---|---|
| 1 | Train and evaluate Phase 1 models on the large Reddit dataset | Establish the primary benchmark performance | Test metrics for DistilBERT, Mistral 7B, and Llama 2 7B |
| 2 | Save validation and test predictions with confidence scores | Create the basis for threshold selection and routing analysis | Prediction CSV files with true label, predicted label, confidence, and correctness |
| 3 | Run risk-coverage threshold sweep | Validate the confidence-guided routing mechanism | Threshold table with coverage, routing rate, selective risk, and captured errors |
| 4 | Evaluate selective end-to-end performance on the primary test set | Measure final framework performance after Phase 2 routing | End-to-end metrics and routed-sample analysis |
| 5 | Run Phase 2 reasoning only on routed samples | Test whether LLM reasoning corrects low-confidence errors | Corrected, unchanged, and worsened case counts |
| 6 | Evaluate the updated 300-example Mixed Emotion Dataset | Update the supplementary stress-test results | Mixed Emotion v2.2 accuracy, confusion matrix, and error analysis |
| 7 | Add calibration and confidence analysis | Defend the use of confidence-based routing | ECE, confidence histogram, reliability diagram, or high-confidence error analysis |
| 8 | Add efficiency analysis | Quantify the benefit of selective routing over all-sample LLM reasoning | LLM call reduction and relative cost/time comparison |
| 9 | Finalize dataset transparency tables | Reduce concerns about proxy labels and dataset construction | Class/source distribution and filtering/balancing tables |
| 10 | Update manuscript text, tables, figures, and appendix | Align all claims with final experimental outputs | Submission-ready manuscript draft |

## 3. Required Experimental Scope

### Primary Reddit Dataset

The primary Reddit dataset should be used for the main Phase 1 modeling results. The preferred setup is to train the Phase 1 classifiers using the large balanced dataset, approximately 40,000 examples per class if computationally feasible.

Required Phase 1 work:

- Train or confirm final DistilBERT, Mistral 7B, and Llama 2 7B classifiers.
- Evaluate each model on the held-out Reddit test set.
- Save prediction-level outputs for validation and test sets.
- Report accuracy, macro precision, macro recall, macro F1, and class-level metrics.

The full Reddit test set does not need to be routed entirely through the LLM reasoning stage. The end-to-end framework should apply Phase 2 only to the subset selected by the confidence threshold.

### Selective End-to-End Evaluation

The correct end-to-end evaluation is not all-sample LLM reasoning. The correct evaluation is:

1. Run Phase 1 on the test set.
2. Compute confidence for each prediction.
3. Accept predictions with confidence above the selected threshold.
4. Route only lower-confidence examples to Phase 2.
5. Combine accepted Phase 1 predictions and Phase 2 predictions.
6. Compute final end-to-end metrics on the full test set.

This directly evaluates the proposed selective routing framework.

### Mixed Emotion Dataset

The updated Mixed Emotion Dataset contains 300 examples, balanced across Depression, Neutral, and Happy. This dataset should not be used for training, threshold selection, or hyperparameter tuning. It should be used only as a supplementary robustness stress test.

Required Mixed Emotion work:

- Run Phase 1 predictions on all 300 examples.
- Apply the selected confidence threshold.
- Route low-confidence examples to Phase 2.
- Compute final end-to-end accuracy.
- Report corrected, unchanged, and worsened cases.
- Provide a confusion matrix and short qualitative error analysis.

## 4. Required Output Files

The following files should be generated and stored under an agreed experiment output directory. Large output files should not be committed to GitHub unless they are small and appropriate for repository storage.

| File | Description |
|---|---|
| `phase1_validation_predictions.csv` | Validation predictions with true label, predicted label, confidence, and correctness |
| `phase1_test_predictions.csv` | Test predictions from the Phase 1 model |
| `threshold_sweep_results.csv` | Candidate thresholds and risk-coverage statistics |
| `routed_test_samples.csv` | Test samples selected for Phase 2 reasoning |
| `phase2_reasoning_outputs.csv` | LLM predictions, rationales, and final labels for routed samples |
| `end_to_end_results.csv` | Final accepted/routed predictions for the full test set |
| `mixed_emotion_v2_2_results.csv` | Results on the 300-example Mixed Emotion Dataset |
| `calibration_results.csv` | Confidence-bin statistics and calibration metrics |

## 5. Manuscript Tables to Update

| Manuscript Table | Content |
|---|---|
| Table 1 | Reddit dataset class and subreddit-source distribution |
| Table 2 | Sentiment-aware filtering counts before and after filtering |
| Table 3 | Phase 1 model performance on the held-out Reddit test set |
| Table 4 | Risk-coverage threshold sweep |
| Table 5 | Selective end-to-end performance |
| Table 6 | Phase 2 correction analysis |
| Table 7 | Mixed Emotion Dataset v2.2 stress-test results |
| Appendix Table A1 | Reddit dataset variable descriptions |
| Appendix Table A2 | Chain-of-Thought prompting protocol |
| Appendix Table A3 | SELF-DISCOVER prompting protocol |
| Appendix Table A4 | Illustrative case-level outputs |
| Appendix Table A5 | Mixed Emotion Dataset generation protocol |

## 6. Manuscript Figures to Update

| Figure | Content |
|---|---|
| Figure 1 | Overall two-phase framework architecture |
| Figure 2 | Confidence score distribution |
| Figure 3 | Risk-coverage curve |
| Figure 4 | Confusion matrix for the primary test set |
| Figure 5 | Mixed Emotion Dataset correction flow or confusion matrix |

## 7. Suggested Role Allocation

| Role | Suggested Responsibility |
|---|---|
| Researcher A | Phase 1 large-run training and prediction export |
| Researcher B | Risk-coverage threshold sweep and calibration analysis |
| Researcher C | Phase 2 reasoning execution and rationale output cleaning |
| Researcher D | Mixed Emotion Dataset v2.2 evaluation and error analysis |
| Joint review | Final result validation, manuscript table updates, and limitation wording |

## 8. Key Manuscript Risks to Resolve

1. The Mixed Emotion Dataset has been expanded to 300 examples, but the manuscript must not keep reporting only the earlier 90-example result without clarification.
2. The confidence-guided routing method needs actual threshold-sweep results to support the equations and algorithm.
3. The reliability of softmax confidence should be addressed using calibration or confidence-bin analysis.
4. Phase 2 reasoning should be reported not only by accuracy improvement, but also by corrected, unchanged, and worsened cases.
5. Proxy labels must be clearly framed as subreddit-derived and sentiment-filtered proxy emotion labels, not clinical diagnostic labels.

## 9. Immediate Next Action

The immediate next action is to generate prediction-level outputs from the Phase 1 models. Once validation and test predictions with confidence scores are available, the team can produce the threshold sweep, routed subset, Phase 2 reasoning outputs, and final end-to-end evaluation tables.

Recommended first deliverables:

1. `phase1_validation_predictions.csv`
2. `phase1_test_predictions.csv`
3. `threshold_sweep_results.csv`
4. `routed_test_samples.csv`

These files will determine the next manuscript update and will clarify how much Phase 2 reasoning is needed for the final experiments.
