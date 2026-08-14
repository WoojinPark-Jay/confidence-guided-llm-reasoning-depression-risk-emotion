# 최종 End-to-End Colab Workflow 설명서

작성일: 2026-08-07  
대상 폴더: `notebooks/colab/final/`

## 1. 목적

이 문서는 논문 제출용 최종 실험을 재현하기 위한 Colab 실행 구조를 정리한다. 기존 `10`, `13`, `14`번 노트북은 실험 히스토리와 재현성을 위해 그대로 유지한다. 최종 논문용 실행은 별도 폴더인 `notebooks/colab/final/` 안의 노트북을 사용한다.

최종 구조를 분리한 이유는 다음과 같다.

| 이유 | 설명 |
|---|---|
| 기존 코드 보존 | 이전 실험 노트북을 직접 수정하지 않아 과거 결과와 비교 가능 |
| 실행 명확성 | 어떤 파일을 어떤 순서로 실행해야 하는지 명확함 |
| 세션 안정성 | Colab 런타임 종료 후에도 결과가 Google Drive에 남도록 설계 |
| 논문 결과 생성 | end-to-end 결과, confusion matrix, paper-ready table을 한 곳에서 생성 |
| 리스크 감소 | Phase 1 학습, Phase 2 reasoning, 최종 평가를 분리해 중간 실패 시 재실행 범위가 작음 |

## 2. 최종 노트북

| 순서 | 노트북 | 역할 |
|---:|---|---|
| 1 | `01_distilbert_phase1_training_final_colab.ipynb` | DistilBERT Phase 1 학습, temperature scaling, routing threshold 선택, advanced confidence-threshold 분석, Mixed Emotion Phase 1 prediction 생성 |
| 2 | `02_llm_phase2_reasoning_final_colab.ipynb` | Phase 1에서 routed된 Mixed Emotion 샘플에 대해 Llama 2 CoT와 Llama 3 SELF-DISCOVER reasoning 실행 |
| 3 | `03_mixed_emotion_end_to_end_orchestration_final_colab.ipynb` | Phase 1 결과와 Phase 2 결과를 merge하고 최종 end-to-end metric, 표, 그림 생성 |
| 4 | `04_reddit_test_routed_phase2_end_to_end_final_colab.ipynb` | Reddit held-out test에서 routed된 샘플만 Phase 2 reasoning에 태우고 Reddit test 전체 기준 end-to-end 결과 생성 |
| 4.5 | `04_5_reddit_test_routed_phase2_original_text_primary_final_colab.ipynb` | **최종 Reddit 재실험**: primary `tau=0.70`의 171건을 원문 `title + selftext`로 복원해 Llama 2와 Llama 3를 실행하고 12,000건 end-to-end 결과 생성 |

중요한 점은 3번 노트북 하나만 돌린다고 전체가 자동으로 실행되는 구조가 아니라는 것이다. 3번은 이미 저장된 1번/2번 산출물을 읽어서 Mixed Emotion 최종 평가와 논문용 표/그림을 만드는 오케스트레이션 파일이다. Reddit 최종 논문 결과에는 기존 cleaned-text 4번 계열 대신 4.5를 사용한다. 4.5는 Phase 1 모델과 routed ID를 바꾸지 않고, Phase 2 입력만 최소 비식별 처리된 원문 `title + selftext`로 바로잡는다.

자세한 검증 규칙과 저장 파일은 `docs/reddit_phase2_original_text_primary_final_run_ko.md`에 정리되어 있다.

### 2.1 완료된 최종 결과

- Reddit 12,000건 Phase 1: 96.69%, 171건 routed
- Reddit + Llama 2 CoT v2: 96.67%, 47 corrected, 50 introduced, net -3
- Reddit + Llama 3 SELF-DISCOVER: 96.94%, 42 corrected, 12 introduced, net +30
- Mixed Emotion 300건 Phase 1: 81.33%, 44건 routed
- Mixed Emotion + Llama 2 CoT v2: 85.33%, net +12
- Mixed Emotion + Llama 3 SELF-DISCOVER: 87.33%, net +18

Reddit 최종 결과는 04.5의 원문-input 산출물을 사용한다. 과거 cleaned-input 04/04.3 결과는 prompt/input 개발 이력으로만 남기며 최종 논문 수치로 사용하지 않는다.

## 3. 전체 실행 흐름

```text
Primary Reddit Dataset
        |
        v
01 DistilBERT Phase 1
        |
        |-- phase1_test_predictions.csv
        |-- phase1_threshold_calibration_table.csv
        |-- advanced_confidence_threshold_analysis/
        |-- phase1_mixed_emotion_predictions.csv
        v
Mixed Emotion routed rows
        |
        v
02 LLM Phase 2 Reasoning
        |
        |-- llama2_cot_routed_mixed_emotion_results.csv
        |-- llama3_self_discover_routed_mixed_emotion_results.csv
        v
03 End-to-End Orchestration
        |
        |-- mixed_emotion_end_to_end_results.csv
        |-- end_to_end_metrics_summary.csv
        |-- routing_coverage_table.csv
        |-- phase2_correction_analysis.csv
        |-- paper_ready_tables.xlsx
        |-- confusion matrix PNG files

Reddit held-out test routed rows
        |
        v
04.5 Reddit Test Original-Text Routed Phase 2 + End-to-End
        |
        |-- reddit_test_llama2_cot_original_text_tau070_results.csv
        |-- reddit_test_llama3_self_discover_original_text_tau070_results.csv
        |-- reddit_test_end_to_end_results.csv
        |-- reddit_test_end_to_end_metrics_summary.csv
        |-- reddit_test_phase2_correction_analysis.csv
        |-- reddit_test_paper_ready_tables.xlsx
        |-- reddit test confusion matrix PNG files
```

## 4. Google Drive 저장 원칙

최종 노트북은 모두 `REQUIRE_PERSISTENT_OUTPUT = True`를 기본값으로 둔다. 즉, Google Drive가 정상적으로 mount되지 않으면 긴 학습이나 추론을 시작하지 않고 바로 중단한다.

이 설정이 중요한 이유는 Colab의 `/content` 경로가 런타임 종료 후 사라지기 때문이다. 모델 학습이나 Llama reasoning은 몇 시간 이상 걸릴 수 있으므로, 결과가 `/content`에만 저장되면 세션 종료 후 파일이 사라질 수 있다.

최종 출력 기본 경로는 다음과 같다.

```text
/content/drive/MyDrive/confidence_guided_llm_reasoning/outputs_final/
```

하위 폴더는 다음과 같이 나뉜다.

| 폴더 | 내용 |
|---|---|
| `outputs_final/phase1_distilbert/` | DistilBERT 모델, Phase 1 예측, calibration 결과 |
| `outputs_final/phase1_distilbert/advanced_confidence_threshold_analysis/` | calibration metric, reliability diagram, risk-coverage curve, ablation, bootstrap, threshold stability 등 confidence-threshold 분석 결과 |
| `outputs_final/phase2_llm_reasoning/` | Llama 2, Llama 3 routed sample reasoning 결과 |
| `outputs_final/end_to_end_orchestration/` | 최종 end-to-end 평가 결과, 논문용 표/그림 |
| `outputs_final/reddit_test_phase2_reasoning/` | Reddit held-out test routed Phase 2 reasoning 및 Reddit test end-to-end 평가 결과 |
| `outputs_final/reddit_test_phase2_reasoning_original_text_primary_tau070_final/` | 최종 Reddit 원문-input primary-policy Phase 2 및 12,000건 end-to-end 결과 |

각 노트북 마지막에는 zip export 셀이 있다. 이 셀은 해당 단계에서 생성된 CSV, JSON, PNG, XLSX 파일을 하나의 zip으로 묶고 브라우저 다운로드를 시도한다. 따라서 Drive 저장이 1차 안전장치이고, zip 다운로드가 2차 안전장치이다.

## 5. 01 DistilBERT Phase 1 노트북

파일:

```text
notebooks/colab/final/01_distilbert_phase1_training_final_colab.ipynb
```

### 5.1 입력

기본 primary dataset 입력은 기존 DistilBERT Colab에서 사용하던 GitHub media URL이다.

```text
https://media.githubusercontent.com/media/Branden-Kang/LLaMA-2/main/data/final_preprocessed_df2.csv
```

따라서 별도로 primary CSV를 Google Drive에 업로드하지 않아도 기본 실행이 가능하도록 구성했다. Google Drive는 입력 저장소가 아니라 출력 저장소로 사용한다.

다만 GitHub media URL이 일시적으로 느리거나 접근이 제한되는 경우를 대비해 Drive fallback 경로도 남겨두었다.

```text
/content/drive/MyDrive/confidence_guided_llm_reasoning/data/final_preprocessed_df.csv
```

기본값에서는 GitHub media URL을 먼저 사용한다.

Mixed Emotion Dataset은 v2.4 neutral-clear 버전을 사용한다. 이 버전은 기존 Depression/Happy stress-test 예시는 유지하고, Neutral 100개를 더 명확한 factual/routine neutral 문장으로 교체한 버전이다.

01번 노트북은 다음 순서로 Mixed Emotion 파일을 찾는다.

```text
/content/mixed_emotion_stress_test_v2_4_neutral_clear_300.csv
/content/drive/MyDrive/confidence_guided_llm_reasoning/data/mixed_emotion_stress_test_v2_4_neutral_clear_300.csv
GitHub raw URL: data/supplementary/mixed_emotion/mixed_emotion_stress_test_v2_4_neutral_clear_300.csv
```

즉 Colab에서 로컬 업로드 파일이 있으면 그 파일을 먼저 읽고, 없으면 Drive 파일, 그것도 없으면 GitHub raw URL의 v2.4 파일을 읽는다.

### 5.2 주요 파라미터

| 파라미터 | 의미 | 최종 권장값 |
|---|---|---:|
| `SAMPLES_PER_CLASS` | class별 학습 샘플 수 | `40000` |
| `SAMPLING_MODE` | class별 sampling 방식 | `reservoir` |
| `TRAIN_RATIO` | train split 비율 | `0.70` |
| `VALIDATION_RATIO` | model selection validation split | `0.10` |
| `CALIBRATION_RATIO` | temperature/threshold calibration split | `0.10` |
| `TEST_RATIO` | final held-out test split | `0.10` |
| `MAX_LENGTH` | token 최대 길이 | `256` |
| `USE_WANDB_SWEEP` | W&B hyperparameter sweep 실행 여부 | `True` |
| `WANDB_SWEEP_COUNT` | sweep trial 수 | `4` |
| `TARGET_SELECTIVE_RISK` | accepted set의 목표 selective risk | `0.05` |
| `THRESHOLD_GRID_MIN` | final operating policy의 threshold 후보 최소값 | `0.70` |
| `THRESHOLD_GRID_MAX` | final operating policy의 threshold 후보 최대값 | `1.00` |

Smoke test에서는 `SAMPLES_PER_CLASS = 1000` 또는 `3000`으로 줄여도 된다. 최종 논문용 결과는 class별 40000개 기준으로 다시 실행한다.

### 5.2.1 DistilBERT 최적화 구조

01번 노트북은 단순히 고정 hyperparameter로 한 번 학습하는 구조가 아니라, W&B sweep으로 validation macro-F1 기준 best hyperparameter를 먼저 선택한다.

실행 흐름은 다음과 같다.

```text
Primary Reddit dataset
-> class별 balanced sampling
-> train / validation / calibration / test split
-> W&B sweep on validation macro-F1
-> best hyperparameter 선택
-> best hyperparameter로 final DistilBERT training
-> best model 저장
-> calibration split으로 temperature scaling
-> calibration split으로 routing threshold 선택: conservative candidate range [0.70, 1.00]
-> held-out Reddit test 평가
-> Mixed Emotion v2.4 300개 전체에 Phase 1 inference
```

Mixed Emotion 300개는 여기서 학습, validation, calibration, threshold 선택에 사용하지 않는다. 완성된 DistilBERT 모델, temperature, threshold를 그대로 적용하는 외부 stress-test set이다.

Threshold 후보 범위는 최종 운영 정책 기준 `[0.70, 1.00]`을 사용한다. 3-class softmax에서 0.50 근처 confidence는 약한 과반 confidence에 가까우므로, depression-risk-related emotion classification에서는 중간 confidence 예측을 Phase 2 reasoning으로 보내는 보수적 운영 정책이 더 방어 가능하다는 판단이다. 동시에 `phase1_threshold_grid_lower_bound_sensitivity.csv`를 저장하여 `[0.50, 0.60, 0.70, 0.75, 0.80]` lower-bound별 routing/coverage/risk 변화를 비교할 수 있게 한다.

또한 01번 노트북은 threshold 선택만 수행하는 것이 아니라, 선택된 confidence filtering 방법을 논문에서 방어하기 위한 advanced confidence-threshold 분석 산출물도 함께 생성한다. 이 산출물은 `advanced_confidence_threshold_analysis/` 하위 폴더에 저장된다.

W&B를 쓰려면 Colab secret에 다음 이름으로 API key를 저장한다.

```text
WANDB_API_KEY
```

### 5.3 출력

| 출력 파일 | 설명 |
|---|---|
| `phase1_training_sample.csv` | 학습에 사용된 balanced sample |
| `phase1_train_split.csv` | train split |
| `phase1_validation_split.csv` | validation split |
| `phase1_calibration_split.csv` | calibration split |
| `phase1_test_split.csv` | held-out test split |
| `distilbert_best_model/` | 저장된 DistilBERT best model |
| `phase1_threshold_calibration_table.csv` | threshold별 coverage/risk table |
| `phase1_threshold_grid_lower_bound_sensitivity.csv` | threshold 후보 lower-bound별 routing/coverage/risk sensitivity table |
| `phase1_selected_threshold.csv` | 최종 선택 threshold |
| `phase1_test_predictions.csv` | held-out test prediction |
| `distilbert_phase1_summary.csv` | Phase 1 핵심 metric |
| `distilbert_phase1_advanced_summary.csv` | calibration, threshold, AURC 등 advanced confidence 분석 요약 |
| `advanced_confidence_threshold_analysis/calibration_metric_summary.csv` | raw MSP와 temperature-scaled MSP의 ECE, adaptive ECE, Brier score, NLL |
| `advanced_confidence_threshold_analysis/calibration_reliability_diagram.png` | raw MSP와 temperature-scaled MSP calibration reliability diagram |
| `advanced_confidence_threshold_analysis/risk_coverage_curve.png` | calibration split의 risk-coverage curve |
| `advanced_confidence_threshold_analysis/confidence_score_ablation.csv` | MSP, margin, negative entropy confidence score ablation |
| `advanced_confidence_threshold_analysis/fixed_threshold_bootstrap_summary.csv` | 선택 threshold 기준 bootstrap confidence interval |
| `advanced_confidence_threshold_analysis/threshold_bootstrap_stability.csv` | bootstrap 기반 threshold stability |
| `advanced_confidence_threshold_analysis/high_confidence_accepted_errors.csv` | threshold 이상 accepted sample 중 high-confidence error 사례 |
| `advanced_confidence_threshold_analysis/per_class_selective_metrics_test.csv` | class별 selective coverage/risk |
| `advanced_confidence_threshold_analysis/class_conditional_thresholds.csv` | class-conditional threshold ablation |
| `advanced_confidence_threshold_analysis/input_length_threshold_analysis.csv` | 입력 길이별 threshold 성능 분석 |
| `advanced_confidence_threshold_analysis/threshold_provenance.json` | threshold 선택 재현성 metadata |
| `phase1_mixed_emotion_predictions.csv` | Mixed Emotion 300개에 대한 Phase 1 prediction/confidence/routing 결과 |
| `confusion_matrix_mixed_phase1_distilbert.png` | Mixed Emotion Phase 1 confusion matrix |
| `mixed_phase1_confidence_distribution.png` | Mixed Emotion confidence 분포 |
| `distilbert_phase1_final_outputs.zip` | Phase 1 산출물 zip |

### 5.4 핵심 스키마

`phase1_mixed_emotion_predictions.csv`는 이후 모든 단계의 기준 파일이다. 필수 컬럼은 다음과 같다.

| 컬럼 | 설명 |
|---|---|
| `example_id` | Mixed Emotion sample ID. 이후 merge key |
| `text` | 입력 text |
| `target_label` | reference label |
| `phase1_label` | DistilBERT calibrated prediction |
| `phase1_confidence` | temperature-scaled MSP confidence |
| `phase1_probability_depression` | calibrated Depression probability |
| `phase1_probability_neutral` | calibrated Neutral probability |
| `phase1_probability_happy` | calibrated Happy probability |
| `phase1_accepted` | threshold 이상이라 Phase 1에서 accept된 경우 |
| `phase1_routed` | threshold 미만이라 Phase 2로 보내는 경우 |
| `routing_threshold` | calibration split에서 선택된 threshold |
| `temperature` | calibration split에서 선택된 temperature |

## 6. 02 LLM Phase 2 Reasoning 노트북

파일:

```text
notebooks/colab/final/02_llm_phase2_reasoning_final_colab.ipynb
```

### 6.1 입력

기본 입력은 01번에서 생성한 다음 파일이다.

```text
outputs_final/phase1_distilbert/phase1_mixed_emotion_predictions.csv
```

이 파일이 있으면 `phase1_routed = True`인 row만 골라서 Phase 2 reasoning을 실행한다. 파일이 없으면 prompt validation 용도로 Mixed Emotion 300개 전체를 사용할 수 있게 되어 있지만, 논문용 end-to-end 결과에는 반드시 Phase 1 파일을 사용해야 한다.

### 6.2 주요 파라미터

| 파라미터 | 의미 | 최종 권장값 |
|---|---|---|
| `RUN_ROUTED_ONLY` | routed sample만 reasoning할지 여부 | `True` |
| `RUN_LLAMA2_COT` | Llama 2 CoT 실행 | `True` |
| `RUN_LLAMA3_SELF_DISCOVER` | Llama 3 SELF-DISCOVER 실행 | `True` |
| `LLAMA2_MODEL_NAME` | Llama 2 model | `NousResearch/Llama-2-7b-chat-hf` |
| `LLAMA3_MODEL_NAME` | Llama 3 model | `NousResearch/Meta-Llama-3-8B-Instruct` |
| `RESUME_FROM_EXISTING` | 기존 CSV가 있으면 완료 row skip | `True` |

### 6.3 출력

| 출력 파일 | 설명 |
|---|---|
| `phase2_selected_input_rows.csv` | Phase 2로 실제 들어간 routed input |
| `llama2_cot_routed_mixed_emotion_results.csv` | Llama 2 CoT 결과 |
| `llama3_self_discover_routed_mixed_emotion_results.csv` | Llama 3 SELF-DISCOVER 결과 |
| `phase2_llm_reasoning_summary.csv` | Phase 2 standalone metric |
| `phase2_classification_reports.csv` | Llama 2/Llama 3 Phase 2 classification report |
| `phase2_predicted_label_counts.csv` | Llama 2/Llama 3 최종 label 분포 |
| `llama2_cot_confusion_matrix.csv/png` | Llama 2 Phase 2 confusion matrix |
| `llama3_self_discover_confusion_matrix.csv/png` | Llama 3 Phase 2 confusion matrix |
| `llama2_cot_parse_failures.csv` | Llama 2 final label parsing 실패 row |
| `llama3_self_discover_parse_failures.csv` | Llama 3 final label parsing 실패 row |
| `phase2_llm_reasoning_combined_outputs.csv` | Llama2/Llama3 결과를 합친 파일 |
| `phase2_llm_reasoning_outputs.zip` | Phase 2 산출물 zip |

### 6.4 Row-level 저장, resume, 자동 백업

Llama reasoning은 시간이 오래 걸리므로 각 row가 끝날 때마다 CSV에 append 저장한다. Colab이 중간에 끊기면 같은 노트북을 다시 실행했을 때 기존 CSV의 `example_id`를 읽고 완료된 row는 skip한다.

따라서 중간 종료 후 재실행 시 흐름은 다음과 같다.

```text
기존 CSV 읽기
-> 완료된 example_id 확인
-> pending row만 추출
-> pending row부터 이어서 실행
-> row마다 CSV append
```

또한 02번 노트북은 Final 01에서 생성된 `phase1_mixed_emotion_predictions.csv`의 fingerprint를 자동으로 계산한다. 현재 Phase 1 routed set이 기존 Phase 2 CSV와 같은 입력이면 그대로 resume하고, 입력이 달라졌으면 예전 CSV들을 자동으로 timestamped backup 폴더로 이동한 뒤 새 결과를 저장한다.

즉 v2.4 Mixed Emotion 결과로 새로 돌릴 때 사용자가 직접 예전 CSV를 삭제하거나 백업할 필요는 없다. 02번을 위에서부터 실행하면 다음 중 하나가 자동으로 처리된다.

```text
같은 Phase 1 routed set -> 기존 Llama CSV에서 완료 row 확인 후 이어서 실행
다른 Phase 1 routed set -> 기존 CSV 자동 백업 -> 새 Phase 2 run 시작
```

자동 백업 폴더 이름은 다음 형식으로 생성된다.

```text
outputs_final/phase2_llm_reasoning/backup_before_mixed_routed{N}_tau{threshold}_{hash}_{timestamp}/
```

현재 run 정보는 다음 파일에 저장된다.

```text
outputs_final/phase2_llm_reasoning/phase2_run_manifest.json
```

따라서 Final 02를 시작하기 전에 사용자가 해야 할 일은 없다. Final 01이 저장한 최신 `phase1_mixed_emotion_predictions.csv`가 Google Drive에 있으면, Final 02가 안전하게 현재 입력 기준으로 정리하고 실행한다.

## 7. 03 End-to-End Orchestration 노트북

파일:

```text
notebooks/colab/final/03_mixed_emotion_end_to_end_orchestration_final_colab.ipynb
```

### 7.1 입력

3번 노트북은 1번과 2번을 대신 실행하지 않는다. 아래 파일들이 이미 Google Drive에 있어야 한다.

| 입력 파일 | 생성 노트북 |
|---|---|
| `phase1_mixed_emotion_predictions.csv` | 01 DistilBERT |
| `llama2_cot_routed_mixed_emotion_results.csv` | 02 LLM reasoning |
| `llama3_self_discover_routed_mixed_emotion_results.csv` | 02 LLM reasoning |

즉 3번은 “실험 실행 파일”이라기보다 “결과 통합 및 논문용 산출물 생성 파일”이다.

### 7.2 최종 label 생성 규칙

End-to-end prediction은 다음 규칙으로 만든다.

```text
if phase1_routed == False:
    final_label = phase1_label
else:
    final_label = phase2_final_label
```

단, routed sample인데 Phase 2 결과가 없거나 parsing에 실패한 경우에는 fallback으로 `phase1_label`을 사용한다. 이 fallback 여부는 `final_source` 계열 컬럼에서 확인할 수 있다.

### 7.3 출력

| 출력 파일 | 설명 |
|---|---|
| `mixed_emotion_end_to_end_results.csv` | sample별 Phase 1, Phase 2, final prediction 전체 |
| `end_to_end_metrics_summary.csv` | Phase 1 only, routed only, end-to-end metric |
| `end_to_end_classification_reports.csv` | Phase 1, routed-only, end-to-end classification report |
| `end_to_end_confusion_matrices_long.csv` | 모든 confusion matrix를 long-format table로 정리한 파일 |
| `prediction_label_distributions.csv` | 각 모델/단계별 predicted label 분포 |
| `routing_coverage_table.csv` | accepted/routed count, coverage, routing rate |
| `phase2_correction_analysis.csv` | Phase 2가 고친 오류와 새로 만든 오류 |
| `paper_ready_tables.xlsx` | 논문 표로 옮기기 쉬운 Excel workbook |
| `confusion_matrix_phase1.png` | Phase 1 confusion matrix |
| `confusion_matrix_llama2_e2e.png` | Llama 2 end-to-end confusion matrix |
| `confusion_matrix_llama3_e2e.png` | Llama 3 end-to-end confusion matrix |
| `phase1_error_examples.csv` | Phase 1 오류 사례 |
| `llama2_e2e_error_examples.csv` | Llama 2 end-to-end 오류 사례 |
| `llama3_e2e_error_examples.csv` | Llama 3 end-to-end 오류 사례 |
| `mixed_emotion_end_to_end_paper_outputs.zip` | 최종 산출물 zip |

03번 노트북 마지막에는 `Final Paper Output Review` 섹션이 있다. 이 섹션은 저장된 파일을 직접 다시 열 필요 없이 Colab 화면에서 metric table, routing table, correction table, label distribution, classification report, confusion matrix table, confusion matrix image, error examples, output file list를 한 번에 확인하기 위한 최종 검토용 화면이다.

## 8. 논문에 들어갈 수 있는 결과

03번 노트북이 생성하는 결과 중 논문 본문 또는 appendix에 바로 활용할 수 있는 항목은 다음과 같다.

| 논문 위치 | 사용할 산출물 |
|---|---|
| Results | `end_to_end_metrics_summary.csv` |
| Results | `routing_coverage_table.csv` |
| Results | `phase2_correction_analysis.csv` |
| Results 또는 Appendix | confusion matrix PNG |
| Appendix | `paper_ready_tables.xlsx`의 confusion matrix sheet |
| Methodology | selected threshold, temperature, calibration metrics, risk-coverage curve, threshold provenance |
| Mixed Emotion section | Phase 1 only vs Llama2/Llama3 end-to-end 비교 |
| Primary Reddit Results | `reddit_test_end_to_end_metrics_summary.csv` |
| Primary Reddit Results | `reddit_test_routing_coverage_table.csv` |
| Primary Reddit Results | `reddit_test_phase2_correction_analysis.csv` |

## 8.1 04.5 Reddit 원문-input Routed Phase 2 노트북

파일:

```text
notebooks/colab/final/04_5_reddit_test_routed_phase2_original_text_primary_final_colab.ipynb
```

04.5 노트북은 Mixed Emotion이 아니라 primary Reddit held-out test set에 대한 최종 two-phase framework 평가를 만든다. 입력은 01번에서 저장한 다음 파일이다.

```text
outputs_final/phase1_distilbert/phase1_test_predictions.csv
```

이 파일에는 held-out Reddit test set 전체에 대한 DistilBERT Phase 1 prediction, calibrated confidence, accepted/routed 여부가 들어 있다. 04.5는 문서화된 클래스당 4,000건으로 test set을 정규화하고, primary `tau=0.70`에서 `phase1_routed=True`인 171건을 고정한다. 그 후 동일 source의 원문 `title + selftext`를 복원하여 Llama 2 CoT와 Llama 3 SELF-DISCOVER에 전달한다. `phase1_accepted=True`인 row는 Phase 2로 보내지 않고 Phase 1 label을 그대로 최종 label로 사용한다.

최종 label 생성 규칙은 다음과 같다.

```text
if phase1_routed == False:
    final_label = phase1_label
else:
    final_label = phase2_final_label
```

04번 노트북이 생성하는 주요 출력은 다음과 같다.

| 출력 파일 | 설명 |
|---|---|
| `reddit_test_primary_tau070_original_text_input_rows.csv` | Phase 2로 실제 들어간 원문-input Reddit 171건 |
| `reddit_test_primary_routed_original_text_mapping_audit.csv` | cleaned field와 원문 연결 감사 결과 |
| `reddit_test_llama2_cot_original_text_tau070_results.csv` | Reddit routed rows에 대한 Llama 2 CoT 결과 |
| `reddit_test_llama3_self_discover_original_text_tau070_results.csv` | Reddit routed rows에 대한 Llama 3 SELF-DISCOVER 결과 |
| `reddit_test_end_to_end_results.csv` | Reddit test 전체에 대한 Phase 1, Phase 2, final prediction |
| `reddit_test_end_to_end_metrics_summary.csv` | Phase 1 only, routed-only, Reddit end-to-end metric |
| `reddit_test_phase2_correction_analysis.csv` | routed rows에서 Phase 2가 고친 오류와 새로 만든 오류 |
| `reddit_test_routing_coverage_table.csv` | Reddit test 기준 coverage, routing rate, accepted/routed accuracy |
| `reddit_test_paper_ready_tables.xlsx` | Reddit test 결과를 논문 표로 옮기기 쉬운 workbook |
| `reddit_test_confusion_matrix_phase1.png` | Reddit test Phase 1 confusion matrix |
| `reddit_test_confusion_matrix_llama2_e2e.png` | Reddit test Llama 2 end-to-end confusion matrix |
| `reddit_test_confusion_matrix_llama3_e2e.png` | Reddit test Llama 3 end-to-end confusion matrix |
| `reddit_test_phase2_original_text_primary_tau070_outputs.zip` | Reddit test Phase 2 및 end-to-end 산출물 zip |

## 9. 실행 순서

최종 실행 순서는 다음과 같다.

1. Colab secret에 `WANDB_API_KEY`를 저장한다.
2. Colab GPU에서 `01_distilbert_phase1_training_final_colab.ipynb`를 실행한다.
3. `outputs_final/phase1_distilbert/phase1_mixed_emotion_predictions.csv`가 생성됐는지 확인한다.
4. Colab GPU에서 `02_llm_phase2_reasoning_final_colab.ipynb`를 실행한다.
5. Llama2/Llama3 routed 결과 CSV가 생성됐는지 확인한다.
6. `03_mixed_emotion_end_to_end_orchestration_final_colab.ipynb`를 실행한다.
7. `Final Paper Output Review` 섹션에서 metric, routing, correction, confusion matrix, 오류 사례를 화면에서 확인한다.
8. primary Reddit held-out test의 최종 two-phase 결과는 `04_5_reddit_test_routed_phase2_original_text_primary_final_colab.ipynb`를 실행한다.
9. `paper_ready_tables.xlsx`, confusion matrix PNG, summary CSV를 확인한다.
10. 논문 Results, Appendix, Limitation에 최종 수치를 반영한다.

## 10. 최종 확인 체크리스트

최종 실험을 논문 결과로 사용하기 전에 다음 항목을 확인한다.

| 확인 항목 | 기준 |
|---|---|
| Colab secret | `WANDB_API_KEY`가 저장되어 있어야 함 |
| Phase 1 완료 | `outputs_final/phase1_distilbert/phase1_mixed_emotion_predictions.csv` 생성 |
| Phase 2 완료 | `outputs_final/phase2_llm_reasoning/`에 Llama 2와 Llama 3 reasoning CSV 생성 |
| End-to-end 완료 | `outputs_final/end_to_end_orchestration/`에 metric/table/figure 산출물 생성 |
| Reddit test end-to-end 완료 | `outputs_final/reddit_test_phase2_reasoning_original_text_primary_tau070_final/`에 원문-input Reddit reasoning 및 end-to-end 산출물 생성 |
| 논문 반영 | 최종 accuracy, routing coverage, correction analysis, confusion matrix 수치를 Results와 Appendix에 반영 |

최종 실험은 `notebooks/colab/final/`의 노트북을 순서대로 실행하는 구조이다. 1번은 DistilBERT Phase 1과 threshold를 만들고, 2번은 Mixed Emotion routed sample에 Llama reasoning을 적용하고, 3번은 Mixed Emotion 결과를 합쳐 논문용 metric, table, figure를 생성한다. 4.5는 Reddit held-out test의 171개 routed sample을 원문으로 복원해 Llama reasoning을 적용하고 primary Reddit test 기준 최종 two-phase end-to-end 결과를 생성한다.
