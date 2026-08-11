# Reddit Risk-Budget Policy Comparison

## 목적

Reddit held-out test에서 동일한 DistilBERT, 동일한 temperature scaling, 동일한 Llama 2 CoT v2 및 Llama 3 SELF-DISCOVER를 사용하면서, calibration risk budget에 따른 two-phase routing 정책을 비교한다.

이 실험은 모델을 다시 학습하지 않는다. Final 01이 저장한 `phase1_test_predictions.csv`와 calibration threshold table을 입력으로 사용한다.

## 비교 정책

각 alpha 정책은 Final 01의 calibration split에서만 선택된다. Reddit test의 정답이나 LLM end-to-end 결과는 threshold 선택에 사용하지 않는다.

\[
\tau^*(\alpha) = \arg\max_{\tau} \operatorname{Coverage}(\tau)
\quad \text{subject to} \quad
\operatorname{UpperBoundRisk}(\tau) \le \alpha.
\]

후보 threshold는 0.70부터 1.00까지 0.01 간격이다. 현재 calibration artifact에서는 다음이 선택된다.

| Risk budget | Calibration-selected threshold | 의미 |
|---:|---:|---|
| `alpha=0.05` | `tau=0.70` | accepted-set risk upper bound를 5% 이하로 제한하는 기본 효율 우선 정책 |
| `alpha=0.025` | `tau=0.80` | accepted-set risk upper bound를 2.5% 이하로 더 엄격히 제한하는 보수적 정책 |

## 실행 노트북

`notebooks/colab/final/04_4_reddit_test_alpha_policy_comparison_final_colab.ipynb`

이 노트북은 먼저 `phase1_confidence < 0.80`인 Reddit held-out test union을 한 번만 Llama reasoning으로 처리한다. 따라서 동일한 Llama 출력에서 nested subset을 재사용할 수 있다.

| Threshold | Routed rows (current test artifact, n=12,001) |
|---:|---:|
| 0.70 | 171 |
| 0.75 | 233 |
| 0.80 | 311 |

## 생성 산출물

Google Drive의 `outputs_final/reddit_test_phase2_alpha_policy_comparison_final/`에 다음 파일을 저장한다.

- `reddit_test_llama2_cot_routed_results.csv`: row-level resumable Llama 2 outputs
- `reddit_test_llama3_self_discover_routed_results.csv`: row-level resumable Llama 3 outputs
- `calibration_risk_budget_policy_selection.csv`: alpha별 calibration-only threshold 선택 근거
- `reddit_test_risk_budget_policy_end_to_end_summary.csv`: Phase 1, Llama 2, Llama 3의 alpha별 전체 test 성능 비교
- `reddit_test_risk_budget_policy_predictions.csv`: alpha별 sample-level final predictions
- `reddit_test_risk_budget_policy_paper_table.csv`: 논문 표 작성용 compact summary

## 해석 원칙

- `alpha=5%`와 `alpha=2.5%`는 서로 다른 모델이 아니라 같은 모델의 서로 다른 routing operating policy다.
- 둘 중 held-out 결과가 높은 정책은 `best observed operating configuration`으로 기술할 수 있다.
- 최종 배포 정책을 하나로 고정하려면, 이 비교 결과를 본 뒤에는 새 held-out split 또는 outer-fold에서 선택된 정책을 재현하는 것이 가장 엄격하다.
