# 논문 업데이트까지 남은 작업

2026-09-27 | 공동연구자 검토용

## 현재 위치

**계획한 모델 2개 × 방법 3개 × 데이터셋 2개의 실행은 완료했다.** Llama 2 CoT 생성 상한 재실행과 제공된 파싱 실패 응답 검토도 끝났다. 이제 기본 순서는 **최종 파일 정리 → 일괄 재집계 → 통계 검증 → 논문 및 그림 교체 → 제출 전 검수**다. 새로운 모델이나 프롬프트 실험을 무조건 추가하는 단계는 아니다.

- [최신 결과표](FINAL_RESULTS_UPDATE_20260925_KO.md)
- [최종 노트북 안내](../../notebooks/colab/final/README.md)

## 우선순위별 필수 작업

| 순서 | 할 일 | 구체적인 내용 | 완료 기준 |
|---|---|---|---|
| 1 | 최종 결과 파일 모으기 | 12개 조건의 원시 생성 결과, 전체 예측, Phase 1 입력, manifest 및 환경 이력을 한 목록으로 연결한다. 특히 최신 Llama 2 CoT 양쪽 원시 파일을 확보·대조한다. | 데이터·모델·방법·경로·해시·행 수를 적은 파일 목록 완성 |
| 2 | 동일 규칙으로 재집계 | 사후 복구한 라벨을 재현 가능한 파서/검토 기록으로 만들고 모든 조건에 동일한 판정 원칙을 적용한다. 정답을 기준으로 복구 여부를 정하지 않는다. 기존 원시 파일은 보존한다. | 최종 예측 CSV, 복구 전후 기록, 비교표가 하나의 스크립트로 재생성됨 |
| 3 | 통계 검증 | Phase 1 대비 개선, SD와 Direct·CoT 차이, 같은 방법에서 모델 간 차이를 구분해 검정한다. 비교군과 다중비교 보정 범위를 먼저 문서화한다. | 효과 크기·95% CI·원 p값·보정 p값 표 완성 |
| 4 | 설정·로그 정리 | 최종 모델·seed·보정값·threshold, 프롬프트, 최대 생성량, 실제 상한 조정, 잘림·실패·fallback 및 환경 변경을 정리한다. | 본문은 핵심 설정, Appendix/저장소는 상세 재현 정보로 역할 분담 |
| 5 | 논문 본문·Appendix 교체 | 최종 DistilBERT에서 routing까지 연결된 구조, 두 모델·세 방법 비교, 최종 SD 프롬프트 및 결과로 업데이트한다. | 초록·방법·결과·논의·결론·Appendix가 동일한 실험을 설명 |
| 6 | 표·그래프·아키텍처 갱신 | 수치는 확정 CSV에서 생성한다. 아키텍처는 수정할 박스·문구·화살표 목록을 먼저 만들고 원본 drawio에 반영한다. | PDF/SVG와 편집 원본이 일치하고 그림 속 글자·수치가 본문과 일치 |
| 7 | 최종 검수·공유 | PDF 페이지, 공백, 표 크기, 글자 겹침, 수식·참조·옛 숫자를 점검한다. 전후 변경 문서를 작성한다. | 최종 PDF, 원고 소스, 그림 원본, 결과·통계 코드, 리뷰 문서를 Git에 정리 |

### 1번에서 모을 파일

Drive 공통 루트는 `MyDrive/confidence_guided_llm_reasoning/outputs_final/unified_final/`다.

| 데이터 / 조건 | 공통 루트 아래 폴더 |
|---|---|
| Reddit / Llama 2 Direct·최종 SD | `phase2/llama2_direct_final_sd_v1/` |
| Reddit / Llama 2 CoT | `phase2/llama2_cot_matched_1024_v1/` |
| Reddit / Llama 3 Direct·CoT | `phase2/llama3_reasoning_comparison/` |
| Reddit / Llama 3 최종 SD | `reddit_sd_exploration/phase2/llama3_sd_independent_round2_v1/` |
| Mixed / Llama 2 Direct·최종 SD | `mixed_emotion/phase2/llama2_direct_final_sd_v1/` |
| Mixed / Llama 2 CoT | `mixed_emotion/phase2/llama2_cot_matched_1024_v1/` |
| Mixed / Llama 3 Direct·CoT | `mixed_emotion/phase2/llama3_reasoning_comparison/` |
| Mixed / Llama 3 최종 SD | `mixed_emotion/phase2/llama3_sd_independent_countercheck_v1/` |

각 폴더의 `*_results.csv`, `*_end_to_end_predictions.csv`, 요약 CSV와 존재하는 manifest/환경 이력을 모은다. Llama 3 기존 comparison 폴더의 옛 SD와 round2 폴더의 neutral_plan은 최종 비교 대상이 아니다. 이미 확보된 파일은 다시 생성하지 말고 파일 목록에서 대조한다.

### 2번에서 확인할 것

- Reddit 전체 12,000건/동일 routed 218건, Mixed 전체 300건/동일 routed 86건의 ID·정답·Phase 1 예측 일치 여부.
- 전체 평가 정확도·macro precision/recall/F1·클래스별 F1·혼동행렬·수정/신규 오류/순수정.
- 복구할 수 없는 응답도 삭제하지 않고 Phase 1 fallback을 적용한다. 이미 검토한 복구 사례와 미확정 사례를 그대로 추적한다.
- 사후 복구 결과가 모든 노트북의 자동 파서에 아직 통합된 것은 아니다. 대화 속 최종 숫자와 자동 summary가 다른 이유를 명확히 남긴다.

### 3번 통계 검증의 질문

| 질문 | 비교 |
|---|---|
| 선택적 재평가로 전체 성능이 개선됐는가? | 각 최종 파이프라인 vs 동일 Phase 1 |
| 최종 SD가 비교 방법보다 높은가? | 같은 모델·데이터의 SD vs Direct, SD vs CoT |
| 모델에 따라 결과가 달라지는가? | 같은 방법·데이터의 Llama 3 vs Llama 2 |

정확도 차이는 사례별 정오를 짝지어 exact McNemar 검정하고, paired bootstrap으로 정확도·macro F1 차이의 CI를 계산한다. Bootstrap에서는 두 시스템에 동일한 사례 인덱스를 적용한다. 보정할 비교군은 결과를 보고 유리하게 나누지 않고 분석 전에 정한다. seed·반복 횟수·보정 범위를 저장한다. 생성 반복을 한 경우가 아니라면 사례 단위 CI를 생성 변동성의 CI로 설명하지 않는다.

## 논문에서 특별히 빠뜨리지 않을 교체 항목

- 별도 operational 모델 설명을 최종 학습 모델과 연결된 절차로 교체하되, 모델 체크포인트 자체의 재현 정보는 남긴다.
- 이번 최종 모델의 temperature·threshold·coverage·routing rate·selective risk·error capture를 사용한다. 과거 171건/44건 routing 분석을 그대로 가져오지 않는다.
- Phase 2 중심 비교를 Llama 3의 세 방법으로 정리하고, Llama 2의 동일 방법 비교를 함께 제시한다. 모든 비교 결과를 보존한다.
- 최종 SD의 독립 판단 및 반대 근거 점검, Direct/CoT의 Phase 1 라벨 제공 방식, 실제 생성 상한 정책을 정확히 기술한다.
- 최종 결과 기준으로 통계, 오류 사례, accepted-error audit, routing enrichment 및 oracle 상한을 다시 계산한다.
- 과거 추가 9,000건 holdout 및 효율성 측정이 현 모델/프롬프트와 같은지 확인한다. 다르다면 새 최종 시스템 결과로 섞지 않는다. 해당 근거를 유지하려면 재실행 필요 여부를 결정하고, 그렇지 않으면 과거 조건의 보조 결과로 구분하거나 제외한다.
- SD 후보 개발·선정 이력을 보존하고, 관측 최고값과 독립적인 일반화 근거를 구분한다. 순위가 가장 높다는 사실과 통계적으로 유의한 차이는 별개다.

## 선택적 보강: 주장 범위에 따라 결정

| 항목 | 필요한 경우 | 현재 우선순위 |
|---|---|---|
| 고정한 최종 파이프라인의 별도 미사용 데이터 평가 | 후보 선정 데이터와 독립적인 일반화 성능을 강하게 주장하려는 경우 | 주장 범위에 따라 우선 결정 |
| Mixed Emotion 및 대표 사례의 복수 연구자 검토 | 라벨 타당성과 사례 해석 근거를 강화하려는 경우 | 권장, 현재 검토 수준 확인 후 범위 결정 |
| 최종 파이프라인 GPU 시간·토큰·호출 수 집계 | end-to-end 비용·속도 우위를 주장하려는 경우 | 기존 로그로 가능한 부분부터; Phase 1 효율성과 구분 |
| 동일 라벨 제공 조건의 추가 비교 | SD 전체 절차가 아니라 구조 자체의 효과를 분리하려는 경우 | 현재 방법 전체 비교에는 추가 필수 아님 |
| 반복 생성·환경 통제 재실행 | 실제 로그에서 환경 차이의 영향이나 재현 문제가 확인되는 경우 | 일괄 재실행하지 않고 로그 확인 후 결정 |

## 바로 다음에 할 일

**먼저 파일 목록과 일괄 재집계 코드를 확정한다. 그 결과로 통계를 계산한 뒤 본문·그림을 바꾼다.** 이 순서면 숫자가 바뀔 때마다 논문 전체를 반복 수정하는 일을 줄일 수 있다. 이 문서는 할 일 정리이며, 통계 계산이나 논문 수정이 이미 완료됐다는 뜻은 아니다.
