# Phase 1 최종 성능·GPU 효율성 반영 보고서

작성일: 2026-09-11

## 1. 이번 버전의 위치

공동연구자가 완료한 2026-09-11 분류기·GPU 효율성 실험을 정식 연구 결과로 원고에 반영했다. 기준 원고는 레퍼런스 검토가 반영된 `Paper_20260909_reference_verified/main.tex`이다. 기존 연구의 핵심은 유지하고, Phase 1 모델 선택을 정확도와 실제 추론 비용으로 함께 뒷받침하도록 강화했다.

이번 결과와 현재 별도로 진행 중인 9,000건 추가 평가는 다른 실험이다. 추가 평가의 137건 라우팅, 새로운 성능·통계·생성 비용 등은 이 원고에 포함하지 않았다. 해당 실험이 완료된 뒤 별도 단계로 검증·반영한다.

결과 기준 문서:
- `phase1_gpu_efficiency_benchmark_results_ko.md`: 최종 측정 결과와 반복 측정 요약.
- `paper_efficiency_insertion_plan_ko.md`: accuracy와 macro F1의 개별 CI, 오류 구성, 설정값.

## 2. 변경 전후: 분류 성능

모두 동일한 12,000건 test set에 대한 3개 학습 시드(42, 43, 44)의 결과이다. 다음 mean ± SD는 accuracy와 macro F1에서 소수 둘째 자리까지 동일하며, CI는 지표별로 별도 반영했다.

| 모델 | 이전 mean ± SD (%) | 이번 mean ± SD (%) | 이번 accuracy 95% CI (%) | 이번 macro F1 95% CI (%) |
|---|---:|---:|---|---|
| DistilBERT | 96.90 ± 0.12 | 96.88 ± 0.04 | [96.79, 96.98] | [96.79, 96.98] |
| Mistral 7B | 95.56 ± 0.11 | 95.36 ± 0.08 | [95.15, 95.57] | [95.16, 95.56] |
| Llama 2 7B | 95.09 ± 0.06 | 95.17 ± 0.07 | [95.00, 95.34] | [94.99, 95.34] |

- DistilBERT와 Mistral의 평균 accuracy 차이: 1.34 → 1.52 percentage points.
- DistilBERT와 Llama 2의 차이: 1.81 → 1.71 percentage points.
- Mistral과 Llama 2의 차이: 0.47 → 0.19 percentage points.
- 모델 비교 CI는 학습 시드 사이의 Student-t 95% CI이다. 타이밍 3회 반복의 SD와 혼용하지 않는다.
- CI 중첩만으로 두 probe의 차이가 유의하지 않다고 결론내리지 않는다. 높은 평균 점수를 보고하되, 두 probe 간 우열을 연구의 결론으로 삼지 않았다.

### 실제 문장 변경 예

수정 전:
> Under this fixed comparison protocol, DistilBERT exceeded Mistral and Llama 2 by 1.34 and 1.81 percentage points in mean accuracy, respectively.

수정 후:
> Under this fixed comparison protocol, DistilBERT exceeded Mistral and Llama 2 by 1.52 and 1.71 percentage points in mean accuracy, respectively. The two probes differed by 0.19 points; their relative ordering is not the basis of the operational choice.

## 3. 변경 전후: 모델 선택 근거

이전에는 높은 평균 점수와 작은 파라미터 수를 주로 제시했다. 이번에는 같은 A100-SXM4-40GB 환경에서 측정한 처리량·단건 지연·메모리를 정량 근거로 추가했다.

| 추론 지표 | DistilBERT | Mistral 7B | Llama 2 7B |
|---|---:|---:|---:|
| 처리량, posts/s | 4,959.5 ± 15.5 | 51.7 ± 0.0 | 55.5 ± 0.0 |
| 단건 p50, ms | 4.80 ± 0.03 | 36.62 ± 0.06 | 34.05 ± 0.10 |
| 단건 p95, ms | 5.11 ± 0.12 | 38.37 ± 0.18 | 36.23 ± 0.40 |
| 배치 추론 peak memory, GB | 0.29 | 15.22 | 14.05 |

Table 1에 (a) 성능 / (b) 효율성 패널을 두어 동일한 세 모델을 함께 비교한다. 처리량·p50·배치 추론 메모리는 본문 표, p95와 나머지 자원 항목은 부록에 제시했다. 새로운 독립 본문 표를 추가하지 않아 Table 2~7 번호는 유지된다.

새 본문 문장:
> On the common A100 benchmark, DistilBERT processed 4,959.5 posts/s, giving 89.4- and 95.9-fold higher batched throughput than the Llama 2 and Mistral probes.

이 수치는 배치 처리량 비교이다. 단건 p50 지연은 4.80 대 34.05/36.62ms로 따로 제시했다. 따라서 “모든 상황에서 90배 빠르다” 또는 “전체 캐스케이드가 90배 효율적이다”로 확대하지 않는다.

## 4. 변경 전후: 설정과 오류 구조

| 모델 | learning rate 전 → 후 | batch 전 → 후 | epochs 전 → 후 | weight decay 전 → 후 |
|---|---|---|---|---|
| DistilBERT | 5.717e-5 → 4.037e-5 | 32 → 64 | 2 → 3 | 1e-3 → 1e-3 |
| Llama 2 7B | 2.824e-4 → 4.151e-4 | 64 → 32 | 3 → 3 | 1e-4 → 1e-4 |
| Mistral 7B | 2.292e-4 → 2.399e-4 | 16 → 64 | 2 → 2 | 1e-2 → 1e-4 |

Appendix A1c는 간결한 열 이름을 유지했다. Epochs는 스윕에서 선택된 학습 예산이며, 각 시드에서 반드시 마지막 epoch를 채택했다는 의미가 아니다. 실제 retained epoch는 validation macro F1에 따라 결정된다는 기존 설명을 유지했다.

Depression-Happy 혼동 비중은 Llama 2 75.4% → 75.6%, Mistral 74.2% → 74.3%로 갱신했고 DistilBERT 72.8%를 추가했다. Neutral F1 > 98%는 세 모델 모두에 해당한다. 합산된 36,000개 예측을 36,000개의 독립 test example로 취급하지 않는다는 기존 설명도 유지했다.

## 5. 그림 변경

Figure 2만 재생성했다.
- 평균 막대: 96.90 / 95.56 / 95.09 → 96.88 / 95.36 / 95.17.
- accuracy와 macro F1의 CI를 각각 사용해 오차 막대 갱신.
- 기존 94.5~97.5% 축, 모델 순서, 색상, Times 계열 폰트, 두 지표 구성 유지.
- SVG와 PDF 모두 실제 텍스트를 보존. 그림을 래스터 이미지로 변환해 원고에 넣지 않았다.
- 재생성 스크립트와 사용한 결과 JSON/CSV를 함께 보관.

Figure 1 아키텍처와 편집 가능한 draw.io, Figure 3~6, SELF-DISCOVER 부록 그림은 변경하지 않았다.

## 6. 부록에 추가한 내용

Appendix A1d는 다음 세 패널로 구성했다.

1. 구축 비용: feature extraction, 4-trial search, 3-seed fitting, 총 GPU-h, extraction/fitting peak memory.
2. 반복 추론: 배치 시간, 단건 p95, 단건 메모리, 1,000건당 GPU-h, 배치 에너지.
3. 전체 측정 실행: 평가·벤치마크 반복을 포함한 총 GPU 시간, GPU-board 에너지, 배치 GPU 사용률.

구축 총액은 DistilBERT 0.59, Llama 2 0.68, Mistral 0.70 GPU-h이다. 7B 모델의 head만 학습하는 비용을 DistilBERT 전체 학습과 비교하지 않고, backbone feature extraction을 합산했다. 전체 측정 실행 총액 0.61/0.87/0.91h는 반복 벤치마크까지 포함하므로 구축 비용 및 실제 배포 비용과 구별했다.

프로토콜에는 A100-SXM4-40GB, bf16 학습/fp16 추론, 256-token cap, batch 32, 12,000건 배치 평가, 512건 단건 평가, warm-up 3 batches, 반복 3회, CUDA 동기화, host-to-device 전송 포함, NVML 1Hz, GPU-board 전력 적분을 기재했다. 에너지는 GPU board 범위이며 짧은 측정 창의 시간 해상도 제한도 명시했다.

## 7. 절별 변경 위치

| 위치 | 수정 전 | 수정 후 |
|---|---|---|
| Abstract | 이전 성능 수치, 효율 실측 언급 없음 | 최신 수치 및 common-A100 처리량·지연 우위 한 문장 |
| Study Contributions | Phase 1의 효율성을 파라미터 규모 중심으로 설명 | 실제 처리량·지연·메모리를 모델 선택 근거에 포함 |
| VI-A Computing Environment | 통제된 latency benchmark가 없다고 서술 | Phase 1 공통 하드웨어 벤치마크 및 측정 조건 설명 |
| VI-F Evaluation Metrics | 규모·학습 범위 중심 | 추론 비용 및 타이밍 반복의 변동성 추가 |
| VII-A / Table 1 | 기존 정확도 및 CI | 새 정확도·CI + 효율성 패널 |
| VII-A / Figure 2 | 기존 막대와 CI | 최신 수치로 벡터 재생성 |
| VII-A 결과 해석 | 기존 점수 차이와 오류 비중 | 새 차이·오류 비중, 처리량·지연·구축 비용 설명 |
| VII-G RQ 해석 | 이전 점수, 비용 미측정 | 최신 점수, Phase 1 측정과 Phase 2 미측정의 구별 |
| VIII Limitations | GPU-time/cost 비교 전무 | 공통 하드웨어 실측 범위와 Phase 2 비용 미측정 설명 |
| Conclusion | 성능 비교로 선택 | 성능과 실측 효율을 함께 선택 근거로 요약 |
| Appendix A1c | 이전 선택 설정 | 새 learning rate/batch/epochs/weight decay |
| Appendix A1d | 없음 | 구축·반복 추론·총 측정 자원 및 프로토콜 |

각 문단의 실제 LaTeX 수정 전후 전문은 `SENTENCE_CHANGES_KO.md`, 전체 소스 차이는 `MANUSCRIPT_FULL_DIFF.md`에 정리했다.

## 8. 유지한 것과 다음 단계

- Operational DistilBERT 96.69%, 기존 하이퍼파라미터, 온도 및 임계값은 유지했다.
- Reddit 라우팅 171건, 오류 397건 중 포착 87건, Llama 3 순개선 30건 등 기존 downstream 결과는 유지했다.
- 기존 calibration·routing·end-to-end 표와 아키텍처/draw.io 파일은 동일하다.
- 레퍼런스 46개와 직전 레퍼런스 교정 내용은 유지했다.
- 전체 GPU 시간·비용 절감률을 라우팅률로 대신하지 않았다.
- 저자·소속·교신·펀딩 및 제출 전 행정 항목은 기존 placeholder 상태이다. 이 버전은 이번 실험 결과 반영을 완료한 검토 원고이며, 행정 항목까지 채워진 제출 파일이라고 표기하지 않는다.

다음 업데이트는 진행 중인 9,000건 평가 완료 후의 실제 성능, 대응 통계, 오류 교정, 생성량·시간을 별도 실험으로 반영하는 단계이다. 그 결과가 나오기 전에는 새 실험을 완료했다고 원고에 쓰지 않는다.
