# 기존 9,000건: 최종 Llama 3 세 방식 평가

## 확인된 분류 결과

2026-09-30에 저장된 1단계 Colab 출력에서 다음 결과를 확인했다.
아래는 노트북 출력의 집계값이며, 다음 코드가 원시 CSV와 handoff를 다시 대조한다.

| 항목 | 값 |
|---|---:|
| 전체 사례 | 9,000 |
| 클래스별 사례 | 3,000 |
| Phase 1 정확도 | 97.5778% |
| Phase 1 Macro F1 | 97.5821% |
| 전체 오류 | 218 |
| 라우팅 사례 | 111 (1.2333%) |
| 라우팅 사례 중 Phase 1 오류 | 48 |

218은 이번 9,000건의 전체 분류 오류 수이며, 이전 본 실험의 라우팅 218건과 다른 의미다.

## 다음 실행

[Colab: Llama 3 Direct / CoT / 최종 SELF-DISCOVER v6c](https://colab.research.google.com/drive/1WPr9Jh5H563GDVm3DYDTRQYX--GiN6Kv)

1. 새 GPU 런타임에서 위부터 모두 실행한다.
2. Drive 연결에 `jay.seizethemoment@gmail.com`을 선택한다.
3. 입력 확인 표에서 전체 9,000건, 라우팅 111건을 확인한다.
4. 세 방식 실행이 끝나면 마지막 비교표를 확인한다.

분류 재실행, 수동 업로드, 입력 경로 수정은 필요 없다. 첫 셀의 패키지 설치는 새 런타임에서도 실행한다. HF_TOKEN이 없어도 Drive 연결 실패로 처리하지 않는다.

## 그대로 유지한 부분

- Direct와 CoT의 프롬프트 및 실행 함수는 기존 최종 노트북에서 가져왔다. Direct는 1회, CoT는 기존 5턴 대화다.
- SELF-DISCOVER는 공동연구자의 v6c 실행 함수와 실제 캐시 계획 JSON을 사용한다. 새 계획 생성이나 프롬프트 탐색은 없다.
- 같은 모델 revision, 4-bit NF4 double quantization, fp16, greedy decoding을 사용한다.
- Direct/CoT의 호출별 상한 1,024토큰, v6c의 상한 1,536토큰은 기존 설정 그대로다. 세 방식의 총 호출 수나 총 토큰 예산이 동일하다고 주장하지 않는다.
- 입력은 라우팅된 글의 원문이며, 기존 6,000자 앞/뒤 보존 규칙을 적용하고 잘림 여부를 기록한다. 모델 문맥 한도를 초과하면 조용히 자르지 않고 멈춘다.
- 최종 라벨 파서는 기존 규칙을 유지한다. 해석하지 못한 결과는 Phase 1 라벨을 유지하며, 정답을 보고 복구하거나 별도 생성으로 답을 바꾸지 않는다.
- 전체 지표의 분모는 9,000건이다. 정답 라벨은 지표 계산에만 쓰며 생성 함수에 넘기지 않는다.

## 저장과 재개

입력 폴더:
`/content/drive/MyDrive/confidence_guided_llm_reasoning/outputs_final/unified_final/additional_9000_20260930/phase1`

출력 폴더:
`/content/drive/MyDrive/confidence_guided_llm_reasoning/outputs_final/unified_final/additional_9000_20260930/phase2/llama3_direct_cot_v6c_v1`

- `records/`: 완료된 사례별 결과.
- `stages/`: 실제 프롬프트, 생성 설정, 답변, 토큰 수, 시간. CoT 중간 호출도 저장한다.
- `llama3_reasoning_method_summary.csv`: 최종 세 방식 비교표.
- `llama3_{direct,cot,v6c}_results.csv`: 라우팅 사례별 원시 결과.
- `llama3_{direct,cot,v6c}_end_to_end_predictions.csv`: 전체 9,000건 최종 예측.
- 방법별 클래스 지표와 confusion matrix, 생성 비용 집계도 함께 저장한다.
- `frozen_llama3_v6c_plan.json`: 받은 최종 계획 원본.
- `experiment_manifest.json`: 입력 해시, 모델 revision, 소스·계획 해시, 생성 설정, 환경.

중단되면 같은 GPU 종류와 라이브러리 환경에서 다시 위부터 실행한다. 완료된 사례와 생성 호출을 재사용한다. 입력·환경·설정이 바뀌면 결과를 섞지 않고 중단한다. 기존 12,000건 및 Mixed Emotion 결과는 변경하지 않는다.

## 준비 검증과 남은 확인

- 로컬 테스트 6개 통과: 공통 계획 해시, 모델 revision 고정, 1/5/1회 호출 구조, 생성 상한, 파싱 실패 fallback, 호출 재개, manifest 불일치 차단, 9,000건 handoff 검증.
- 노트북 Python 셀의 문법 검사 완료.
- 실제 Llama GPU 생성은 아직 실행하지 않았다. 결과가 모두 나온 뒤 원시 로그를 대조하고 대응 통계검증을 수행한다.
- 이번 데이터는 이전 평가 이력이 있는 동일 출처의 9,000건이다. 이번 최종 프로토콜로 추가 평가하는 것이며, 새로 확보한 미사용 평가셋으로 표현하지 않는다.
