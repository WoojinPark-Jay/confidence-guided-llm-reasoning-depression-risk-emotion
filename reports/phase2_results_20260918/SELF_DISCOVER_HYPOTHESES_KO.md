# Reddit SELF-DISCOVER 개선 가설 실험

## 목적

동일한 최종 Phase 1 출력의 Reddit routed 218건에서 SELF-DISCOVER의 약점을 탐색한다. 기존 원문 실험, 정제 텍스트 실험, Mixed Emotion 결과는 변경하지 않는다. 이번 비교는 이미 결과를 확인한 Reddit 평가 사례에서 수행하는 프롬프트 개발 실험이며, 독립적인 최종 일반화 평가로 해석하지 않는다.

## 세 가지 변형

| 코드 이름 | 가설 | 변경 | 유지 |
|---|---|---|---|
| `sd_independent` | 먼저 보이는 Phase 1 라벨이 판단을 고정하는가? | 모든 단계에서 Phase 1 라벨과 비교 요청 제거 | 기존 감정 정의와 마지막 감정 흐름 지침 |
| `sd_evidence` | 마지막 긍정 표현이나 표면적인 단어가 전체 문맥을 덮는가? | 마지막 감정 흐름 우선 규칙을 전체 문맥, 화자, 부정, 비유, 실현된 감정 근거 중심 지침으로 교체 | Phase 1 라벨 공개, 기존 모듈과 4단계 구조 |
| `sd_compact` | 범용 모듈에서 생성한 긴 계획이 불필요한 해석을 늘리는가? | 모듈/계획 최대 3개, 간결한 실행 설명 요청 | 기존 감정 지침, Phase 1 라벨 공개 |

세 변형은 서로 합치지 않는다. 근거 집중형은 관련 지침을 묶어 변경하므로 개선되더라도 개별 문구 하나의 효과로 단정하지 않는다. 모듈 수와 단어 수는 프롬프트 요청이며 강제 파서 제한은 아니다. 구조 생성과 최종 응답을 모두 저장해 실제 준수 여부를 볼 수 있다.

기존 코드에서 Phase 1 라벨이 모든 단계에 들어가는 점, 마지막 감정 흐름 강조, 범용 모듈 39개를 확인했다. 저장된 오답 일부에서도 긍정 단어의 문자적 해석과 Phase 1 라벨 유지가 관찰됐지만, 전체 오류의 원인이라는 결론은 아직 없다.

## 실행

[Colab 열기](https://colab.research.google.com/github/WoojinPark-Jay/confidence-guided-llm-reasoning-depression-risk-emotion/blob/main/notebooks/colab/final/08_reddit_llama3_self_discover_hypotheses_colab.ipynb)

1. 기존 Drive 계정에서 GPU 런타임으로 연다.
2. 설치 셀 실행 후 런타임 재시작, imports부터 순서대로 실행한다.
3. 기존 `unified_final/phase1/final_phase1_reasoning_input.csv`를 자동으로 읽는다. 추가 업로드는 필요 없다.
4. 기본 설정으로 세 버전을 순차 실행한다. 218건 × 3버전 × 4단계 = 2,616회 생성이며, 수 시간이 걸릴 수 있다. 완료 사례는 행 단위로 저장되어 같은 설정으로 재개할 수 있다.

별도 저장 위치: `outputs_final/unified_final/reddit_sd_exploration/phase2/llama3_sd_hypotheses_v1/`.

동일 Llama 3 revision, 4-bit NF4, greedy decoding, 단계별 최대 1,024토큰, 기존 원문 6,000자 head/tail 정책을 유지한다. 미추출 라벨은 실패로 기록하고 Phase 1 예측을 유지한다. 생성 함수에는 정답 라벨을 전달하지 않는다. 입력 해시와 설정 manifest가 달라지면 재개를 중단한다. 짧은 smoke test는 반드시 다른 출력 폴더를 사용한다.

`RUN_METHODS`에 `sd_baseline`을 추가하면 기존 SELF-DISCOVER 프롬프트도 같은 런타임에서 다시 실행할 수 있다. 기본값에는 넣지 않아 기존 결과의 불필요한 재생성을 줄였다. 패키지 설치는 기존 노트북 방식을 유지하므로 과거 실행 환경과 완전히 같다는 보장은 없다.

## 결과 검토 기준

- 전체 12,000건 정확도와 macro F1, 수정/신규 오류/순수정으로 비교한다. 생성된 218건만의 정확도와 혼동하지 않는다.
- 기존 원문 결과는 Direct +23건, CoT +28건, SELF-DISCOVER +15건 순수정이다. 세 변형 모두 같은 fallback 기준으로 비교한다.
- 라벨 미추출, 입력/생성 잘림, 단계별 응답과 토큰 수를 확인한다. 높은 점수만 보고 원인이 검증됐다고 결론 내리지 않는다.
- 이번에는 Mixed Emotion을 실행하지 않는다. Reddit에서 후보를 정리한 뒤, 다른 데이터에서의 안정성은 별도로 확인한다.

로컬에서는 구문과 모의 생성 기반 4단계 실행, 라벨 비공개, 로그 합산을 테스트했다. 실제 GPU 생성과 성능 향상은 아직 확인되지 않았다.
