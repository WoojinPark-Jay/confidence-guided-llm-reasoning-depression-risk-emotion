# 기존 아키텍처 그림 최소 수정 가이드

논문 Figure 1은 기존 원본 PDF로 복원했다. 원본 drawio는 수정하지 않았다.
전체 3열 배치, 데이터 분할, accept/route 구조는 그대로 유지하면 된다.
아래 수정은 사용자가 drawio에서 진행할 부분이다. 현재 그림에 반영된 것으로 보아서는 안 된다.

## 1. Phase 1 model development 박스

기존 `as frozen operational classifier`를 선택된 설정의 최종 모델이라는 뜻으로 바꾼다.
별도 비교용/운영용 모델처럼 읽히지 않도록 아래 문구로 교체하면 된다.

```text
Phase 1 model development
train + validation: Bayesian search
DistilBERT: selected configuration
final trained model used downstream
Llama 2 / Mistral comparators
```

바로 아래 `Frozen Phase 1 inference / DistilBERT logits`는 그대로 둬도 된다.
여기서 frozen은 학습 후 추론 중 가중치를 바꾸지 않는다는 뜻이지, 별도 모델이라는 뜻이 아니다.

## 2. Phase 2 re-evaluators 박스: 가장 중요한 교체

현재 Llama 2 CoT와 Llama 3 SELF-DISCOVER만 적힌 부분을 다음으로 바꾼다.

```text
Phase 2 re-evaluators
Llama 2 / Llama 3
Direct / CoT / SELF-DISCOVER
SD: task-level plan reused
one configuration per run
no ensemble
```

의미: 두 모델 각각 세 방법을 비교하며, 최종 SD는 과제 수준의 계획을 한 번 생성해 재사용한다.
`v6c`, 토큰 상한, 세부 모듈 수, 실제 정확도는 그림에 넣을 필요 없다. 본문/부록에서 설명한다.
기존처럼 SELECT → ADAPT → IMPLEMENT를 원문 입력 뒤의 매 사례 과정으로만 보이게 적으면
이번 최종 SD와 달라지므로, 작은 박스에 모두 넣으려 하지 말고 위의 재사용 문구로 정리하는 것을 추천한다.

선택 사항: 공간이 충분하면 작은 보조 박스 하나에 다음 문구를 넣고 Phase 2 박스로 연결한다.

```text
SELF-DISCOVER task-level plan
SELECT → ADAPT → IMPLEMENT
cached and reused across posts
```

이 보조 박스는 routed text의 필수 직렬 단계가 아니라, SD 실행에 계획을 제공하는 별도 연결이다.
Direct와 CoT까지 이 계획을 쓰는 것처럼 연결하지 않는다.

## 3. End-to-end prediction 박스

파싱 실패 시 Phase 1을 유지하는 실제 집계 규칙을 한 줄 추가한다.

```text
End-to-end prediction
accepted: retain Phase 1 label
routed: use parsed Phase 2 label
parse failure: retain Phase 1 label
```

파싱 실패를 자동 오답 처리한다는 뜻이 아니다. 유지된 Phase 1 예측과 정답을 비교해 평가한다.
`Parsed Phase 2 label`과 최상단 `Final label selection` 박스는 그대로 둬도 된다.

## 4. Routing and end-to-end evaluation 박스

`Corrected counts` 한 줄만 수정·신규 오류·순수정 모두를 드러내도록 바꾼다.

```text
Accuracy and macro precision/recall/F1
Coverage and routing rate
Selective risk and error capture
Corrected, introduced, net corrections
Confusion matrices
```

혼동행렬은 평가 항목으로 유지할 수 있다. 다만 최종 Phase 2 행별 자료가 확보되기 전에는
최종 Phase 2 혼동행렬까지 이미 완성됐다고 본문에서 주장하지 않는다.

## 5. Statistical and audit artifacts 박스

출력 PDF의 마지막 줄에 `Acepted-error audit` 오탈자가 보여 `Accepted-error audit`로 고친다.
현재 최종 모델의 accepted-error 사례 검토가 끝나지 않았으므로, 검토 완료 산출물로 오해되지 않게
현 단계에서는 더 정확한 `Accepted-error records`라는 표현을 추천한다.

```text
Statistical and audit artifacts
Routed-case output records
Paired bootstrap confidence intervals
Exact McNemar test
Holm correction
Accepted-error records
```

## 6. 연결선 두 개만 보강 권장

- `Phase 1 model development` → `Frozen Phase 1 inference`: 선택된 설정으로 학습한 동일 모델을 추론에 사용한다는 연결. 라벨은 `final trained model`.
- `Calibration and routing policy` → `Calibrated confidence`: 온도 T*를 전달하는 점선. 라벨은 `fixed T*`.

이미 있는 calibration → 판정 마름모의 `FIXED τ*` 점선은 유지한다.
원문의 실선 데이터 흐름과 충돌하지 않도록 모델/설정 전달선은 빈 공간으로 우회한다.

## 그대로 두는 부분

- 120,000 Reddit posts와 70/10/10/10 분할.
- 300-example Mixed Emotion stress test와 evaluation-only 표시.
- Phase 1에는 cleaned text, Phase 2에는 minimally sanitized original text라는 입력 정책.
- calibration에서 T*, τ*를 정하는 구조와 accept/route 판정.
- 나머지 배치, 색상, 박스 스타일, 전체 데이터 흐름.

## 그림을 수정한 뒤

최종 drawio와 텍스트 선택 가능한 PDF를 전달하면 논문 Figure 1을 교체한다.
그때 캡션의 `Original artwork is retained ... pending label updates`라는 임시 안내도 제거한다.
현재 버전은 본문과 수치는 최신이지만, Figure 1 내부 일부 문구는 아직 구형인 검토본이다.
