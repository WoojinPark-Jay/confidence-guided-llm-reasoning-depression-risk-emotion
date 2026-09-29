# Figure 1 최소 수정 가이드

## 이번에 할 작업

**기존 전체 아키텍처의 틀을 유지하고, 텍스트 박스 3개와 점선 1개만 수정한다.**

전체 3열 배치, 색상, 글꼴, 데이터 분할, 실선 데이터 흐름은 그대로 둔다. 아래는 다음에 draw.io에서 진행할 작업 목록이며, 아직 원본 그림을 수정했다는 뜻은 아니다.

| 순서 | 위치 | 할 일 |
|---|---|---|
| 1 | 중앙 위 `Phase 1 model development` | 별도 운영 모델처럼 읽히는 문구를 선택 설정의 최종 모델로 교체 |
| 2 | 중앙 아래 `Phase 2 re-evaluators` | 두 모델 모두 Direct·CoT·SELF-DISCOVER를 비교한 구조로 교체 |
| 3 | 오른쪽 `End-to-end prediction` | 파싱 실패 시 Phase 1 예측 유지 규칙 추가 |
| 4 | 보정 박스 → 신뢰도 박스 | `fixed T*` 점선 화살표 추가 |

## 1. Phase 1 model development

### 수정할 위치

중앙 Phase 1 영역의 왼쪽 위 박스. 기존 `as frozen operational classifier` 문구가 있는 부분이다.

### 박스 전체 교체 문구

```text
Phase 1 model development
train + validation · Bayesian sweep
DistilBERT selected
final seed-42 model from selected settings
Llama 2 / Mistral comparator runs
```

### 수정 이유

선택된 하이퍼파라미터 설정의 최종 seed-42 모델을 분류와 이후 라우팅에 이어서 사용한다는 의미다. 별도의 비교용 모델과 운영용 모델을 두는 구조로 읽히지 않도록 한다.

바로 아래 `Frozen Phase 1 inference / DistilBERT logits`는 **그대로 유지한다.** 여기서 frozen은 평가 중 가중치를 고정한다는 뜻이며, 별도 체크포인트를 사용한다는 뜻이 아니다.

## 2. Phase 2 re-evaluators

### 수정할 위치

중앙 Phase 2 영역에서 `Routed Phase 2 text`와 `Parsed Phase 2 label` 사이의 박스다. 현재 Llama 2 CoT와 Llama 3 SELF-DISCOVER를 각각 적어 둔 부분을 교체한다.

### 박스 전체 교체 문구

```text
Phase 2 re-evaluators
Llama 2 / Llama 3
Direct · CoT · SELF-DISCOVER
SD: cached task-level plan
+ per-post evidence checks
one configuration per run; no ensemble
```

### 수정 이유

- Llama 2와 Llama 3 모두 세 방법을 각각 실행한 비교 구조를 보여준다.
- SD는 공통 계획을 저장해 재사용하되, 각 게시물의 근거와 최종 라벨은 새로 생성한다.
- 한 실행에는 한 모델·한 방법을 적용한다. 여러 결과를 합치는 앙상블이 아니다.

SELECT → ADAPT → IMPLEMENT의 상세 흐름은 새 본문 Figure 2와 부록 Figure C1에서 설명한다. 이 작은 박스에 다시 모두 넣거나 보조 박스를 추가할 필요는 없다.

`v6c`, 생성 토큰 상한, 정확도 수치 등은 이번 전체 아키텍처 그림에 추가하지 않는다.

## 3. End-to-end prediction

### 수정할 위치

오른쪽 Outputs and Audit 영역의 위에서 두 번째 박스다.

### 박스 전체 교체 문구

```text
End-to-end prediction
Accepted: retain Phase 1 label
Routed: use parsed Phase 2 label
Parse failure: retain Phase 1 label
```

### 수정 이유

라우팅된 응답에서 라벨을 파싱하지 못한 경우의 처리까지 보여준다. 파싱 실패를 자동으로 오답 처리하는 것이 아니라, 기존 Phase 1 예측을 유지한 뒤 정답과 비교해 평가한다.

## 4. fixed T* 점선 추가

### 연결할 위치

```text
Calibration and routing policy
             ┆  fixed T*
             ↓
Calibrated confidence
```

보정 박스에서 아래 `Calibrated confidence` 박스로 짧은 점선 화살표를 추가하고, 선 옆에 `fixed T*`를 표시한다. 가능하면 T의 별표는 위첨자로 처리한다.

이미 있는 `FIXED τ*` 점선은 **그대로 유지한다.** 이 선은 보정 박스에서 accept/route 판단 마름모로 연결된다.

| 값 | 전달 대상 | 의미 |
|---|---|---|
| T* | Calibrated confidence | 모델 출력 확률을 보정하는 온도 |
| τ* | accept/route 판단 마름모 | 보정된 신뢰도를 받아들일지 재평가할지 나누는 임계값 |

### 선 배치 기준

- 기존 실선 데이터 흐름과 겹치지 않는 빈 공간으로 짧게 연결한다.
- 불필요하게 바깥으로 돌아가는 경로는 만들지 않는다.
- 점선 라벨이 박스 글씨나 선과 겹치지 않게 한다.
- 화살촉 바로 앞의 직선 구간에 충분한 길이를 둔다. 꺾이는 지점과 삼각형 화살촉이 붙지 않게 한다.

## 그대로 유지할 부분

- Reddit 120,000건과 70/10/10/10 분할, test 12,000건.
- Mixed Emotion 300건 및 `evaluation only` 표시.
- 정제된 Phase 1 입력과 원문 기반 Phase 2 입력의 구분.
- 전체 accept/route 분기 및 실선 연결 구조.
- 나머지 평가 지표와 출력 박스, 전체 색상과 글꼴.

이번 최소 수정에서는 추가 보조 박스나 모델 전달선을 만들지 않는다. SD 세부 설계는 본문 Figure 2, 실제 실행·파싱과 프롬프트는 부록이 담당한다.

## 작업 완료 체크리스트

- [ ] Phase 1 개발 박스 문구 교체.
- [ ] Phase 2 재평가 박스 문구 교체.
- [ ] End-to-end 박스에 파싱 실패 규칙 추가.
- [ ] 보정 → 신뢰도 박스에 `fixed T*` 점선 추가.
- [ ] 기존 `FIXED τ*` 점선과 나머지 배치 유지.
- [ ] 줄바꿈·박스 여백·화살촉 간격 확인.
- [ ] 편집 가능한 최종 `.drawio`와 텍스트 선택 가능한 `.pdf` 보관.
- [ ] 논문 Figure 1 교체 후, 캡션의 원본 그림 유지·라벨 수정 예정이라는 임시 안내 삭제.

PDF 내보내기 후 글씨를 실제로 선택·복사할 수 있는지, 원래 PNG와 줄바꿈 및 배치가 같은지 확인한다. 그림 교체만으로 실험 수치나 프롬프트를 변경하지 않는다.
