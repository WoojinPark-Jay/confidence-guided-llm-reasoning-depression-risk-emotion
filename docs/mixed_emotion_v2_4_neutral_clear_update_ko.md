# Mixed Emotion v2.4 Neutral-clear 업데이트

## 업데이트 목적

기존 Mixed Emotion v2.3 데이터셋은 Depression, Neutral, Happy 각 100개로 구성되어 있었지만, Neutral 예시 일부가 실제로는 긍정 단서, 우울 단서, 회복 단서, 감정 변화 단서를 함께 포함하고 있었다. 이 경우 사람이 봐도 Neutral로 판단하기 애매할 수 있고, DistilBERT가 Neutral을 Happy 또는 Depression으로 예측하는 것이 모델 오류라기보다 라벨 정의 문제처럼 보일 수 있다.

따라서 v2.4에서는 Depression과 Happy stress-test 예시는 유지하고, Neutral 100개만 더 명확한 technical/informational neutral 문장으로 교체했다.

## 변경된 파일

새 데이터셋:

`data/supplementary/mixed_emotion/mixed_emotion_stress_test_v2_4_neutral_clear_300.csv`

기존 비교용 데이터셋:

`data/supplementary/mixed_emotion/mixed_emotion_stress_test_v2_3_300.csv`

## v2.4 Neutral 작성 기준

Neutral 예시는 다음 기준을 따르도록 수정했다.

- 감정 강도가 낮은 technical/informational 문장으로 작성
- 긍정 또는 우울 단서가 dominant trajectory로 해석되지 않도록 구성
- Technology, datascience, AskScienceDiscussion, webdev 계열에 가까운 정보성 질문과 절차 설명 중심
- sadness, hopelessness, relief, accomplishment 같은 강한 정서 단어는 Neutral 예시에서 최대한 제외
- 사람이 읽어도 Neutral 라벨이 방어 가능한 문장으로 조정

## Final 01 반영 내용

Final 01 Colab 노트북은 v2.4 파일을 우선 사용하도록 업데이트했다.

노트북:

`notebooks/colab/final/01_distilbert_phase1_training_final_colab.ipynb`

Mixed Emotion 로딩 순서:

1. `/content/mixed_emotion_stress_test_v2_4_neutral_clear_300.csv`
2. `/content/drive/MyDrive/confidence_guided_llm_reasoning/data/mixed_emotion_stress_test_v2_4_neutral_clear_300.csv`
3. GitHub raw URL의 v2.4 CSV

즉 Colab에서 로컬 업로드 파일이 있으면 그 파일을 먼저 읽고, 없으면 Drive 파일, 그것도 없으면 GitHub의 v2.4 파일을 읽는다.

## 함께 반영된 threshold 운영 정책

Final 01에서는 threshold 후보 범위를 기존 `[0.50, 1.00]` 중심에서 final operating policy 기준 `[0.70, 1.00]`로 조정했다.

이유는 3-class softmax에서 0.50 근처 confidence는 약한 과반 confidence에 가까우며, depression-risk-related emotion classification에서는 중간 confidence 예측을 Phase 1에서 바로 확정하기보다 Phase 2 reasoning으로 보내는 보수적 운영 정책이 더 방어 가능하기 때문이다.

추가로 `phase1_threshold_grid_lower_bound_sensitivity.csv`를 저장하여 `[0.50, 0.60, 0.70, 0.75, 0.80]` lower-bound별 routing/coverage/risk 변화를 비교할 수 있게 했다.

## 최종 v2.4 Neutral 구성

최종 v2.4 Neutral subset은 너무 쉬운 완전 기술문만으로 구성하지 않고, 다음과 같이 나누었다.

- `clear_neutral_technical_informational`: 75개
- `neutral_mild_ambiguity_factual`: 25개

`neutral_mild_ambiguity_factual` 예시는 일정 불일치, 양식 필드 차이, 문서 버전 차이, 누락된 절차 정보처럼 약간의 procedural ambiguity를 포함한다. 다만 sadness, happiness, depression, relief 같은 강한 정서 단서는 넣지 않았고, dominant label은 여전히 Neutral로 방어 가능하게 유지했다.

이 구성의 목적은 Neutral을 사람이 봐도 명확하게 유지하되, Mixed Emotion stress-test가 지나치게 쉬운 sanity-check로만 보이지 않도록 일부 Neutral 예시에 낮은 수준의 난이도를 부여하는 것이다.

## 기대되는 확인 포인트

v2.4를 사용한 뒤에는 다음을 다시 확인한다.

- Mixed Emotion Phase 1 전체 정확도
- Neutral class confusion matrix 개선 여부
- Mixed Emotion routed count 및 routing rate
- routed subset에서 Phase 1 오류가 충분히 포함되는지
- Final 02에서 Llama 2 / Llama 3 reasoning이 routed subset을 얼마나 correction하는지

## 논문 반영 시 주의점

v2.4는 Neutral 라벨 방어력을 높이기 위한 데이터셋 품질 개선이며, Neutral subset은 Reddit primary dataset의 Neutral source에 더 가까운 technical/informational 스타일로 정제했다. 논문에서는 Mixed Emotion Dataset을 primary benchmark로 과장하지 않고, emotionally complex or stress-test set으로 설명해야 한다.

권장 표현:

> The Neutral subset was revised to contain clearer factual and routine descriptions with low emotional intensity, so that errors on the Mixed Emotion stress-test would more directly reflect model behavior rather than ambiguous neutral-label construction.
