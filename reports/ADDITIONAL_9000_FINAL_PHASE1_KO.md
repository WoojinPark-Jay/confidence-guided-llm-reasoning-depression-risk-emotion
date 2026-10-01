# 기존 9,000건: 최종 모델 재평가 1단계

## 실행 링크

[Colab: 최종 DistilBERT 분류](https://colab.research.google.com/drive/1JTqh8ExWiXKYzLVpiUTLd5LhE2TGOm4h)

노트북 원본: `notebooks/colab/final/07_1_final_distilbert_existing9000_colab.ipynb`

1. GPU 런타임을 선택한다.
2. 위에서부터 실행하고 Drive 연결과 추가 인증 모두 `jay.seizethemoment@gmail.com`을 선택한다. 모델과 데이터는 자동으로 가져오므로 수동 업로드나 경로 수정은 필요 없다.
3. 입력 확인 단계에서 9,000건, 클래스별 3,000건을 확인한다.
4. 분류 실행 후 정확도, Macro F1, 라우팅 건수를 확인한다.

## 이번 단계에서 하는 일

최종 seed42 모델을 불러와 추론만 한다. 학습, temperature 재적합, 임계값 탐색은 하지 않는다.

| 항목 | 설정 |
|---|---|
| 모델 | 공동연구자에게 받은 최종 `seed_42_model` (4개 파일 SHA256 고정) |
| 온도 | 1.4801950079829693 |
| 라우팅 임계값 | 0.70 미만 |
| 최대 토큰 길이 | 256 |
| 평가 배치 | 64 |
| 평가 정밀도 | fp32 |

이전 9,000건 패키지의 데이터만 읽는다. 그 패키지 안의 예전 모델이나 실행 코드를 사용하지 않는다. 예전 라우팅 137건 또는 본 실험 라우팅 218건을 재사용하지 않고 이번 예측으로 라우팅을 결정한다.

## 파일 위치

자동 준비 폴더: `/content/drive/MyDrive/confidence_guided_llm_reasoning/assets/final_seed42_existing9000_20260930`

이 폴더 아래 `data/`에 기존 9,000건 파일 4개, `seed_42_model/`에 받은 최종 모델 파일 4개를 둔다. 전송 크기 제한 때문에 비공개 파일 3개로 나누어 올렸으며, 노트북이 자동으로 내려받아 합친 후 전체 ZIP과 각 파일의 해시를 확인한다. 다음 실행에서는 검증된 캐시를 재사용한다.

출력: `/content/drive/MyDrive/confidence_guided_llm_reasoning/outputs_final/unified_final/additional_9000_20260930/phase1`

다음 Llama 3 단계 입력은 출력 폴더의 `final_phase1_reasoning_input.csv`다. 전체 9,000건 예측과 정답을 포함하되 원문은 라우팅된 사례에만 기록한다. 생성 프롬프트는 정답을 사용하지 않는다. `phase2_handoff.json`에 경로, 해시, 전체/라우팅 건수가 저장된다.

이 노트북은 Llama 3를 실행하지 않는다. 다음 단계에서 동일한 라우팅 사례에 Direct, CoT, 최종 SELF-DISCOVER v6c를 적용한다.

## 검증과 재개

- 로컬 실제 입력 파일 4개의 해시, ID 9,000개, 클래스별 3,000건, ZIP과 풀린 폴더의 동일성을 확인했다.
- 기존 정답의 `_Group` 접미사는 명시적으로 정규화하며 숫자 라벨과 일치하는지 확인한다.
- 원문 파일과 분류 입력 파일의 행 순서가 다르므로 `example_id`로 결합한다.
- 라우팅, ID 결합, 저장, 재개, 라벨 순서, 다른 모델 차단 및 압축 해제 검증에 대한 테스트 9개를 통과했다.
- 모델 config의 `Depression_Group / Neutral_Group / Happy_Group`은 클래스 순서를 바꾸지 않고 출력 이름만 정규화한다. DistilBERT 입력에서 불필요한 token_type_ids를 제외한다.
- Drive 파일 3개의 업로드 크기와 실행 계정의 읽기 권한을 확인했다. 다른 사람에게 공개하지 않았다. 사용자 계정 인증을 통한 다운로드와 GPU 추론은 Colab 실행 시 최종 확인한다.
- 중단 시 같은 GPU 종류와 라이브러리 환경에서 다시 실행하면 이미 저장된 배치를 건너뛴다. 다른 설정의 결과는 같은 폴더에 섞지 않는다.
- 기존 12,000건, Mixed Emotion, 이전 9,000건 결과를 변경하지 않는다.

이 평가는 이미 평가 이력이 있는 동일 출처 9,000건에 최종 파이프라인을 적용하는 추가 평가다. 새로 확보한 미사용 평가셋으로 표기하지 않는다.
