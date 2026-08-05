# Phase 2 Trajectory-Aware Prompt Experiment Plan

이 문서는 Mixed Emotion Dataset에 대한 Phase 2 reasoning 실험에서 `trajectory-aware prompt variant`를 별도 노트북으로 분리한 이유와 실험 목적을 정리한다. 이 실험은 기존 Phase 2 reasoning prompt를 바로 대체하기 위한 것이 아니라, blended emotion 또는 emotionally shifting text에서 더 타당한 판단 기준을 제공하는지 비교 검증하기 위한 추가 실험이다.

## 1. 실험을 새 파일로 분리한 이유

기존 Phase 2 reasoning notebook은 논문 Appendix B/C에 맞춘 기본 prompt protocol을 구현한다. 이 버전은 dominant emotion과 overall sentiment를 중심으로 재평가하도록 설계되어 있다.

그러나 Mixed Emotion Dataset에는 다음과 같은 샘플이 포함된다.

- 긍정, 중립, 우울 관련 단서가 동시에 존재하는 경우
- 글의 초반 감정과 후반 감정이 다르게 전개되는 경우
- 표면적으로는 담담하지만 마지막 정서적 결론이 분명한 경우
- 힘든 경험을 서술하다가 회복, 안도, 성취로 끝나는 경우
- 긍정적 표현이 일부 있지만 최종적으로는 unresolved distress로 끝나는 경우

이런 케이스에서는 단순히 전체 단어 분위기를 평균 내거나 `dominant emotion`만 요구하면 모델이 애매한 샘플을 Neutral로 보내는 경향이 생길 수 있다. 따라서 기존 prompt를 직접 덮어쓰지 않고, 새 prompt policy가 실제로 도움이 되는지 확인하기 위해 별도 실험용 노트북을 만들었다.

새 노트북:

```text
notebooks/colab/14_phase2_mixed_emotion_reasoning_trajectory_prompt_colab.ipynb
```

기존 노트북:

```text
notebooks/colab/13_phase2_mixed_emotion_reasoning_colab.ipynb
```

## 2. 실험 목적

이 실험의 목적은 다음 질문을 확인하는 것이다.

1. Mixed 또는 shifting emotion 샘플에서 모델이 Neutral로 과도하게 빠지는 현상이 줄어드는가?
2. 글의 최종 정서적 귀결(final emotional trajectory / final takeaway)을 기준으로 판단하게 했을 때, target label과의 일치도가 개선되는가?
3. Depression, Neutral, Happy 중 특정 클래스만 유리해지는 것이 아니라 세 클래스 모두에서 더 합리적인 판단 기준을 제공하는가?
4. Llama 2 CoT와 Llama 3 SELF-DISCOVER 모두에서 동일한 prompt policy가 안정적으로 작동하는가?
5. 새 prompt가 논문에 들어갈 만큼 명확하고 publication-safe한가?

## 3. 핵심 변경점

Trajectory-aware prompt variant는 기존 prompt 구조를 유지하면서 classification guideline만 강화한다.

핵심 규칙은 다음과 같다.

```text
For blended or emotionally shifting texts, classify the post according to the final emotional trajectory and overall takeaway rather than averaging isolated emotional cues. Do not default to Neutral merely because multiple emotions are present.
```

즉, 혼합 감정 텍스트를 판단할 때 단순히 여러 감정이 섞였다는 이유만으로 Neutral을 선택하지 않는다. 대신 글이 최종적으로 어떤 정서적 방향으로 끝나는지 확인한다.

구체적인 판단 기준은 다음과 같다.

- distress에서 relief, accomplishment, positive resolution으로 이동하면 Happy로 판단할 수 있다.
- neutral 또는 positive context에서 hopelessness, emotional exhaustion, unresolved distress로 이동하면 Depression으로 판단할 수 있다.
- 최종 takeaway가 factual, balanced, emotionally mild이고 Depression/Happy 방향이 분명하지 않을 때만 Neutral로 판단한다.

## 4. 왜 치팅이 아닌가

이 변경은 test result를 보고 특정 정답을 맞히도록 만든 규칙이 아니라, mixed emotion annotation policy를 명확히 정의하는 것이다.

Blended 또는 shifting emotion 텍스트는 원래 라벨 기준이 애매할 수 있다. 사람도 이런 글을 읽을 때 단어 단위 감정의 평균보다 글의 최종 정서적 결론을 중요하게 보는 경우가 많다. 따라서 final emotional trajectory를 우선한다는 기준은 합리적인 분류 정책으로 볼 수 있다.

다만 치팅처럼 보이지 않으려면 다음 조건을 지켜야 한다.

- 정답 label을 직접 암시하지 않는다.
- 특정 샘플에만 맞춘 규칙을 넣지 않는다.
- Depression, Neutral, Happy 모두에 대칭적으로 적용한다.
- prompt protocol을 논문 Appendix에 투명하게 공개한다.
- test set에서 prompt를 반복적으로 튜닝했다면 그 과정을 명확히 기록한다.

현재 variant는 세 클래스 모두에 같은 trajectory rule을 적용하므로, 성능 조작보다는 label policy 명확화에 가깝다.

## 5. 비교해야 할 실험 조건

비교 대상은 다음 두 조건이다.

| Condition | Notebook | Prompt policy | Output files |
|---|---|---|---|
| 기본 Phase 2 prompt | `13_phase2_mixed_emotion_reasoning_colab.ipynb` | Dominant emotion / overall sentiment 중심 | `mixed_emotion_llama2_cot_results.csv`, `mixed_emotion_llama3_self_discover_results.csv` |
| Trajectory-aware prompt variant | `14_phase2_mixed_emotion_reasoning_trajectory_prompt_colab.ipynb` | Mixed/shifting text에서 final emotional trajectory와 final takeaway 우선 | `mixed_emotion_llama2_cot_trajectory_prompt_results.csv`, `mixed_emotion_llama3_self_discover_trajectory_prompt_results.csv` |

두 노트북은 같은 dataset, 같은 model, 같은 checkpoint/resume 구조를 사용한다. 차이는 prompt policy와 output filename이다.

## 6. 확인해야 할 주요 지표

최소한 다음 결과를 비교한다.

- 전체 accuracy
- class-wise precision, recall, F1
- Depression / Neutral / Happy별 confusion matrix
- Neutral prediction 비율 변화
- target label이 Depression 또는 Happy인데 Neutral로 예측된 오류 수
- target label이 Neutral인데 Depression 또는 Happy로 과하게 이동한 오류 수
- Llama 2와 Llama 3 각각의 변화 방향

특히 중요한 것은 단순 accuracy 상승이 아니라, mixed/shifting case에서 Neutral default 문제가 줄었는지 확인하는 것이다.

## 7. 결과 해석 기준

Trajectory-aware prompt를 논문에 반영하려면 다음 조건을 만족하는지 확인해야 한다.

1. Neutral 과다 예측이 줄어든다.
2. Depression과 Happy의 recall이 개선되거나 최소한 악화되지 않는다.
3. Neutral class의 precision이 크게 무너지지 않는다.
4. 새 prompt가 특정 label로 과도하게 bias되지 않는다.
5. raw output에서 final label parsing이 안정적으로 작동한다.
6. Llama 2와 Llama 3 중 최소 하나 이상에서 명확한 개선 패턴이 보인다.
7. 개선이 일부 샘플에만 우연히 나타나는 것이 아니라 scenario type별로 설명 가능하다.

만약 accuracy는 조금 개선되지만 Neutral precision이 크게 떨어진다면, prompt를 바로 논문 기본 버전으로 반영하기보다 ablation 또는 supplementary prompt variant로 보고하는 것이 더 안전하다.

## 8. 논문 반영 방식

실험 결과가 좋으면 논문에서는 다음 위치를 업데이트할 수 있다.

- Methodology의 Phase 2 reasoning 설명
- Mixed Emotion Dataset stress-test 설명
- Appendix B: Llama 2 Chain-of-Thought prompting protocol
- Appendix C: Llama 3 SELF-DISCOVER prompting protocol
- Ablation 또는 supplementary analysis table

본문에서는 다음과 같이 설명할 수 있다.

```text
For blended or emotionally shifting posts, the Phase 2 prompt protocol instructed the reasoning model to prioritize the final emotional trajectory and overall takeaway rather than averaging isolated emotional cues or defaulting to Neutral when multiple cues co-occurred.
```

Appendix에는 classification guideline row와 별도 `Mixed and Shifting Emotion Handling` row를 추가하는 방식이 적절하다.

## 9. 현재 실행 권장 방식

먼저 기본 prompt notebook과 trajectory-aware prompt notebook을 같은 300-example dataset에서 각각 실행한다.

기본 prompt:

```text
notebooks/colab/13_phase2_mixed_emotion_reasoning_colab.ipynb
```

Trajectory-aware prompt:

```text
notebooks/colab/14_phase2_mixed_emotion_reasoning_trajectory_prompt_colab.ipynb
```

두 노트북 모두 Google Drive row-level checkpoint/resume 방식을 사용한다. Colab runtime이 끊겨도 Google Drive에 저장된 CSV를 읽고 완료된 `example_id`를 skip한 뒤 남은 샘플부터 이어서 실행할 수 있다.

## 10. 주의사항

- 이 variant는 아직 논문 최종 prompt로 확정된 것이 아니다.
- 실험 결과를 본 뒤 논문 Appendix B/C에 반영할지 결정한다.
- prompt를 바꿨다면 반드시 어떤 prompt로 얻은 결과인지 output filename과 문서에 명확히 남긴다.
- Phase 1 routed prediction이 준비되면 `target_as_placeholder`가 아니라 `prediction_column` 모드로 다시 실행해야 한다.
- 최종 논문 결과에는 prompt version, model version, dataset version을 함께 기록해야 한다.
