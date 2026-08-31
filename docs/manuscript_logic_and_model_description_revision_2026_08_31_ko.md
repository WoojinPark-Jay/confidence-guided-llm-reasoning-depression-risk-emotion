# 논문 논리 구조 및 모델 설명 보강 상세 기록

최종 갱신: 2026-08-31

대상 원고: *Confidence-Guided Selective LLM Re-Evaluation for Depression-Risk-Related Emotion Classification in Social Media Text*

용도: 공동연구자 문장 단위 검토, 교수 검토본 변경 이력 확인, 최종 3-seed 결과 반영 전 감사 기준

## 1. 문서 목적

이번 수정은 단순한 영문 윤문이 아니다. 다음 세 가지 논문 리스크를 줄이기 위해 원고의 논리와 모델 설명을 함께 보강했다.

1. 제안 기법의 핵심이 기존 모델 소개에 묻혀 보이는 문제
2. DistilBERT와 7B comparator의 학습 범위가 달라 비교가 과장되어 보일 수 있는 문제
3. calibration, routing, Phase 2 re-evaluation이 하나의 성능 주장처럼 섞여 보이는 문제

수정 원칙은 다음과 같다.

- 모델 설명은 무조건 짧게 줄이지 않는다.
- 논문의 실제 실험 설계와 연결되는 구조적 특징은 남긴다.
- 이 연구에서 실행하지 않은 역사적 학습 목적이나 수식은 축약한다.
- completed result, bounded comparator, illustrative placeholder를 명확히 구분한다.
- classifier, router, re-evaluator가 각각 무엇으로 평가되는지 분리한다.
- 실제 7B 3-seed 결과가 들어오기 전에는 DistilBERT의 보편적 우월성을 주장하지 않는다.

## 2. 이번 수정의 핵심 결론

모델별 설명을 두세 문장으로 일괄 축소하지 않았다. 대신 각 모델 문단을 대체로 다음 네 요소로 정리했다.

1. **Architecture:** 모델의 핵심 구조는 무엇인가
2. **Relevance:** 왜 이 논문의 Phase 1 비교에 포함되는가
3. **Experimental use:** 실제로 어떤 checkpoint와 trainable scope를 사용했는가
4. **Interpretation boundary:** 결과가 무엇을 말할 수 있고 무엇을 말할 수 없는가

이 구조는 모델을 장황하게 소개하지 않으면서도, reviewer가 실험 공정성과 재현 범위를 판단하는 데 필요한 정보를 남긴다.

## 3. 변경하지 않은 연구의 뼈대

이번 문장 보강으로 다음 항목은 바뀌지 않았다.

- RQ1a: Phase 1 predictive performance와 calibration quality
- RQ1b: prespecified matched Phase 1 classifier comparison
- RQ2: routing coverage, selective risk, error concentration
- RQ3: routed re-evaluation과 full-set end-to-end correction
- Reddit 120,000건의 70/10/10/10 prespecified split
- 별도 calibration split에서의 temperature scaling
- `alpha=0.05`, 후보 grid `0.70--1.00`, calibration-only threshold selection
- primary policy `tau*=0.70`
- Reddit held-out test 12,000건과 Mixed Emotion 300건의 평가 구조
- Reddit routed 171건과 Mixed Emotion routed 44건
- Llama 2 CoT와 Llama 3 SELF-DISCOVER의 Phase 2 결과
- paired bootstrap, exact McNemar test, Holm correction
- accepted high-confidence error audit
- 기존 표와 그림의 실측 DistilBERT 및 Phase 2 결과

즉, 이번 수정은 결과를 다시 만든 것이 아니라 **같은 실험을 더 정확한 범위와 언어로 설명한 작업**이다.

## 4. 변경 사항 요약표

| 영역 | 수정 전 위험 | 수정 후 핵심 | 논문 방어 효과 |
|---|---|---|---|
| Related Work | calibration과 selective classification이 한 문단에 압축됨 | 독립 subsection에서 calibration, abstention, cascaded correction의 차이를 설명 | novelty gap을 명확히 함 |
| DistilBERT | 역사적 distillation 수식이 본 연구 수식처럼 보일 수 있음 | 구조, 효율성, exact checkpoint, full fine-tuning, routing 역할을 연결 | 실제 사용과 배경지식을 분리 |
| Mistral 7B | 구조 설명은 있으나 비교 범위가 불명확함 | GQA/SWA의 의미와 frozen-backbone comparator 범위를 함께 명시 | exhaustive tuning 비교라는 오해 방지 |
| Llama 2 7B | Phase 1 base model과 Phase 2 chat model 구분이 약함 | 두 checkpoint 역할을 명시적으로 분리 | classifier와 reasoner 혼동 방지 |
| Fallback | paper-scale run에서 fallback 가능성이 남아 보임 | smoke test 전용이며 보고 실험에서는 미사용이라고 명시 | risk-control claim의 재현성 강화 |
| Computing Environment | 라이브러리와 quantization 설명이 포괄적임 | exact Phase 2 checkpoints와 generation settings 추가 | routed run 재현성 강화 |
| Results bridge | Reddit/Mixed 수치가 뒤 subsection과 중복됨 | 평가 원칙만 남기고 상세 수치는 각 결과 subsection에 집중 | 논리 흐름과 가독성 개선 |
| Phase 1 comparison | 7B 값이 completed evidence로 오해될 위험 | 표, 그림, 본문에서 professor-review placeholder임을 반복 표시 | 허위 실측 주장 방지 |

## 5. Related Work 보강

### 5.1 변경 전

기존 원고는 calibration과 selective classification의 관계를 한 문단으로 설명했다.

```text
The routing component of the proposed framework is also related to confidence
calibration and selective classification. Modern neural networks can be poorly
calibrated ... Selective classification addresses this problem by allowing a
model to abstain ... The present study extends this abstention perspective into
a complete re-evaluation workflow ...
```

### 5.2 문제점

- calibration과 abstention의 역할 차이가 빠르게 지나갔다.
- 기존 selective classification과 본 연구의 cascaded LLM re-evaluation 사이의 gap이 충분히 두드러지지 않았다.
- 좋은 router와 좋은 reasoner가 서로 다른 실증 질문이라는 점이 약했다.

### 5.3 변경 후

새 subsection `Confidence Calibration, Selective Prediction, and Cascaded Re-Evaluation`을 만들고 두 문단으로 분리했다.

```text
The routing component of the proposed framework is grounded in confidence
calibration and selective classification. Modern neural networks can be poorly
calibrated, meaning that their predicted probabilities do not necessarily
reflect their empirical likelihood of correctness. Selective classification
addresses this limitation by allowing a model to abstain on uncertain inputs,
thereby trading prediction coverage for lower risk among accepted predictions.
These methods establish how to decide which predictions should be retained,
but they do not by themselves specify how abstained predictions should be
corrected.
```

```text
The present study turns that abstention decision into a fixed cascaded
re-evaluation workflow ... The study then evaluates the classifier, router,
and re-evaluator with distinct outcomes ... This separation is important
because a well-calibrated router can identify difficult cases even when a
particular reasoner fails to improve them, and a plausible rationale does not
establish a correct replacement decision.
```

### 5.4 의도

이 수정으로 novelty를 “LLM을 사용했다”가 아니라 다음 결합으로 제시한다.

- calibrated confidence
- risk-controlled selective routing
- fixed Phase 2 re-evaluation
- corrected/introduced/net error accounting
- row-level audit artifacts

## 6. DistilBERT 설명 보강

### 6.1 변경 전

```text
DistilBERT is a compressed version of BERT trained using knowledge
distillation ... A generic distribution-matching component can be written as
the Kullback--Leibler divergence between teacher and student distributions at
a distillation temperature Td.
```

기존 원고에는 다음 역사적 수식도 있었다.

```latex
\mathcal{L}_{\mathrm{distill}} =
\mathrm{KL}(p_{\mathrm{teacher}}^{T_d} \parallel
p_{\mathrm{student}}^{T_d})
```

### 6.2 문제점

- 이 연구는 DistilBERT를 새로 distill하지 않는다.
- 역사적 pretraining 식의 `T_d`가 본 연구의 calibration temperature `T*`와 혼동될 수 있다.
- 실제 중요한 정보인 exact checkpoint, full fine-tuning, routing confidence 공급 역할이 상대적으로 약했다.

### 6.3 변경 후

```text
DistilBERT is a compact bidirectional encoder obtained by distilling BERT into
a six-layer student. Its original pretraining combines language-modeling,
teacher-distribution matching, and representation-alignment objectives ...
Those properties make it an appropriate candidate for the high-volume first
stage of a selective pipeline ... In this study, the released
distilbert-base-uncased checkpoint is fully fine-tuned with supervised
three-class cross-entropy; the historical distillation process is not repeated.
Its logits are subsequently temperature-calibrated, and only this operational
checkpoint supplies the confidence scores used for routing.
```

### 6.4 유지한 내용

- six-layer student라는 구조적 설명
- BERT-base 대비 약 40% 적은 parameter
- 원 논문에서 보고된 약 60% inference speedup
- 대규모 Phase 1에 적합한 이유

### 6.5 삭제한 내용

- 본 연구에서 계산하거나 최적화하지 않은 역사적 KL distillation 식
- `T_d`와 `T*`를 구분하기 위한 별도 해명 문장

수식을 삭제한 것은 DistilBERT 설명을 약화한 것이 아니라, **본 연구가 실제 최적화한 수식에 독자의 주의를 집중시킨 것**이다.

## 7. Mistral 7B 설명 보강

### 7.1 변경 전

```text
Mistral 7B is a decoder-only transformer designed to achieve strong performance
while maintaining computational efficiency ... GQA improves inference
efficiency ... SWA enables efficient long-sequence processing ... Its matched
full-scale fine-tuning and evaluation remain pending.
```

### 7.2 문제점

- GQA와 SWA 설명은 있었지만, 이 구조가 본 연구의 contribution처럼 읽힐 수 있었다.
- 실제 classifier adaptation 범위가 frozen backbone인지 불분명했다.
- 왜 7B model을 비교하면서 full fine-tuning 또는 LoRA를 하지 않았는지 방어가 약했다.

### 7.3 변경 후

```text
Mistral 7B is a decoder-only transformer that combines grouped-query attention
(GQA) with sliding-window attention (SWA) ... These features explain why
Mistral is a relevant larger-model comparator, but they are not treated as
contributions of the present work. Here, the prespecified Mistral checkpoint is
used as a frozen representation backbone and only a task-specific three-class
classification head is optimized ... The experiment therefore asks whether
the frozen 7B representation yields a sufficient predictive advantage to
displace the much smaller operational classifier; it does not estimate the best
performance attainable through full fine-tuning, LoRA, QLoRA, or
architecture-specific adaptation.
```

### 7.4 해석 범위

이 비교가 답하는 질문:

> 공통 split과 bounded supervised protocol에서 frozen 7B representation이 작은 fully fine-tuned DistilBERT를 운영 모델에서 대체할 만큼 유리한가?

이 비교가 답하지 않는 질문:

> 가능한 모든 adaptation을 적용했을 때 Mistral 7B가 달성할 수 있는 최고 성능은 얼마인가?

## 8. Llama 2 설명 보강

### 8.1 변경 전

```text
Llama 2 is a family of pretrained and fine-tuned autoregressive language models
released by Meta, with model sizes ranging from 7B to 70B parameters. Within
the proposed Phase 1 design, the 7B model is a prespecified higher-capacity
comparison classifier ...
```

### 8.2 문제점

- Phase 1의 base checkpoint와 Phase 2의 chat checkpoint가 같은 “Llama 2”로 보일 수 있었다.
- comparator의 frozen scope가 약했다.
- model size만으로 일반적 우열을 말하는 것처럼 오해될 수 있었다.

### 8.3 변경 후

```text
Llama 2 is a family of decoder-only autoregressive transformers released at
several parameter scales. The 7B base checkpoint provides a second
higher-capacity representation family for the Phase 1 comparison and is
distinct from the chat-tuned Llama 2 checkpoint used later as a Phase 2
reasoner. For classification, the pretrained 7B backbone remains frozen and
only the task-specific three-class parameters are optimized ... Accordingly,
the comparison supports an operational model choice within the tested scope
rather than a general ranking of DistilBERT and Llama 2.
```

### 8.4 역할 구분

| 위치 | Llama 2 역할 | 학습/추론 범위 |
|---|---|---|
| Phase 1 | 7B base representation comparator | frozen backbone + task-specific classifier parameters |
| Phase 2 | `NousResearch/Llama-2-7b-chat-hf` reasoner | 4-bit inference, CoT prompt, no Phase 1 classifier training |

## 9. Phase 1 비교 공정성 문장 보강

원고의 Phase 1 방법론에는 다음 경계를 추가했다.

```text
For these substantially larger comparators, the pretrained backbone was frozen
and only the task-specific classification parameters were optimized; LoRA,
QLoRA, and other adapter updates were not used. This deliberately bounded
comparison asks whether either 7B representation provides enough predictive
benefit to displace the much smaller operational model, rather than attempting
an exhaustive optimization study of 7B fine-tuning methods.
```

이 문장의 목적은 DistilBERT 결과를 약하게 만드는 것이 아니다. 오히려 reviewer가 비교의 정확한 범위를 이해하게 하여 다음과 같은 과장된 주장을 차단한다.

- “DistilBERT가 모든 조건에서 Mistral보다 우수하다.”
- “DistilBERT가 fully optimized Llama 2보다 우수하다.”
- “7B model에 대한 최적 fine-tuning 연구를 완료했다.”

최종 주장은 다음 수준으로 제한한다.

> Under the tested frozen-backbone comparator protocol, the completed three-seed results support the operational Phase 1 model choice.

## 10. Threshold fallback 설명 보강

### 10.1 변경 전

```text
If no candidate threshold satisfies the risk constraint in small calibration
runs, the implementation records the infeasibility status and may use a
minimum-risk fallback for smoke testing. For final paper-scale experiments,
fallback behavior should be disabled or reported explicitly ...
```

### 10.2 문제점

`should be disabled or reported`라는 표현은 최종 reported experiment에서 fallback이 실제 사용되었는지 모호하게 남긴다.

### 10.3 변경 후

```text
The implementation records an explicit infeasibility status if no candidate
satisfies the risk constraint. A minimum-risk fallback exists only for notebook
smoke tests and was not invoked in the reported paper-scale experiment; no
fallback threshold is treated as satisfying the risk-control constraint ...
```

### 10.4 재현성 의미

- reported `tau*=0.70`은 fallback 결과가 아니다.
- calibration split에서 risk constraint를 만족한 feasible candidate다.
- fallback threshold는 risk-controlled threshold라고 서술하지 않는다.
- `threshold_provenance.json`에 feasibility와 selection metadata를 저장한다.

## 11. Computing Environment 보강

### 11.1 변경 전

```text
The completed DistilBERT experiment was implemented in PyTorch using the
Hugging Face Transformers library ... Routed-sample Llama inference was
executed in GPU-backed Google Colab sessions with quantized model loading.
```

### 11.2 변경 후 추가된 재현 정보

| 항목 | 기록 내용 |
|---|---|
| 공통 환경 | Python, PyTorch, Transformers, Datasets, scikit-learn, SciPy, pandas |
| sweep 기록 | W&B |
| DistilBERT checkpoint | `distilbert-base-uncased` |
| Phase 2 Llama 2 | `NousResearch/Llama-2-7b-chat-hf` |
| Phase 2 Llama 3 | `NousResearch/Meta-Llama-3-8B-Instruct` |
| quantization | bitsandbytes 4-bit loading |
| Llama 2 generation | max new tokens 256, temperature 0.6, top-p 0.9 |
| Llama 3 structure stages | max new tokens 1,024 |
| Llama 3 terminal answer | deterministic decoding |
| persistence | routed row마다 prediction output 저장 |
| timing claim | controlled latency benchmark가 없어 비교 수치 미보고 |

### 11.3 의도

hardware-dependent 속도나 비용을 측정하지 않았는데도 효율을 수치로 과장하지 않는다. 현재 실증된 효율 주장은 다음으로 제한한다.

- Reddit에서 1.42%만 LLM re-evaluation 호출
- Mixed Emotion에서 14.67%만 LLM re-evaluation 호출
- 이는 **LLM invocation reduction**이며 직접 측정한 wall-clock, energy, monetary saving은 아니다.

## 12. Results 연결 문단 정리

### 12.1 변경 전

`Two-Phase System Efficacy and Confidence-Guided Routing` subsection에서 다음 내용이 네 문단에 걸쳐 반복되었다.

- routing의 목적
- Reddit과 Mixed의 상대적 난이도
- Mixed의 corrected/introduced 수치
- selective invocation rate

동일 수치가 바로 다음 Reddit 및 Mixed 결과 subsection에서 다시 제시되었다.

### 12.2 변경 후

bridge paragraph 하나로 줄였다.

```text
With the classifier and routing policy fixed, RQ3 evaluates whether the
assigned Phase 2 reasoner improves the complete system rather than merely
producing plausible explanations on selected cases. Llama 2 CoT and Llama 3
SELF-DISCOVER therefore receive the same routed IDs within each evaluation set,
and every corrected error is considered together with any newly introduced
error. The following Reddit and Mixed Emotion subsections report the
routed-only behavior, recombined full-set metrics, and paired statistical
results without treating low confidence as a guarantee of successful
correction.
```

### 12.3 효과

- 원칙은 bridge paragraph에서 설명한다.
- 정확한 수치는 Reddit/Mixed subsection과 표에서 한 번씩 집중 보고한다.
- Llama 2의 negative/neutral result도 감추지 않는다.
- plausible rationale와 correct replacement decision을 구분한다.

## 13. 현재 실측과 placeholder의 경계

### 13.1 현재 실측으로 사용 가능한 항목

- DistilBERT held-out metrics
- temperature scaling과 calibration metrics
- primary routing policy와 routing outputs
- Reddit/Mixed Phase 1 predictions
- Reddit/Mixed Llama 2 및 Llama 3 routed results
- full-set end-to-end recombination
- corrected, introduced, net corrections
- paired statistical analysis
- accepted-error audit

### 13.2 아직 최종 실측이 아닌 항목

- Mistral 7B Phase 1 three-seed mean and standard deviation
- Llama 2 7B Phase 1 three-seed mean and standard deviation
- 두 comparator의 최종 selected hyperparameters와 run manifest 일부

### 13.3 현재 교수 검토본 처리

Table I과 classifier comparison figure에는 예상 배치를 검토하기 위한 illustrative value가 남아 있다.

- italic 또는 TBD/professor-review 문구로 명시한다.
- model selection evidence에 포함하지 않는다.
- abstract, conclusion, routing policy의 실증 근거로 사용하지 않는다.
- 제출본 전에는 반드시 exact three-seed aggregate로 교체한다.

## 14. 3-seed 결과 수령 후 갱신 절차

1. seed별 raw metric과 run identifier를 보존한다.
2. Accuracy, macro precision, macro recall, macro F1의 mean과 standard deviation을 계산한다.
3. 동일 12,000 held-out split 사용 여부를 확인한다.
4. frozen backbone과 trainable parameter 범위를 run manifest와 대조한다.
5. Table I의 illustrative cells를 `mean +/- SD`로 교체한다.
6. classifier comparison figure를 같은 aggregate로 재생성한다.
7. `professor-review`, `illustrative`, `TBD`, `placeholder`, `pending` 문구를 전역 검색한다.
8. Phase 1 model selection paragraph를 실제 결과에 맞게 고친다.
9. Abstract, Contributions, Discussion, Limitations, Conclusion의 선택 근거를 동기화한다.
10. clean compile 후 PDF, source ZIP, supplementary outputs를 다시 동결한다.

## 15. 공동연구자 검토 체크리스트

### 15.1 모델 설명

- [ ] DistilBERT 설명이 너무 짧지 않으면서 본 연구 사용과 직접 연결되는가?
- [ ] Mistral의 GQA/SWA 설명이 contribution처럼 과장되지 않는가?
- [ ] Phase 1 Llama 2 base checkpoint와 Phase 2 chat checkpoint가 분명히 구분되는가?
- [ ] fully fine-tuned DistilBERT와 frozen 7B comparator의 trainable scope가 명확한가?
- [ ] LoRA/QLoRA 미사용이 limitation이자 범위 정의로 정확히 표현되는가?

### 15.2 방법론 논리

- [ ] calibration, routing, re-evaluation의 역할이 구분되는가?
- [ ] `T*`와 `tau*`가 calibration split에서만 고정된다는 점이 일관적인가?
- [ ] smoke-test fallback이 reported experiment에 사용되지 않았다는 점이 명확한가?
- [ ] routed subset improvement와 full-set end-to-end improvement가 혼동되지 않는가?
- [ ] rationale가 correctness 증거로 취급되지 않는가?

### 15.3 주장 범위

- [ ] proxy emotion classification을 clinical diagnosis로 표현하지 않는가?
- [ ] synthetic Mixed Emotion 결과를 population-level generalization으로 과장하지 않는가?
- [ ] 7B comparison을 exhaustive adaptation benchmark로 표현하지 않는가?
- [ ] invocation reduction과 실제 시간/비용 절감을 구분하는가?
- [ ] illustrative classifier 값이 실측처럼 읽힐 여지가 없는가?

## 16. 이번 회의에서 확인할 질문

1. 7B comparator의 frozen-backbone protocol이 연구 질문에 충분히 부합하는가?
2. 모델 설명의 현재 길이가 IEEE Access 독자에게 필요한 최소 배경을 제공하는가?
3. DistilBERT의 operational selection을 accuracy와 parameter scale 두 기준으로 설명하는 것이 적절한가?
4. actual 3-seed results가 DistilBERT보다 높을 경우 operational classifier를 재검토할 것인가?
5. exact Mistral base checkpoint identifier와 selected hyperparameters를 언제 동결할 것인가?
6. latency 또는 cost benchmark를 이번 제출에 포함할지 future work로 남길 것인가?

## 17. 관련 원고 및 산출물

교수 검토용 최신 산출물:

- `CGSLR_IEEE_Access_Professor_Review_Logic_and_Methodology_Refined_2026_08_31.pdf`
- `CGSLR_IEEE_Access_Overleaf_Logic_and_Methodology_Refined_2026_08_31.zip`

저장소 연결 문서:

- `docs/final_research_status_and_remaining_work_ko.md`
- `docs/ieee_access_submission_readiness_checklist_ko.md`
- `docs/routing_threshold_policy_audit_ko.md`
- `docs/paired_end_to_end_statistical_analysis_ko.md`
- `docs/final_model_specific_prompt_policy_ko.md`

## 18. 한 문장 회의 요약

이번 수정은 모델 설명을 줄인 작업이 아니라, **각 모델의 구조적 의미는 보존하면서 실제 사용 범위, 비교 공정성, calibration-routing-reasoning의 논리 경계를 명확히 한 원고 감사 작업**이며, 최종 제출 전 가장 큰 남은 실증 작업은 Mistral 7B와 Llama 2 7B의 matched three-seed aggregate 반영이다.
