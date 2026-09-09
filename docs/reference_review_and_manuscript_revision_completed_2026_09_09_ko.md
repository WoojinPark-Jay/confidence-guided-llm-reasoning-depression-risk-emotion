# 참고문헌 검수 및 원고 반영 완료: 공동연구자 리뷰

작성일: 2026-09-09

대상 원고: *Confidence-Guided Selective LLM Re-Evaluation for Depression-Risk-Related Emotion Classification in Social Media Text*

## 1. 이번 정리의 결론

참고문헌 자료 확보와 현재 원고의 인용 내용 검수를 마무리했고, 확인된 수정 사항을 본문과 부록에 반영했다. 현재 원고 기준으로 **레퍼런스 때문에 반드시 추가로 확보하거나 수정해야 하는 항목은 남아 있지 않다.**

자료는 전문 PDF 41편과 공식 웹 자료 5개, 총 46개다. 검수는 각 문헌에서 **우리 원고가 실제로 인용하는 주장**을 관련 초록·방법·결과·논의와 대조하는 방식으로 진행했다. 원문 전체의 모든 주장이나 실험을 독립 재현했다는 의미는 아니다.

이번 수정의 목적은 연구 방향을 바꾸는 것이 아니라, 선행연구의 역할과 우리 구현의 차이를 정확히 설명하는 데 있다. 연구 질문, 데이터 분할, 실험 수치, routing 정책과 Phase 2 결과는 그대로 유지했다.

> 아래 [번호]는 동료 간 논의와 확보 폴더에 사용하던 **기존 감사 번호**다. 새 원고는 최초 등장 순서에 따라 자동 번호를 사용한다. [기존 번호 → 새 원고 번호 대조표](reference_number_map_2026_09_09_ko.md)를 함께 보면 된다.

## 2. 원고에 반영한 주요 수정

| 항목 | 수정 전의 문제 | 반영한 내용 |
|---|---|---|
| [9], [10]의 역할 | 데이터 품질·제한된 문맥·시간적 신호를 한꺼번에 설명 | [9]는 데이터 품질과 해석 가능성, [10]은 시간 및 게시물 간 정보의 근거로 분리 |
| [20]의 일반화 범위 | 대학생 에세이라는 연구 맥락이 충분히 드러나지 않음 | 현재 우울 집단과 비우울 집단의 해당 글쓰기 과제 차이로 범위를 명시 |
| [29], [31]의 배치 | 특징 기반 연구가 대규모 심층 문맥 모델의 발전 근거처럼 읽힘 | 특징 기반 분류 단락으로 이동. [31]은 1,841개 게시물(1,293 + 548)의 연구로 정확히 소개 |
| [27], [28], [30], [32] | 구조의 설계 특성과 우울 분류의 실증 근거가 섞임 | 기반 모델, 실제 응용 실험, 시간순 사용자 평가의 역할을 구분 |
| [19] 합성 데이터 | 선행연구의 합성 학습 데이터를 우리 평가셋의 검증 근거처럼 읽을 여지 | 임상 인터뷰 요약 기반 training augmentation과 우리 synthetic stress test를 구분 |
| [35], [43], [44], [45] | LLM 연구의 입력·참조 기준·분류 구조가 포괄적으로 서술됨 | diary/설문 기반 평가, 문항별 예측·설명, LLM 특징 + GBT, encoder를 포함한 리뷰라는 차이를 명시 |
| [17] CoT | 원 논문의 few-shot 설정과 우리 zero-shot 다단계 프로토콜의 차이가 흐림 | 원 방법의 아이디어를 적용한 프로토콜임을 명시 |
| [18] SELF-DISCOVER | 원 방법 자체가 모든 입력마다 새 계획을 만든다고 설명 | 원 방법의 task-level plan 재사용과 우리 routed post별 discovery를 구분. 본문·부록·캡션에 반영 |
| Calibration 대상 | 비교한 모든 분류기를 보정한 것으로 읽힐 여지 | calibration과 routing은 고정 operational DistilBERT에만 적용했다고 명시 |
| Prompt와 rationale 해석 | anchoring 감소나 특정 단서의 인과 효과가 검증된 것처럼 표현 | 설계 의도와 생성된 설명 기록으로 한정. 별도로 측정하지 않은 효과는 주장하지 않음 |
| 개인정보 표현 | 일부 정보만 제거한 원문을 de-identified라고 표현 | minimally sanitized로 수정하고 원문 검색 가능성이 남는다는 점 명시 |
| 참고문헌 관리 | 수동 번호가 본문 등장 순서와 불일치 | `\cite{refN}` 기반으로 전환하고 최초 등장 순서로 정렬 |
| 부록 표 | 이어지는 사례 표의 식별 정보와 oracle 수식 단위가 불명확 | continued 표제 추가, 비율/퍼센트와 음수 correction share 의미 설명 |

## 3. 핵심 문장 수정 예시

### 특징 기반 선행연구의 역할

기존에는 Tadesse 연구를 “a sizable Reddit corpus”라고 소개하면서 딥러닝 발전 흐름에 묶었다. 실제 연구 규모와 방법에 맞춰 다음처럼 바꿨다.

> Tadesse et al. evaluated linguistic and topic-based features with several classifiers on 1,841 Reddit posts, comprising 1,293 depression-indicative and 548 comparison posts.

해당 연구를 제외한 것이 아니라, 적절한 선행연구 위치로 옮긴 것이다.

### SELF-DISCOVER의 원 방법과 우리 적용 방식

기존에는 “SELF-DISCOVER constructs a reasoning structure for each routed input”이라고 적어 원 방법과 우리 구현을 구분하지 못했다. 다음과 같이 수정했다.

> The original method discovers a task-level plan from unlabeled task examples and reuses it across instances. Our implementation instead constructs and stores a plan for each routed post.

이 구분에 맞춰 본문에서 input-specific adaptation으로 정의했고, 원 논문의 계획 재사용에 따른 효율을 우리 구현의 실측 효율처럼 가져오지 않도록 한계도 연결했다.

### 생성된 설명과 인과 근거

부록의 “the final trajectory ... controls the decision”은 설명 기록만으로 확인할 수 있는 범위를 넘는다. 해당 응답이 마지막 감정 흐름을 강조했다는 관찰로 바꾸고, 그 문구가 실제 예측 원인을 입증하지는 않는다고 정리했다.

## 4. 앞서 주의가 필요했던 문헌의 최종 처리

- **[9]: 인용 유지.** 확보본은 저자 공개본이며 최종 저널본과의 전체 동일성까지 확인한 것은 아니다. 현재 원고는 확보본에서 확인되는 일반적인 배경 설명만 사용한다. 따라서 필수 추가 다운로드 항목으로 남기지 않는다. 이후 특정 수치·표·직접 인용을 추가한다면 해당 부분의 출판본 대조가 필요하다.
- **[20], [27]: 인용 유지.** 오래된 PDF의 텍스트 추출 문제와 연구 내용의 타당성은 별개다. 각각 집단 비교와 LSTM의 일반적인 구조 설명 범위에 맞춰 사용했다.
- **[31]: 수정 반영 완료.** 규모 표현과 문헌 배치를 바로잡았다. 논문 자체를 삭제할 이유는 없다.
- **[35]: 인용 유지.** 공식 정정 공지는 지원 과제번호 수정이며, 여기서 인용한 연구 방법·결과를 무효화하는 정정은 아니다.
- **[43]: 공식 서지 기준으로 정리.** PDF의 저자명 표기와 현재 ACL 서지의 표기 차이를 기록했고, 참고문헌은 현재 공식 export의 복합 성 표기에 맞췄다.
- **[45]: 최종 출판본 기준으로 인용 유지.** BERT·RoBERTa 등도 포함하는 리뷰이므로 생성형 챗봇의 치료 효능 근거로 확대하지 않았다.

## 5. 이번에 바꾸지 않은 것

| 구분 | 유지한 결과 |
|---|---|
| Phase 1, 3-seed accuracy | DistilBERT 96.90 ± 0.12%, Mistral 95.56 ± 0.11%, Llama 2 95.09 ± 0.06% |
| Operational DistilBERT | Accuracy 96.69%, T*=1.7706, τ*=0.70 |
| Reddit | Routed 171개, routed Phase 1 오류 87개, Llama 2 net −3, Llama 3 net +30 |
| Mixed Emotion | Routed 44개, routed Phase 1 오류 21개, Llama 2 net +12, Llama 3 net +18 |
| 통계 및 그림 | 기존 paired 결과·구간 유지. 본문 결과 표와 그림 파일은 기준 원고와 동일성 확인 |

이번 작업은 문헌과 서술의 검수다. 학습·추론을 새로 실행한 결과로 바꾼 것이 아니다. Operational checkpoint와 matched comparison의 역할 구분도 유지했다.

## 6. 제출 전 남은 사항과의 구분

**레퍼런스 검수·반영은 현재 원고 기준으로 완료했다.** 다음 항목은 참고문헌 미확보 문제가 아니라 제출 정보와 연구 범위에 관한 확인 사항이다.

1. 저자·소속·교신저자·지원금·저자 약력 및 최종 AI 사용 공개 문구 확정.
2. 실제 Reddit 원문 사례의 공개 범위와 연구윤리 심의 또는 면제 여부 확인.
3. 현재 명시한 한계 유지: routed Reddit 사례의 prompt 개발 재사용, 모델과 prompting method가 함께 바뀌는 비교, 합성 데이터의 독립 전문가 검증 부재, 실제 GPU/토큰/비용 측정 부재.

추가 실험을 수행하지 않은 상태에서 이 한계들이 해결됐다고 쓰지는 않는다. 반대로 이 항목들을 “레퍼런스 검수가 아직 안 끝났다”는 의미로 묶지도 않는다.

## 7. 검토할 산출물

- 원고 버전: `Paper_20260909_reference_verified`
- Overleaf: `CGSLR_IEEE_Access_Overleaf_Reference_Reviewed_2026_09_09.zip`
- PDF: `CGSLR_IEEE_Access_Reference_Reviewed_2026_09_09.pdf` (27페이지)
- [문헌별 인용 용도 대조](reference_claim_audit_completed_2026_09_09_ko.md)
- [기존 번호와 새 원고 번호 대조](reference_number_map_2026_09_09_ko.md)

원고 패키지에는 문단별 수정 전후, 전체 소스 diff, 빌드·레이아웃 확인 기록도 포함했다. 이번 Git 업데이트의 범위는 공동 검토용 문서이며, 원문 PDF 모음이나 Overleaf ZIP을 공개 저장소에 새로 업로드하는 작업은 포함하지 않는다.

동료 검토에서는 문헌을 더 모으는 것보다 **수정된 선행연구 설명이 실제 구현과 일치하는지, 결론이 보고된 실험 범위를 넘지 않는지**를 중심으로 보면 된다.
