# 9월 29일 공동 리뷰: 논문 업데이트와 마무리 작업

## 먼저 볼 문서

| 확인할 내용 | 문서 |
|---|---|
| 동료에게 필요한 파일, 저장 위치, 요청 이유 | [최종 결과 파일 공유 요청](COLLABORATOR_FILE_REQUEST_KO.md) |
| 오늘 완료한 일과 다음 작업 순서 | [업데이트 현황과 마무리 계획](UPDATE_AND_REMAINING_KO.md) |
| 원래 논문 스타일로 복원한 그림·표·부록 | [시각 품질 복원 내역](VISUAL_RESTORATION_KO.md) |
| 기여·파인딩 연결, SD 설명·흐름도·프롬프트 안내 보강 | [본문·부록 설명 보강 내역](EXPOSITION_UPGRADE_KO.md) |
| SD 설계 기여와 연구자·LLM 역할, 계획 생성 시점 명확화 | [SD 기여·흐름 설명 보강](SD_CONTRIBUTION_CLARITY_KO.md) |
| 원본 아키텍처에서 바꿀 부분 | [아키텍처 수정 가이드](ARCHITECTURE_EDIT_GUIDE_KO.md) |
| 원고 변경 문장 대조 | [원고 변경 내역](MANUSCRIPT_FULL_DIFF.md) |
| 12조건의 성능과 Phase 1 대비 통계 | [전체 결과 CSV](evidence/final_results_and_count_statistics.csv) |

**지금 요청하는 것은 새 실험이 아니라, 이미 실행한 최종 SD(v6c)의 결과 CSV 네 개다.** 원고와 집계 표는 업데이트했고, 파일을 받은 뒤 게시글별 대조와 SD 대 Direct·CoT 비교를 마무리한다.

오늘은 기존 논문의 색상·글꼴·벡터 그림 스타일도 복원했다. 이어 본문과 부록의 방법 설명, SD 흐름도, 이전/최종 설계 비교를 보강했다. 추가로 SD를 감정 분류에 맞게 구체화한 기여를 명시하고, 누가·언제·무엇을 보고 계획을 만드는지 본문·알고리즘·그림에서 일관되게 설명했다. Phase 1 혼동행렬은 본문에 두고, 메인 아키텍처는 기존 원본을 유지한다. 실제 프롬프트와 실험 집계는 변경하지 않았다.

## 이 폴더의 범위

본문 9쪽 Figure 2에는 SD 핵심 흐름을 새로 추가했다. 계획 생성과 게시물별 실행을 구분하며, 부록 20쪽 Figure C1의 상세 도식 및 원문 프롬프트와 연결된다. 기존 전체 아키텍처와 실험 수치는 유지한다.

이 폴더는 공동 리뷰용 원고·문서·그림·프롬프트·집계 근거의 스냅샷이다. 단독 Overleaf 프로젝트가 아니며, 컴파일에는 별도 배포한 Overleaf ZIP의 IEEE Access 클래스와 지원 파일이 필요하다.

아직 받지 않은 v6c 행별 CSV, 모델 가중치, Reddit 원문 전체는 이 폴더에 포함하지 않는다. 아래 설명과 작업 목록은 검토 완료 범위를 구분하기 위한 것이다.

## Manuscript Notes

In the complete Overleaf package, open `main.tex` and compile with pdfLaTeX.
All referenced figures are vector PDFs with selectable text. The original
reviewed literature and reference entries are preserved. The final SD protocol
is v6c; the former independent-countercheck SD is an appendix variant only.

This is an updated review manuscript, not a declaration that every release
check is complete. Read `UPDATE_AND_REMAINING_KO.md` before submission. The
remaining work includes final row-level SD output reconciliation, aligned
between-method statistical comparisons, and author information.

The evidence folder contains aggregate results and calibration provenance;
it is not a redistribution of the Reddit post corpus. Prompt text is extracted
from notebook string literals, and recorded plans are extracted from saved
notebook output. Nominal generation budgets are disclosed separately.

The existing template produces header/footer overfull-box warnings that also
occurred before this revision. These are distinct from new body/table overflow;
page rendering is reviewed separately.
