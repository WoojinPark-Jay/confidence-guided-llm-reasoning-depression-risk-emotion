# 2026-09-11 Phase 1 성능 및 GPU 효율성 반영본

## 공동 검토 순서

1. [변경 요약 및 근거](REVISION_REPORT_KO.md): 이번 단계의 수치, 설정, 본문 및 부록 변경과 유지한 결과.
2. [실제 문장·표 수정 전후 비교](SENTENCE_CHANGES_KO.md): 기준 원고 대비 변경된 문장과 표 전체.
3. [논문 PDF](CGSLR_IEEE_Access_Phase1_Efficiency_Updated_2026_09_11.pdf): 29쪽 조판본.
4. [Overleaf ZIP](CGSLR_IEEE_Access_Overleaf_Phase1_Efficiency_2026_09_11.zip): main.tex, 벡터 그림 및 편집 가능한 도식 원본을 포함한 재빌드 패키지.

## 보조 자료

- [LaTeX 전체 diff](MANUSCRIPT_FULL_DIFF.md)
- [검수 결과](QA_RESULT_KO.md)
- [정리된 최종 수치 JSON](phase1_final_results.json), [CSV](phase1_final_results.csv)
- [패키지 및 PDF SHA-256](release_sha256.json)

공동연구자가 완료한 분류 성능 및 GPU 효율성 측정을 정식 연구 결과로 반영한 버전이다. Phase 1 비교 성능과 Figure 2를 갱신하고, Table 1 및 Appendix A1d에 효율성 측정을 추가했다. 기존 operational DistilBERT checkpoint, calibration/routing 및 downstream 결과는 변경하지 않았다.

현재 별도로 진행 중인 9,000건 추가 평가는 이번 원고와 분리한다. 해당 결과가 완료되면 다음 단계로 업데이트한다. 이번 파일은 현재 단계의 논문 업데이트이며, 저자·소속·지원 정보 등 제출 행정 항목의 최종 확정을 뜻하지 않는다.
