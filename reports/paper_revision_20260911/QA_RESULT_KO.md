# 2026-09-11 성능·효율성 반영본 검수 결과

## 산출물

- PDF: `CGSLR_IEEE_Access_Phase1_Efficiency_Updated_2026_09_11.pdf`, 29쪽.
- Overleaf: `CGSLR_IEEE_Access_Overleaf_Phase1_Efficiency_2026_09_11.zip`.
- 수정 보고서: `REVISION_REPORT_KO.md`.
- 실제 문단·표 수정 전후: `SENTENCE_CHANGES_KO.md`.
- 전체 LaTeX 차이: `MANUSCRIPT_FULL_DIFF.md`.

## 내용 및 소스 검수

- Table 1 및 Figure 2의 평균·CI를 2026-09-11 결과로 갱신했다. Accuracy와 macro F1의 CI는 별도로 반영했다.
- 기존 operational accuracy 96.69%는 원고 내 11곳 모두 유지했다.
- Table 2~7의 LaTeX 본문은 직전 버전과 완전히 동일하다.
- Figure 2를 제외한 그림과 최종 draw.io는 파일 바이트 비교로 동일함을 확인했다.
- 본문 표·그림 번호는 모두 유지했다. Appendix A1d를 새로 추가했다.
- 레퍼런스 46개와 순서는 직전 원고와 동일하다.
- 학습 3시드 변동성과 타이밍 3회 반복 변동성을 구별했다.
- 진행 중인 9,000건 추가 평가 수치를 넣지 않았다.

## 빌드 및 배치 검수

- pdfLaTeX 3회 컴파일 완료. undefined reference 및 미해결 `??` 없음.
- 전체 29쪽을 렌더링하고 접촉 시트로 배치 검토했다.
- 주요 변경 페이지 12쪽(Table 1 및 Figure 2), 21쪽(Appendix A1d)는 최종 수정 후 확대 렌더로 재확인했다.
- Table 1의 성능 패널을 전체 폭으로 정렬했고, 부록 효율성 표는 `small` 크기로 조정했다.
- 변경 표·그림에서 글자 잘림, 박스·선과 텍스트의 겹침은 발견되지 않았다.
- Figure 2 PDF에서 96.88, 95.36, 95.17이 각 두 차례 텍스트로 추출됨을 확인했다. SVG 원본도 함께 포함했다.
- 기존 IEEE Access 템플릿의 output-routine hbox 경고, 첫 페이지 hbox 경고 및 약 1.2pt 이하 vbox 경고는 남아 있다. 렌더링에서 내용 침범이나 잘림은 확인되지 않았다. 경고가 전혀 없는 빌드라고 주장하지 않는다.
- Overleaf ZIP의 무결성 및 패키지 안 `main.tex`와 검수한 소스의 일치를 확인했다.

## 제출 전 별도 항목

저자·소속·교신·펀딩 등의 기존 placeholder와 학술·윤리 관련 최종 저자 결정은 유지했다. 이번 산출물은 성능·효율성 반영을 완료한 검토 원고이다. 현재 진행 중인 추가 검증 실험은 완료 후 다음 버전에 반영한다.
