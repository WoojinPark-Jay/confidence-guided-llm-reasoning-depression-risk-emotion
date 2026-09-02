# 연구 진행 체크포인트

최종 갱신: 2026-09-03

파일명 유지 이유: 기존 회의 링크의 안정성을 보존하기 위해 날짜가 포함된 경로를 그대로 사용한다.

> 최신 단일 기준 문서는 [`final_research_status_and_remaining_work_ko.md`](final_research_status_and_remaining_work_ko.md)다.

## 현재 결론

- DistilBERT, Mistral 7B, Llama 2 7B의 동일 12,000건 held-out test, 3-seed Phase 1 비교를 완료했다.
- Accuracy/macro F1은 DistilBERT `96.70 +/- 0.10%`, Mistral `95.56 +/- 0.11%`, Llama 2 `95.09 +/- 0.06%`다.
- Mistral과 Llama 2는 frozen backbone + 12,288-parameter linear probe이고, DistilBERT는 full fine-tuning이다.
- 고정 DistilBERT operational checkpoint는 Reddit 96.69%, `T*=1.7706`, `alpha=0.05`, `tau*=0.70`이다.
- Reddit 12,000건 중 171건을 routing했으며 Llama 3 end-to-end accuracy는 96.94%(+0.25 pp, net +30)다.
- Mixed Emotion 300건 중 44건을 routing했으며 Llama 3 end-to-end accuracy는 87.33%(+6.00 pp, net +18)다.
- Paired bootstrap, exact McNemar, Holm correction, accepted high-confidence error audit를 완료했다.
- 최종 architecture vector/Draw.io와 IEEE Access 원고 패키지를 정리했다.

## 남은 필수 작업

1. 저자, 소속, 이메일, corresponding author, ORCID와 biographies 확정
2. Funding/Acknowledgment, conflict, AI-use, ethics, data/code availability 문구 확정
3. 공개 artifact와 release tag/DOI 범위 확정
4. 메타데이터 입력 후 Overleaf pdfLaTeX clean compile과 전 페이지 최종 확인
5. Source/PDF/supplementary/run manifest/checksum 동결

결과표, 방법론 해석, 완료 단계와 권장 후속 작업은 최신 기준 문서에서 확인한다.
