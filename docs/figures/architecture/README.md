# Confidence-Guided Two-Phase Architecture Figure

이 디렉터리는 논문 방법론의 최종 아키텍처 도식과 편집 가능한 원본을 보존한다. 현재 파일은 2026-08-29 사용자 최종 승인본이다.

## 파일

- `confidence_guided_two_phase_architecture.drawio`: 최종 승인된 편집 원본. diagrams.net 또는 Draw.io에서 열어 문구, 연결선, 색상과 레이아웃을 수정할 수 있다.
- `confidence_guided_two_phase_architecture.pdf`: 논문에 삽입한 publication-ready 벡터 PDF. 도식 텍스트를 검색, 선택, 복사할 수 있으며 확대해도 선과 글자가 래스터 이미지처럼 흐려지지 않는다.

## 재현 방법

1. `.drawio` 파일을 diagrams.net에서 연다.
2. `File > Export as > PDF`를 선택한다.
3. `Crop`을 활성화하고 PDF를 내보낸다.
4. 내보낸 PDF에서 전체 도식, 화살표, 수식, 박스 텍스트가 잘리지 않았는지 확인한다.
5. 논문에서는 전체 폭 도식으로 삽입한다.

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/confidence_guided_two_phase_architecture.pdf}
\caption{Confidence-guided two-phase pipeline.}
\label{fig:architecture}
\end{figure*}
```

최종 PDF는 벡터 객체로 구성되고, 포함된 도식 텍스트가 실제 PDF 텍스트로 추출되는지 확인한 버전이다.
