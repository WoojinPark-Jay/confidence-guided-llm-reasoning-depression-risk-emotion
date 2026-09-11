# 논문에서 옮겨 보관하는 실행 기록

수치는 기존 Colab 집계표와 실행 lock에서 확인한 기록이다. 원시 생성 trace 전체의 시간·토큰 재합산은 아직 완료하지 않았다. 논문에서는 이 표로 GPU/금전 절감 효과를 주장하지 않는다.

| 항목 | Llama 2 CoT | Llama 3 SELF-DISCOVER |
|---|---:|---:|
| routed 게시물 | 137 | 137 |
| 생성 호출 | 685 | 548 |
| 입력 토큰 | 583,566 | 732,956 |
| 생성 토큰 | 100,840 | 199,716 |
| 생성 시간 합계, 초 | 5502.859108 | 11372.39543 |
| 게시물별 경과시간 합계, 초 | 5504.926802 | 11375.39771 |

성공적으로 저장한 호출만 집계했다. 다운로드·모델 로딩·재시도·전체 시스템 에너지는 포함하지 않는다. 전체 9,000건 all-input LLM baseline이 없으므로 98.48% 재평가 회피를 같은 비율의 시간 절감으로 해석할 수 없다. Phase 1 forward 기록은 23.604693207998253초, non-padding 입력 토큰 743,332다.

## 설정

- T = 1.7705598383050807, threshold = 0.70, Phase 1 max_length = 256, batch = 32.
- Llama 2: NousResearch/Llama-2-7b-chat-hf; revision `351844e75ed0bcbbe3f10671b3c808d2b83894ee`.
- Llama 3: NousResearch/Meta-Llama-3-8B-Instruct; revision `53346005fb0ef11d3b6a83b12c895cca40156b6c`.
- revision은 이번 실행에서 고정했다. 과거 LLM commit hash는 없어 과거와 byte-identical 가중치를 확인한 것은 아니다.
- 개별 seed: `SHA256(42:model_key:example_id)`의 첫 32비트. 재개 순서에 관계없이 같은 example/model에 같은 seed를 배정한다.
- GPU NVIDIA L4; CUDA 12.8; Python 3.13.15; torch 2.11.0+cu128; transformers 5.14.1; accelerate 1.14.0; bitsandbytes 0.50.2.
- bundle manifest SHA-256: `ec109e76ab8d6e9c7fcebf9adf20da30b4bfc6019a8b8b0dcf5d6db0f5666012`.

이 문서는 aggregate 및 실행 설정만 포함한다. 원문 게시물, 개별 예측, 인증정보, 모델 가중치는 포함하지 않는다.
