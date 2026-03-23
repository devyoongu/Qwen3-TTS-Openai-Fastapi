# Deployment History — tts-dev-001

> 서버: `172.31.79.202` / Ubuntu 18.04 / RTX 3090 (24GB)

---

## 왜 Dockerfile.vllm을 사용하는가

Qwen3-TTS를 서빙하는 백엔드는 두 가지가 있다:

| | `official` 백엔드 | `vllm_omni` 백엔드 |
|--|--|--|
| 추론 방식 | HuggingFace `generate()` 직접 호출 | vLLM 엔진 위에서 `Omni` 오케스트레이터 |
| Latency | 높음 | **낮음** (KV cache, PagedAttention) |
| 설치 | 일반 pip으로 가능 | `vllm/vllm-omni` 전용 Docker 이미지 필요 |

**Latency를 낮추기 위해 `vllm_omni` 백엔드를 선택**했고, 이 백엔드는 `pip install`로 설치 불가능한 `vllm_omni` 패키지(`from vllm_omni import Omni`)를 사용한다. 이 패키지는 `vllm/vllm-omni` 베이스 이미지에만 내장되어 있어 `Dockerfile.vllm`으로 빌드해야 한다.

---

## 환경

- OS: Ubuntu 18.04.5 LTS
- GPU: NVIDIA GeForce RTX 3090 (24GB)
- 최종 드라이버: 575.57.08 (CUDA 12.9)

---

## 문제 및 해결 과정

### 1. 포트 불일치
- 앱은 컨테이너 내부 `8880`에서 실행되나 `-p 8000:8000`으로 매핑
- **수정**: `-p 8000:8880`

### 2. NVIDIA 드라이버 업그레이드 (2회)
- 초기 드라이버 525 (CUDA 12.0) → vllm-omni 이미지 실행 불가
- **1차**: 570.133.07 설치 (CUDA 12.8) — v0.11.0rc1 실행 가능하나 qwen3_tts 미지원
- **2차**: 575.57.08 설치 (CUDA 12.9) — v0.14.0 실행 가능, 최종 성공
- 설치 방법: 컨테이너 중지 후 `.run` 파일로 수동 설치

```bash
sudo sh NVIDIA-Linux-x86_64-575.57.08.run --no-questions --ui=none
sudo reboot
```

### 3. Dockerfile.vllm 수정 사항

| 항목 | 내용 |
|------|------|
| 베이스 이미지 | `vllm/vllm-omni:v0.14.0` (qwen3_tts 지원) |
| `git` 추가 | transformers git 설치에 필요 |
| `transformers` git 버전 | qwen3_tts 아키텍처 인식을 위해 최신 버전 필요 |
| flash-attn 제거 | 소스 컴파일 시 OOM 발생 → 베이스 이미지 내장 버전 사용 |
| `ENTRYPOINT []` 추가 | 베이스 이미지의 `ENTRYPOINT ["vllm"]` 충돌 방지 |

---

## 최종 실행 명령어

```bash
# 빌드
docker build -f Dockerfile.vllm -t qwen3-tts-openai .

# 실행
docker run -d --gpus all -p 8000:8880 -v ~/.cache/huggingface:/root/.cache/huggingface -e TTS_WARMUP_ON_START=false --name qwen3-tts-api qwen3-tts-openai

# 테스트
curl -X POST http://localhost:8000/v1/audio/speech -H "Authorization: Bearer dummy" -H "Content-Type: application/json" -d '{"model":"tts-1","voice":"Vivian","input":"안녕하세요 테스트입니다.","response_format":"wav"}' -o test.wav
```
