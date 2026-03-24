"""
Qwen3 TTS - HTTP API (OpenAI-compatible)
- POST http://172.31.79.202:8000/v1/audio/speech
- response_format: pcm → 스트리밍 raw PCM (24000Hz, 16-bit, Mono)
- TTFA(Time to First Audio) 측정
- pyaudio 실시간 재생 + WAV 저장
"""

import os
import time
import wave
import queue
import threading
import logging
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 설정 ──────────────────────────────────────────────────────────────────────
API_URL  = "http://172.31.79.202:8000/v1/audio/speech"
API_KEY  = "dummy"
MODEL    = "tts-1"

# 오디오 스펙 (qwen3-tts 고정값)
SAMPLE_RATE  = 24000
CHANNELS     = 1
SAMPLE_WIDTH = 2   # 16-bit = 2 bytes
# CHUNK_SIZE   = 4096
CHUNK_SIZE   = 1024
# ─────────────────────────────────────────────────────────────────────────────

# pyaudio 사용 가능 여부 확인
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("pyaudio 미설치 — WAV 저장만 수행합니다.")


def synthesize(
    text: str,
    voice: str = "Vivian",
    save_wav: bool = True,
    play_audio: bool = True,
    out_dir: str = "out_wav",
) -> str | None:
    """
    Qwen3 TTS HTTP API 호출 — 스트리밍 수신, 실시간 재생, WAV 저장

    Args:
        text:       합성할 텍스트
        voice:      음성 (Vivian / Ryan / Dylan / Serena 등)
        save_wav:   WAV 파일 저장 여부
        play_audio: pyaudio 실시간 재생 여부
        out_dir:    WAV 저장 디렉토리

    Returns:
        저장된 WAV 파일 경로 (save_wav=False 이면 None)
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "voice": voice,
        "input": text,
        "response_format": "pcm",
    }

    logger.info("=== Qwen3 TTS 시작 ===")
    logger.info("voice : %s", voice)
    logger.info("text  : %s", text)

    pcm_buffer = bytearray()
    first_chunk = True
    ttfa_ms = None
    t_start = time.perf_counter()

    # pyaudio callback 방식 준비
    pa = stream = None
    if play_audio and PYAUDIO_AVAILABLE:
        audio_queue: queue.Queue = queue.Queue()
        done_event = threading.Event()
        _leftover = bytearray()

        def _callback(in_data, frame_count, time_info, status):
            nonlocal _leftover
            needed = frame_count * CHANNELS * SAMPLE_WIDTH
            buf = bytearray(_leftover)

            while len(buf) < needed:
                try:
                    item = audio_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:          # sentinel: 합성 완료
                    done_event.set()
                    break
                buf.extend(item)

            _leftover = buf[needed:]
            out = bytes(buf[:needed])
            if len(out) < needed:         # 데이터 부족 → silence padding
                out += b'\x00' * (needed - len(out))

            flag = (pyaudio.paComplete
                    if (done_event.is_set() and audio_queue.empty() and not _leftover)
                    else pyaudio.paContinue)
            return out, flag

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE // SAMPLE_WIDTH,
            stream_callback=_callback,
        )
        stream.start_stream()

    try:
        with requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=60) as resp:
            resp.raise_for_status()

            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue

                t_chunk = time.perf_counter()

                if first_chunk:
                    ttfa_ms = (t_chunk - t_start) * 1000
                    duration_ms = len(chunk) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH) * 1000
                    logger.info(
                        "★ TTFA (첫 청크): %.1f ms  |  PCM %.1f KB  |  오디오 길이 %.0f ms",
                        ttfa_ms, len(chunk) / 1024, duration_ms,
                    )
                    first_chunk = False
                else:
                    elapsed_ms = (t_chunk - t_start) * 1000
                    duration_ms = len(chunk) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH) * 1000
                    logger.debug(
                        "  청크  경과 %6.0f ms  |  PCM %.1f KB  |  오디오 %.0f ms",
                        elapsed_ms, len(chunk) / 1024, duration_ms,
                    )

                # 큐에 적재 (callback이 별도 스레드에서 소비)
                if stream:
                    audio_queue.put(chunk)

                pcm_buffer.extend(chunk)

        # sentinel으로 재생 완료 신호
        if stream:
            audio_queue.put(None)
            while stream.is_active():
                time.sleep(0.01)

    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        if pa:
            pa.terminate()

    t_total = (time.perf_counter() - t_start) * 1000
    total_audio_ms = len(pcm_buffer) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH) * 1000

    logger.info("─" * 60)
    logger.info("TTFA (첫 청크)  : %.1f ms", ttfa_ms or 0)
    logger.info("전체 응답시간   : %.0f ms", t_total)
    logger.info("오디오 총길이   : %.0f ms (%.2f 초)", total_audio_ms, total_audio_ms / 1000)
    logger.info("PCM 데이터      : %.1f KB", len(pcm_buffer) / 1024)

    # WAV 저장
    if save_wav and pcm_buffer:
        os.makedirs(out_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.abspath(os.path.join(out_dir, f"qwen3_{voice.lower()}_{timestamp}.wav"))
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(bytes(pcm_buffer))
        logger.info("WAV 저장 완료   : %s", out_path)
        return out_path

    return None


if __name__ == "__main__":
    test_cases = [
        ("안녕하세요 테스트입니다.", "Vivian"),
        ("안녕하세요 테스트입니다.", "Ryan"),
        ("안녕하세요 테스트입니다.", "Dylan"),
        ("안녕하세요 테스트입니다.", "Serena"),
    ]

    for text, voice in test_cases:
        print(f"\n{'=' * 60}")
        print(f"voice: {voice}")
        synthesize(text, voice=voice, play_audio=True, save_wav=True)
