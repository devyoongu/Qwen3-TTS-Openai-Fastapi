"""
Qwen3-TTS: PCM 스트리밍 + 상세 Latency 분석
- response_format="pcm" + httpx.stream() 으로 TTFA 단축
- 첫 PCM 청크 수신 즉시 재생 시작
"""

import time
import threading
import queue
import httpx
import sounddevice as sd
import numpy as np
from pathlib import Path

BASE_URL = "http://172.31.88.110:8880/v1"
TEXT = "안녕하세요. Qwen3 TTS 테스트입니다. 지금 바로 재생됩니다."

# Qwen3-TTS PCM 스펙: 24kHz, 16-bit, mono
SAMPLE_RATE = 24000
CHANNELS = 1
DTYPE = np.int16
BYTES_PER_SAMPLE = 2

# 스트리밍 청크 크기 (bytes): 4800 = 0.1초 분량
PLAY_CHUNK_BYTES = 4800


def stream_and_play(text: str, save_path: Path):
    """PCM 스트리밍 수신 + 실시간 재생 + latency 측정"""

    audio_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=32)
    pcm_buffer = bytearray()

    # --- 타임스탬프 공유 ---
    ts = {
        "request_sent": 0.0,
        "first_byte": 0.0,
        "first_audio": 0.0,
        "last_byte": 0.0,
        "playback_done": 0.0,
        "bytes_received": 0,
    }

    def producer():
        """HTTP 스트림 수신 → audio_queue에 넣기"""
        leftover = b""
        first = True

        with httpx.stream(
            "POST",
            f"{BASE_URL}/audio/speech",
            json={
                "model": "tts-1",
                "voice": "Vivian",
                "input": text,
                "response_format": "pcm",
            },
            headers={"Authorization": "Bearer dummy"},
            timeout=60,
        ) as resp:
            resp.raise_for_status()

            for chunk in resp.iter_bytes(chunk_size=PLAY_CHUNK_BYTES):
                if not chunk:
                    continue

                if first:
                    ts["first_byte"] = time.perf_counter()
                    first = False

                ts["bytes_received"] += len(chunk)
                pcm_buffer.extend(chunk)

                # leftover + chunk → 완전한 샘플 단위로 분리
                data = leftover + chunk
                usable = len(data) - (len(data) % BYTES_PER_SAMPLE)
                if usable > 0:
                    arr = np.frombuffer(data[:usable], dtype=DTYPE).copy()
                    audio_queue.put(arr)
                    leftover = data[usable:]
                else:
                    leftover = data

        # 남은 leftover 처리
        if leftover:
            usable = len(leftover) - (len(leftover) % BYTES_PER_SAMPLE)
            if usable > 0:
                arr = np.frombuffer(leftover[:usable], dtype=DTYPE).copy()
                audio_queue.put(arr)

        ts["last_byte"] = time.perf_counter()
        audio_queue.put(None)  # sentinel

    def consumer():
        """audio_queue에서 꺼내 sounddevice로 재생"""
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=PLAY_CHUNK_BYTES // BYTES_PER_SAMPLE,
        )
        stream.start()

        first = True
        while True:
            arr = audio_queue.get()
            if arr is None:
                break
            if first:
                ts["first_audio"] = time.perf_counter()
                first = False
            stream.write(arr)

        stream.stop()
        stream.close()
        ts["playback_done"] = time.perf_counter()

    # --- 실행 ---
    ts["request_sent"] = time.perf_counter()

    prod_thread = threading.Thread(target=producer, daemon=True)
    cons_thread = threading.Thread(target=consumer, daemon=True)

    prod_thread.start()
    cons_thread.start()

    prod_thread.join()
    cons_thread.join()

    # --- PCM → WAV 저장 ---
    if pcm_buffer:
        import wave
        save_path.parent.mkdir(exist_ok=True)
        with wave.open(str(save_path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(BYTES_PER_SAMPLE)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(bytes(pcm_buffer))

    return ts


def print_report(ts: dict):
    t0 = ts["request_sent"]
    ttfb = ts["first_byte"] - t0
    ttfa = ts["first_audio"] - t0
    total_recv = ts["last_byte"] - t0
    total_done = ts["playback_done"] - t0

    audio_seconds = ts["bytes_received"] / (SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS)
    rtf = total_recv / audio_seconds if audio_seconds > 0 else 0

    print("\n[Latency 분석]")
    print(f"  TTFB (첫 바이트):   {ttfb:.3f}s  ← 서버 생성 시작까지")
    print(f"  TTFA (첫 재생):     {ttfa:.3f}s  ← 실제 소리 시작")
    print(f"  전체 수신:          {total_recv:.3f}s")
    print(f"  오디오 길이:        {audio_seconds:.2f}s")
    print(f"  RTF (수신/생성):    {rtf:.2f}x  {'(스트리밍 가능)' if rtf < 1.5 else '(서버 느림)'}")
    print(f"  재생 완료:          {total_done:.3f}s")
    print(f"  수신 bytes:         {ts['bytes_received']:,}")


def main():
    save_path = Path("out_wav/qwen3_latency.wav")

    print(f"텍스트: {TEXT}")
    print(f"서버:   {BASE_URL}")
    print("스트리밍 시작...\n")

    ts = stream_and_play(TEXT, save_path)

    print_report(ts)
    print(f"\n저장: {save_path}")


if __name__ == "__main__":
    main()
