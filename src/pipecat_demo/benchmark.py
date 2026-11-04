"""Headless full-audio cascade benchmark and single-turn voice simulator (16kHz/24kHz & 8kHz Telephony)."""

from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import io
import struct
import time
import uuid
import wave
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    EndFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from pipecat_demo.config import CascadeConfig, TurnStopMode, load_config
from pipecat_demo.latency import (
    LatencyStore,
    LatencySummary,
    TurnLatencyRecord,
    format_turn_report,
)
from pipecat_demo.pipeline import (
    build_transport_params,
    create_cascade_session,
    run_cascade_worker,
)
from pipecat_demo.telephony import (
    STT_PIPELINE_SAMPLE_RATE,
    TELEPHONY_SAMPLE_RATE,
    TTS_NATIVE_SAMPLE_RATE,
    DialerEncoding,
    normalize_dialer_encoding,
    simulate_pstn_8k_roundtrip,
)

DEFAULT_BENCHMARK_QUESTIONS: tuple[str, ...] = (
    "Why does ice float on liquid water instead of sinking?",
    "How do noise cancelling headphones block out background sound?",
    "What causes the northern lights in Earth's polar skies?",
    "What is the difference between a solar eclipse and a lunar eclipse?",
)


def pcm16le_to_wav_bytes(
    pcm_bytes: bytes,
    *,
    sample_rate: int = 24000,
    num_channels: int = 1,
) -> bytes:
    """Wrap raw 16-bit little-endian mono PCM bytes in a standard RIFF WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def strip_wav_header_if_present(raw_audio: bytes, *, default_rate: int = 24000) -> tuple[bytes, int]:
    """Extract raw PCM16LE bytes and sample rate from a standard or streaming RIFF WAV buffer."""
    if len(raw_audio) < 44 or raw_audio[:4] != b"RIFF" or raw_audio[8:12] != b"WAVE":
        return raw_audio, default_rate

    sample_rate = default_rate
    offset = 12
    total_len = len(raw_audio)
    while offset + 8 <= total_len:
        chunk_id, chunk_size = struct.unpack_from("<4sI", raw_audio, offset)
        payload_start = offset + 8
        if chunk_id == b"fmt " and payload_start + 8 <= total_len:
            sample_rate = struct.unpack_from("<I", raw_audio, payload_start + 4)[0] or default_rate
        elif chunk_id == b"data":
            if chunk_size in (0, 0xFFFFFFFF) or payload_start + chunk_size > total_len:
                return raw_audio[payload_start:], sample_rate
            return raw_audio[payload_start : payload_start + chunk_size], sample_rate

        if chunk_size == 0xFFFFFFFF:
            break
        offset = payload_start + chunk_size + (chunk_size % 2)

    return raw_audio[44:], sample_rate


async def synthesize_question_pcm16k(
    question: str,
    cfg: CascadeConfig,
    *,
    voice: str = "Puck",
    leading_silence_ms: int = 240,
    trailing_silence_ms: int = 1100,
) -> bytes:
    """Synthesize spoken 16kHz PCM16LE user question audio via Gemini 3.8 Flash-Lite TTS."""
    api_key = cfg.require_api_key()
    client = genai.Client(api_key=api_key)

    is_gemini_38 = "3.8" in cfg.tts_model
    if is_gemini_38:
        contents: Any = [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=question,
                        speech_metadata=types.SpeechMetadata(
                            style="Clear, natural everyday question spoken at a conversational pace."
                        ),
                    )
                ],
            )
        ]
    else:
        contents = question

    gen_config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            language_code="en-US",
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            ),
        ),
    )

    stream = await client.aio.models.generate_content_stream(
        model=cfg.tts_model,
        contents=contents,
        config=gen_config,
    )

    raw_buf = bytearray()
    async for chunk in stream:
        if not (
            chunk.candidates
            and chunk.candidates[0].content
            and chunk.candidates[0].content.parts
        ):
            continue
        for part in chunk.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                raw_buf.extend(part.inline_data.data)

    if not raw_buf:
        raise RuntimeError("Gemini TTS returned empty audio while synthesizing test question.")

    pcm_src, src_rate = strip_wav_header_if_present(
        bytes(raw_buf), default_rate=TTS_NATIVE_SAMPLE_RATE
    )
    if len(pcm_src) % 2 == 1:
        pcm_src = pcm_src[:-1]

    resampler = create_stream_resampler()
    pcm_16k = await resampler.resample(pcm_src, src_rate, STT_PIPELINE_SAMPLE_RATE)
    pcm_16k += await resampler.flush()

    bytes_per_ms = (STT_PIPELINE_SAMPLE_RATE * 2) // 1000  # 32 bytes/ms at 16kHz int16 mono
    lead = bytes(leading_silence_ms * bytes_per_ms)
    trail = bytes(trailing_silence_ms * bytes_per_ms)
    return lead + pcm_16k + trail


async def synthesize_question_telephony_8k(
    question: str,
    cfg: CascadeConfig,
    *,
    encoding: DialerEncoding | None = None,
    voice: str = "Puck",
    leading_silence_ms: int = 240,
    trailing_silence_ms: int = 1100,
) -> bytes:
    """Synthesize spoken question audio and pass it through an 8kHz PSTN dialer codec roundtrip."""
    pcm_16k = await synthesize_question_pcm16k(
        question,
        cfg,
        voice=voice,
        leading_silence_ms=leading_silence_ms,
        trailing_silence_ms=trailing_silence_ms,
    )
    active_encoding = normalize_dialer_encoding(encoding, default=cfg.dialer_encoding)
    return await simulate_pstn_8k_roundtrip(
        pcm_16k,
        encoding=active_encoding,
        dialer_rate=cfg.dialer_sample_rate,
    )


class HeadlessInputTransport(BaseInputTransport):
    """In-memory microphone/dialer transport that paces 8kHz or 16kHz PCM chunks in real time."""

    def __init__(self, params: TransportParams):
        super().__init__(params)
        self.ready_event = asyncio.Event()

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self.set_transport_ready(frame)
        self.ready_event.set()

    async def feed_pcm16k_realtime(
        self,
        pcm_bytes: bytes,
        *,
        sample_rate: int | None = None,
        chunk_ms: int = 20,
    ) -> None:
        """Stream mono int16 PCM at `sample_rate` (8kHz or 16kHz) in real-time 20ms frames."""
        active_rate = sample_rate or self._params.audio_in_sample_rate or 16000
        chunk_bytes = (active_rate * 2 * chunk_ms) // 1000
        interval_s = chunk_ms / 1000.0
        next_deadline = time.perf_counter()

        for offset in range(0, len(pcm_bytes), chunk_bytes):
            chunk = pcm_bytes[offset : offset + chunk_bytes]
            if len(chunk) < chunk_bytes:
                chunk = chunk + bytes(chunk_bytes - len(chunk))
            await self.push_audio_frame(
                InputAudioRawFrame(
                    audio=chunk,
                    sample_rate=active_rate,
                    num_channels=1,
                )
            )
            next_deadline += interval_s
            sleep_for = next_deadline - time.perf_counter()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)


class HeadlessOutputTransport(BaseOutputTransport):
    """In-memory speaker/dialer transport that captures bot audio (8kHz or 24kHz) and speaking frames."""

    def __init__(self, params: TransportParams):
        super().__init__(params)
        self.ready_event = asyncio.Event()
        self._turn_pcm = bytearray()

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self.set_transport_ready(frame)
        self.ready_event.set()

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        self._turn_pcm.extend(frame.audio)
        await asyncio.sleep(0)
        return True

    def take_turn_audio(self) -> bytes:
        """Return and clear the recorded PCM16LE audio for the completed turn."""
        data = bytes(self._turn_pcm)
        self._turn_pcm.clear()
        return data


class HeadlessBenchmarkTransport(BaseTransport):
    """Transport pair driving the real Gemini STT -> LLM -> TTS cascade without WebRTC."""

    def __init__(self, params: TransportParams):
        super().__init__()
        self._input = HeadlessInputTransport(params)
        self._output = HeadlessOutputTransport(params)

    def input(self) -> HeadlessInputTransport:
        return self._input

    def output(self) -> HeadlessOutputTransport:
        return self._output

    async def wait_until_ready(self, *, timeout: float = 20.0) -> None:
        await asyncio.wait_for(
            asyncio.gather(
                self._input.ready_event.wait(),
                self._output.ready_event.wait(),
            ),
            timeout=timeout,
        )


async def run_benchmark_suite(
    questions: list[str],
    cfg: CascadeConfig,
    *,
    store: LatencyStore,
    save_wav_dir: Path | None = None,
    log_reports: bool = True,
    telephony_8k: bool | None = None,
    dialer_encoding: str | None = None,
) -> tuple[list[TurnLatencyRecord], list[bytes], list[bytes]]:
    """Run one or more spoken questions sequentially through a live cascade session."""
    cfg.require_api_key()
    if save_wav_dir is not None:
        save_wav_dir.mkdir(parents=True, exist_ok=True)

    use_telephony_8k = cfg.telephony_8k_enabled if telephony_8k is None else telephony_8k
    active_cfg = cfg.with_telephony_8k(enabled=use_telephony_8k, encoding=dialer_encoding)
    input_feed_rate = active_cfg.dialer_sample_rate if use_telephony_8k else active_cfg.audio_in_sample_rate
    in_tag = f"{input_feed_rate // 1000}k"
    out_tag = f"{active_cfg.audio_out_sample_rate // 1000}k"

    # Pre-synthesize all test questions up front so test-harness TTS generation
    # never overlaps or interferes with the measured cascade pipeline session.
    synthesized_inputs: list[bytes] = []
    for idx, question in enumerate(questions, start=1):
        if use_telephony_8k:
            logger.info(
                f"[Benchmark] Pre-synthesizing 8kHz telephony question #{idx} "
                f"(codec={active_cfg.dialer_encoding}): {question!r}"
            )
            pcm_in = await synthesize_question_telephony_8k(
                question,
                active_cfg,
                encoding=active_cfg.dialer_encoding,
            )
        else:
            logger.info(f"[Benchmark] Pre-synthesizing 16kHz wideband question #{idx}: {question!r}")
            pcm_in = await synthesize_question_pcm16k(question, active_cfg)

        synthesized_inputs.append(pcm_in)
        if save_wav_dir is not None:
            (save_wav_dir / f"turn_{idx:02d}_user_{in_tag}.wav").write_bytes(
                pcm16le_to_wav_bytes(pcm_in, sample_rate=input_feed_rate)
            )

    session_id = f"bench-{uuid.uuid4().hex[:8]}"
    transport = HeadlessBenchmarkTransport(build_transport_params(active_cfg))
    worker, monitor = create_cascade_session(
        transport,
        active_cfg,
        store=store,
        session_id=session_id,
        greet_on_connect=False,
        emit_rtvi_messages=False,
        log_reports=log_reports,
        cancel_on_idle_timeout=True,
    )

    worker_task = asyncio.create_task(run_cascade_worker(worker, handle_sigint=False))
    completed_records: list[TurnLatencyRecord] = []
    bot_pcm_outputs: list[bytes] = []

    try:
        await transport.wait_until_ready(timeout=25.0)
        # Allow Gemini Live STT WebSocket session to settle before streaming speech
        await asyncio.sleep(0.25)

        for turn_idx, (question, pcm_in) in enumerate(
            zip(questions, synthesized_inputs, strict=True), start=1
        ):
            logger.info(
                f"[Benchmark] Streaming spoken question #{turn_idx}/{len(questions)} "
                f"({len(pcm_in) / (input_feed_rate * 2):.2f}s @ {input_feed_rate}Hz): {question!r}"
            )
            await transport.input().feed_pcm16k_realtime(pcm_in, sample_rate=input_feed_rate)
            record = await monitor.wait_for_turn(turn_idx, timeout=45.0)
            bot_pcm = transport.output().take_turn_audio()
            completed_records.append(record)
            bot_pcm_outputs.append(bot_pcm)

            if save_wav_dir is not None and bot_pcm:
                (save_wav_dir / f"turn_{turn_idx:02d}_bot_{out_tag}.wav").write_bytes(
                    pcm16le_to_wav_bytes(bot_pcm, sample_rate=active_cfg.audio_out_sample_rate)
                )
            if turn_idx < len(questions):
                await asyncio.sleep(0.35)
    finally:
        await monitor.flush_open_turn()
        try:
            await worker.queue_frame(EndFrame())
        except Exception:
            await worker.cancel()
        try:
            await asyncio.wait_for(worker_task, timeout=15.0)
        except TimeoutError:
            await worker.cancel()
            await asyncio.gather(worker_task, return_exceptions=True)

    return completed_records, synthesized_inputs, bot_pcm_outputs


async def run_simulated_voice_turn(
    question: str,
    cfg: CascadeConfig,
    *,
    store: LatencyStore,
    telephony_8k: bool | None = None,
    dialer_encoding: str | None = None,
) -> dict[str, Any]:
    """Execute one full spoken question turn and return latency breakdown + WAV audio."""
    use_telephony_8k = cfg.telephony_8k_enabled if telephony_8k is None else telephony_8k
    active_cfg = cfg.with_telephony_8k(enabled=use_telephony_8k, encoding=dialer_encoding)
    user_sample_rate = (
        active_cfg.dialer_sample_rate if use_telephony_8k else active_cfg.audio_in_sample_rate
    )
    bot_sample_rate = active_cfg.audio_out_sample_rate

    records, user_pcms, bot_pcms = await run_benchmark_suite(
        [question],
        active_cfg,
        store=store,
        log_reports=True,
        telephony_8k=use_telephony_8k,
        dialer_encoding=active_cfg.dialer_encoding,
    )
    if not records:
        raise RuntimeError("Cascade pipeline completed without producing a turn latency record.")

    record = records[0]
    user_wav = pcm16le_to_wav_bytes(user_pcms[0], sample_rate=user_sample_rate)
    bot_wav = (
        pcm16le_to_wav_bytes(bot_pcms[0], sample_rate=bot_sample_rate)
        if bot_pcms[0]
        else b""
    )
    return {
        "turn": record.model_dump(),
        "summary": store.summary().model_dump(),
        "telephony_8k": use_telephony_8k,
        "dialer_encoding": active_cfg.dialer_encoding,
        "user_sample_rate_hz": user_sample_rate,
        "stt_pipeline_sample_rate_hz": active_cfg.audio_in_sample_rate,
        "tts_native_sample_rate_hz": TTS_NATIVE_SAMPLE_RATE,
        "bot_sample_rate_hz": bot_sample_rate,
        "user_audio_wav_base64": base64.b64encode(user_wav).decode("ascii"),
        "bot_audio_wav_base64": base64.b64encode(bot_wav).decode("ascii") if bot_wav else None,
    }


def format_suite_summary(summary: LatencySummary, cfg: CascadeConfig) -> str:
    """Format an end-of-benchmark statistical summary table across all measured turns."""
    rows = [
        ("E2E Perceived (from silence)", summary.e2e_from_user_silence),
        ("Net Cascade (after VAD stop)", summary.e2e_from_vad_stop),
        ("1. VAD Endpointing Wait", summary.endpointing_wait),
        ("2. STT Finalize (post-silence)", summary.stt_finalize),
        ("   STT Service TTFB", summary.stt_ttfb),
        ("3. SmartTurn v3 Detection", summary.turn_detection),
        ("4. LLM TTFB (first token)", summary.llm_ttfb),
        ("   LLM TTFAT (actionable text)", summary.llm_ttfat),
        ("   LLM Total Stream Time", summary.llm_total_processing),
        ("5. Sentence Aggregation", summary.sentence_aggregation),
        ("6. TTS TTFB (first audio byte)", summary.tts_ttfb),
        ("   TTS TTFA (first audible)", summary.tts_ttfa),
        ("   TTS Total Synthesis Time", summary.tts_total_processing),
        ("7. Output Transport Framing", summary.output_transport),
        ("8. Pipeline Queue Overhead", summary.pipeline_overhead),
    ]
    audio_path_desc = (
        f"8kHz Dialer ({cfg.dialer_encoding.upper()}) -> 16kHz STT -> 24kHz TTS -> 8kHz Dialer"
        if cfg.telephony_8k_enabled
        else f"{cfg.audio_in_sample_rate}Hz In -> {cfg.audio_out_sample_rate}Hz Out (Wideband)"
    )
    lines = [
        "",
        "══════════════════════════════════════════════════════════════════════════════════",
        f" CASCADE BENCHMARK SUMMARY (n={summary.user_turn_count} spoken turns)",
        f" Models: STT={cfg.stt_model} | LLM={cfg.llm_model} ({cfg.llm_thinking_level}) | TTS={cfg.tts_model}",
        f" Audio Path:  {audio_path_desc}",
        f" Endpointing: mode={cfg.turn_stop_mode} | vad_stop_secs={cfg.vad_stop_secs:.2f}s",
        "──────────────────────────────────────────────────────────────────────────────────",
        f" {'Metric / Stage':<32} │ {'Mean':>9} │ {'P50':>9} │ {'P95':>9} │ {'Min':>9} │ {'Max':>9}",
        "──────────────────────────────────┼───────────┼───────────┼───────────┼───────────┼───────────",
    ]
    for label, stat in rows:
        if stat.count == 0:
            lines.append(
                f" {label:<32} │ {'—':>9} │ {'—':>9} │ {'—':>9} │ {'—':>9} │ {'—':>9}"
            )
        else:
            lines.append(
                f" {label:<32} │ {stat.mean_ms:>7.1f}ms │ {stat.p50_ms:>7.1f}ms │ "
                f"{stat.p95_ms:>7.1f}ms │ {stat.min_ms:>7.1f}ms │ {stat.max_ms:>7.1f}ms"
            )
    lines.append("══════════════════════════════════════════════════════════════════════════════════")
    return "\n".join(lines)


async def _async_main(args: argparse.Namespace) -> int:
    cfg = load_config()
    if args.turn_stop_mode:
        turn_mode: TurnStopMode = "vad" if args.turn_stop_mode == "vad" else "smart_turn"
        cfg = dataclasses.replace(cfg, turn_stop_mode=turn_mode)
    if args.vad_stop_secs is not None:
        cfg = dataclasses.replace(cfg, vad_stop_secs=args.vad_stop_secs)

    telephony_8k = cfg.telephony_8k_enabled
    if args.telephony_8k:
        telephony_8k = True
    elif args.wideband:
        telephony_8k = False

    cfg = cfg.with_telephony_8k(enabled=telephony_8k, encoding=args.dialer_encoding)

    if args.questions:
        questions = list(args.questions)
    else:
        pool = list(DEFAULT_BENCHMARK_QUESTIONS)
        questions = [pool[i % len(pool)] for i in range(max(1, args.turns))]

    store = LatencyStore(jsonl_path=cfg.metrics_jsonl_path, load_existing=False)
    save_dir = Path(args.save_wav_dir) if args.save_wav_dir else None

    records, _, _ = await run_benchmark_suite(
        questions,
        cfg,
        store=store,
        save_wav_dir=save_dir,
        log_reports=False,
        telephony_8k=telephony_8k,
        dialer_encoding=cfg.dialer_encoding,
    )
    run_summary = store.summary()
    for rec in records:
        print(format_turn_report(rec, run_summary))
    print(format_suite_summary(run_summary, cfg))
    print(f"\nSaved JSONL turn records to: {cfg.metrics_jsonl_path}")
    if save_dir is not None:
        print(f"Saved user & bot WAV recordings to: {save_dir.resolve()}")
    return 0


def run() -> None:
    """CLI entrypoint for `pipecat-benchmark` and `benchmark_cascade.py`."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure real end-to-end and per-block latency across Gemini Live Transcribe -> "
            "Gemini 3.5 Flash-Lite -> Gemini 3.8 Flash-Lite TTS (8kHz Telephony or 16k/24k Wideband)."
        )
    )
    parser.add_argument(
        "-q",
        "--question",
        dest="questions",
        action="append",
        help="Spoken question to synthesize and stream through the cascade (repeatable).",
    )
    parser.add_argument(
        "-n",
        "--turns",
        "--runs",
        dest="turns",
        type=int,
        default=2,
        help="Number of default general-knowledge questions to run when --question is omitted (default: 2).",
    )
    parser.add_argument(
        "--turn-stop-mode",
        choices=["smart_turn", "vad"],
        default=None,
        help="Override TURN_STOP_MODE ('smart_turn' uses SmartTurn v3 ONNX; 'vad' uses pure silence).",
    )
    parser.add_argument(
        "--vad-stop-secs",
        type=float,
        default=None,
        help="Override Silero VAD silence window in seconds (default from .env: 0.2).",
    )
    parser.add_argument(
        "--telephony-8k",
        action="store_true",
        help="Simulate an 8kHz telephony dialer (8kHz caller input -> 16kHz STT -> 24kHz TTS -> 8kHz bot output).",
    )
    parser.add_argument(
        "--wideband",
        action="store_true",
        help="Run in wideband mode (16kHz input -> 24kHz output without 8kHz dialer conversion).",
    )
    parser.add_argument(
        "--dialer-encoding",
        choices=["ulaw", "alaw", "pcm16"],
        default=None,
        help="8kHz dialer codec to simulate when --telephony-8k is active (default from .env: ulaw).",
    )
    parser.add_argument(
        "--save-wav-dir",
        type=str,
        default=None,
        help="Optional directory to save each turn's user input WAV and bot output WAV.",
    )
    parsed = parser.parse_args()
    raise SystemExit(asyncio.run(_async_main(parsed)))


if __name__ == "__main__":
    run()
