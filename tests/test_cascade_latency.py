"""Automated tests for cascade configuration, latency waterfall tracking, WAV helpers, and HTTP routes."""

from __future__ import annotations

import struct
import tempfile
import unittest
import warnings
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ClientConnectedFrame,
    LLMContextAssistantTurnFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TextAggregationMetricsData,
    TTFAMetricsData,
    TTFATMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
    TurnMetricsData,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_demo.benchmark import pcm16le_to_wav_bytes, strip_wav_header_if_present
from pipecat_demo.config import CascadeConfig
from pipecat_demo.latency import CascadeLatencyMonitor, LatencyStore, format_turn_report
from pipecat_demo.pipeline import (
    build_transport_params,
    build_user_aggregator_params,
    build_websocket_params,
    create_cascade_services,
)
from pipecat_demo.server import register_observatory_routes


class _FakeClock:
    def __init__(self, initial: float = 1000.0):
        self.now = initial

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now = round(self.now + seconds, 6)
        return self.now


class _StubProcessor(FrameProcessor):
    def __init__(self, name: str):
        super().__init__(name=name)


class CascadeLatencyTest(unittest.IsolatedAsyncioTestCase):
    async def _emit(
        self,
        monitor: CascadeLatencyMonitor,
        src: FrameProcessor,
        dst: FrameProcessor,
        frame: object,
        clock: _FakeClock,
    ) -> None:
        pushed = FramePushed(
            source=src,
            destination=dst,
            frame=frame,  # type: ignore[arg-type]
            direction=FrameDirection.DOWNSTREAM,
            timestamp=int(clock() * 1_000_000_000),
            first_push=True,
        )
        for obs in monitor.observers():
            await obs.on_push_frame(pushed)
        await monitor.user_bot_observer.cleanup()

    def test_zero_deprecation_cascade_service_construction(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            cfg = CascadeConfig(api_key="unit-test-dummy-key")
            params = build_transport_params(cfg)
            self.assertEqual(params.audio_in_sample_rate, 16000)
            self.assertEqual(params.audio_out_sample_rate, 24000)

            user_params = build_user_aggregator_params(cfg)
            self.assertIsNotNone(user_params.vad_analyzer)

            stt, llm, tts = create_cascade_services(cfg)
            self.assertEqual(stt._settings.model, "gemini-3.5-transcribe-live")
            self.assertEqual(llm._settings.model, "gemini-3.5-flash-lite")
            self.assertEqual(tts._settings.model, "gemini-3.8-flash-lite-tts")

        pipecat_deprecations = [
            str(w.message)
            for w in caught
            if issubclass(w.category, DeprecationWarning) and "audioop" not in str(w.message)
        ]
        self.assertEqual(pipecat_deprecations, [])

    def test_wav_roundtrip_and_streaming_riff_header_strip(self) -> None:
        pcm_orig = b"\x10\x00\xf0\xff" * 480
        wav_bytes = pcm16le_to_wav_bytes(pcm_orig, sample_rate=24000)
        stripped, sr = strip_wav_header_if_present(wav_bytes)
        self.assertEqual(sr, 24000)
        self.assertEqual(stripped, pcm_orig)

        # Simulate Gemini 3.8 TTS streaming WAV header where RIFF/data sizes are 0xFFFFFFFF
        streaming_header = bytearray(wav_bytes[:44])
        struct.pack_into("<I", streaming_header, 4, 0xFFFFFFFF)
        struct.pack_into("<I", streaming_header, 40, 0xFFFFFFFF)
        stripped_stream, sr_stream = strip_wav_header_if_present(
            bytes(streaming_header) + pcm_orig
        )
        self.assertEqual(sr_stream, 24000)
        self.assertEqual(stripped_stream, pcm_orig)

    async def test_greeting_isolated_and_multi_sentence_user_turn_waterfall(self) -> None:
        clock = _FakeClock()
        src = _StubProcessor("GeminiSTTService#0")
        dst = _StubProcessor("GoogleLLMService#0")

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "latency_metrics.jsonl"
            store = LatencyStore(jsonl_path=jsonl_path, load_existing=False)
            monitor = CascadeLatencyMonitor(
                session_id="sess-12345678",
                store=store,
                stt_model="gemini-3.5-transcribe-live",
                llm_model="gemini-3.5-flash-lite",
                tts_model="gemini-3.8-flash-lite-tts",
                emit_rtvi_messages=False,
                log_reports=False,
                time_source=clock,
            )

            # 1) Initial greeting turn (should NOT count in user_turn P50/P95)
            await self._emit(monitor, src, dst, ClientConnectedFrame(), clock)
            clock.advance(0.18)
            await self._emit(monitor, src, dst, LLMTextFrame(text="Hello! Ask me anything."), clock)
            clock.advance(0.12)
            await self._emit(monitor, src, dst, BotStartedSpeakingFrame(), clock)
            clock.advance(0.60)
            await self._emit(monitor, src, dst, BotStoppedSpeakingFrame(), clock)
            await self._emit(
                monitor,
                src,
                dst,
                LLMContextAssistantTurnFrame(text="Hello! Ask me anything.", timestamp="t0"),
                clock,
            )

            snap_after_greeting = store.summary()
            self.assertEqual(snap_after_greeting.greeting_count, 1)
            self.assertEqual(snap_after_greeting.user_turn_count, 0)

            # 2) Spoken User Turn #1 (multi-sentence response)
            clock.advance(1.0)
            await self._emit(
                monitor, src, dst, VADUserStartedSpeakingFrame(timestamp=clock()), clock
            )
            clock.advance(1.4)  # User speaks for 1.2s + 0.2s VAD silence
            vad_stopped_ts = clock()
            await self._emit(
                monitor,
                src,
                dst,
                VADUserStoppedSpeakingFrame(stop_secs=0.20, timestamp=vad_stopped_ts),
                clock,
            )

            # Live STT finalizes 110ms after VAD stop (310ms after actual user silence)
            clock.advance(0.11)
            await self._emit(
                monitor,
                src,
                dst,
                MetricsFrame(
                    data=[
                        TTFBMetricsData(
                            processor="GeminiSTTService#0",
                            model="gemini-3.5-transcribe-live",
                            value=0.31,
                        ),
                        ProcessingMetricsData(
                            processor="GeminiSTTService#0",
                            model="gemini-3.5-transcribe-live",
                            value=1.51,
                        ),
                    ]
                ),
                clock,
            )
            await self._emit(
                monitor,
                src,
                dst,
                TranscriptionFrame(
                    text="Why does ice float?",
                    user_id="u1",
                    timestamp="now",
                    finalized=True,
                ),
                clock,
            )

            # SmartTurn v3 completes 15ms later
            clock.advance(0.015)
            await self._emit(
                monitor,
                src,
                dst,
                MetricsFrame(
                    data=[
                        TurnMetricsData(
                            processor="LLMUserAggregator#0",
                            is_complete=True,
                            probability=0.97,
                            e2e_processing_time_ms=14.2,
                        )
                    ]
                ),
                clock,
            )
            await self._emit(monitor, src, dst, UserStoppedSpeakingFrame(), clock)

            # LLM TTFB 220ms
            clock.advance(0.010)
            llm_start = clock()
            await self._emit(monitor, dst, dst, LLMFullResponseStartFrame(), clock)
            clock.advance(0.220)
            await self._emit(
                monitor,
                dst,
                dst,
                MetricsFrame(
                    data=[
                        TTFBMetricsData(
                            processor="GoogleLLMService#0",
                            model="gemini-3.5-flash-lite",
                            value=0.220,
                        ),
                        TTFATMetricsData(
                            processor="GoogleLLMService#0",
                            model="gemini-3.5-flash-lite",
                            ttfb=0.220,
                            ttfat=0.220,
                            thinking_time=0.0,
                        ),
                    ]
                ),
                clock,
            )
            await self._emit(
                monitor,
                dst,
                dst,
                LLMTextFrame(text="Ice forms a crystal lattice. "),
                clock,
            )

            # Sentence aggregation takes 35ms, then TTS TTFB takes 190ms
            clock.advance(0.035)
            await self._emit(
                monitor,
                src,
                dst,
                MetricsFrame(
                    data=[
                        TextAggregationMetricsData(
                            processor="GeminiTTSService#0",
                            model="gemini-3.8-flash-lite-tts",
                            value=0.035,
                        )
                    ]
                ),
                clock,
            )
            tts_start = clock()
            clock.advance(0.190)
            await self._emit(
                monitor,
                src,
                dst,
                MetricsFrame(
                    data=[
                        TTFBMetricsData(
                            processor="GeminiTTSService#0",
                            model="gemini-3.8-flash-lite-tts",
                            value=0.190,
                        ),
                        TTFAMetricsData(
                            processor="GeminiTTSService#0",
                            model="gemini-3.8-flash-lite-tts",
                            ttfb=0.190,
                            ttfa=0.198,
                            leading_silence=0.008,
                        ),
                    ]
                ),
                clock,
            )
            await self._emit(
                monitor,
                _StubProcessor("GeminiTTSService#0"),
                _StubProcessor("HeadlessOutputTransport#0"),
                TTSAudioRawFrame(audio=b"\x00\x00" * 240, sample_rate=24000, num_channels=1),
                clock,
            )

            # Output transport emits BotStartedSpeakingFrame 12ms later
            clock.advance(0.012)
            await self._emit(
                monitor,
                _StubProcessor("HeadlessOutputTransport#0"),
                dst,
                BotStartedSpeakingFrame(),
                clock,
            )

            # Second LLM sentence streams and LLM finishes
            clock.advance(0.080)
            await self._emit(
                monitor,
                src,
                dst,
                LLMTextFrame(text="That makes it less dense than water."),
                clock,
            )
            await self._emit(
                monitor,
                src,
                dst,
                MetricsFrame(
                    data=[
                        ProcessingMetricsData(
                            processor="GoogleLLMService#0",
                            model="gemini-3.5-flash-lite",
                            value=0.335,
                        ),
                        LLMUsageMetricsData(
                            processor="GoogleLLMService#0",
                            model="gemini-3.5-flash-lite",
                            value=LLMTokenUsage(
                                prompt_tokens=48,
                                completion_tokens=16,
                                total_tokens=64,
                            ),
                        ),
                    ]
                ),
                clock,
            )
            await self._emit(monitor, src, dst, LLMFullResponseEndFrame(), clock)

            # Simulate brief gap between sentence 1 and sentence 2 audio:
            # BotStoppedSpeakingFrame before LLMContextAssistantTurnFrame must NOT close the turn early!
            clock.advance(0.40)
            await self._emit(monitor, src, dst, BotStoppedSpeakingFrame(), clock)
            self.assertIsNotNone(monitor._active_record)

            # Sentence 2 TTS metrics + playback complete
            await self._emit(
                monitor,
                src,
                dst,
                MetricsFrame(
                    data=[
                        ProcessingMetricsData(
                            processor="GeminiTTSService#0",
                            model="gemini-3.8-flash-lite-tts",
                            value=0.410,
                        ),
                        TTSUsageMetricsData(
                            processor="GeminiTTSService#0",
                            model="gemini-3.8-flash-lite-tts",
                            value=65,
                        ),
                    ]
                ),
                clock,
            )
            await self._emit(monitor, src, dst, BotStartedSpeakingFrame(), clock)
            clock.advance(0.55)
            await self._emit(monitor, src, dst, BotStoppedSpeakingFrame(), clock)
            await self._emit(
                monitor,
                src,
                dst,
                LLMContextAssistantTurnFrame(
                    text="Ice forms a crystal lattice. That makes it less dense than water.",
                    timestamp="t1",
                ),
                clock,
            )

            rec = await monitor.wait_for_turn(2, timeout=1.0)
            self.assertTrue(rec.completed)
            self.assertEqual(rec.turn_type, "user_turn")
            self.assertEqual(rec.user_transcript, "Why does ice float?")
            self.assertAlmostEqual(rec.endpointing_wait_ms, 200.0, places=1)
            self.assertAlmostEqual(rec.stt_finalize_ms, 110.0, places=1)
            self.assertAlmostEqual(rec.llm_ttfb_ms or 0.0, 220.0, places=1)
            self.assertAlmostEqual(rec.tts_ttfb_ms or 0.0, 190.0, places=1)
            self.assertAlmostEqual(rec.tts_total_processing_ms or 0.0, 410.0, places=1)
            self.assertEqual(rec.tts_characters, 65)

            # Waterfall stages must sum to total E2E latency
            stage_sum = sum(s.duration_ms for s in rec.waterfall)
            self.assertAlmostEqual(stage_sum, rec.e2e_from_user_silence_ms, places=1)
            self.assertAlmostEqual(
                rec.e2e_from_vad_stop_ms,
                rec.e2e_from_user_silence_ms - rec.endpointing_wait_ms,
                places=1,
            )

            # Reload from JSONL and verify persistence
            reloaded = LatencyStore(jsonl_path=jsonl_path, load_existing=True)
            summary = reloaded.summary()
            self.assertEqual(summary.user_turn_count, 1)
            self.assertEqual(summary.greeting_count, 1)
            report = format_turn_report(rec, summary)
            self.assertIn("CASCADE LATENCY REPORT", report)
            self.assertIn("Why does ice float?", report)

    def test_observatory_http_routes(self) -> None:
        test_app = FastAPI()
        store = LatencyStore(jsonl_path=None, load_existing=False)
        register_observatory_routes(test_app, store)

        with TestClient(test_app) as client:
            r_root = client.get("/", follow_redirects=False)
            self.assertIn(r_root.status_code, (302, 307))
            self.assertEqual(r_root.headers["location"], "/dashboard")

            r_dash = client.get("/dashboard")
            self.assertEqual(r_dash.status_code, 200)
            self.assertIn("Pipecat Gemini Cascade", r_dash.text)

            r_cfg = client.get("/api/config")
            self.assertEqual(r_cfg.status_code, 200)
            cfg_json = r_cfg.json()
            self.assertEqual(cfg_json["stt"]["model"], "gemini-3.5-transcribe-live")
            self.assertEqual(cfg_json["llm"]["model"], "gemini-3.5-flash-lite")
            self.assertEqual(cfg_json["tts"]["model"], "gemini-3.8-flash-lite-tts")

            r_lat = client.get("/api/latency")
            self.assertEqual(r_lat.status_code, 200)
            self.assertEqual(r_lat.json()["summary"]["user_turn_count"], 0)

    def test_start_ws_url_middleware_rewrites_0000_and_forwarded_headers(self) -> None:
        test_app = FastAPI()

        @test_app.post("/start")
        async def fake_runner_start() -> dict[str, object]:
            # Simulates pipecat.runner.run returning wss://0.0.0.0:7860/ws-client when --host 0.0.0.0
            return {
                "wsUrl": "wss://0.0.0.0:7860/ws-client",
                "sessionId": "sess-ws-1",
                "token": None,
            }

        store = LatencyStore(jsonl_path=None, load_existing=False)
        register_observatory_routes(test_app, store)

        with TestClient(test_app, base_url="http://localhost:7860") as client:
            res_local = client.post("/start", json={"transport": "websocket"})
            self.assertEqual(res_local.status_code, 200)
            self.assertEqual(res_local.json()["wsUrl"], "ws://localhost:7860/ws-client")

            res_proxy = client.post(
                "/start",
                json={"transport": "websocket"},
                headers={
                    "X-Forwarded-Proto": "https",
                    "X-Forwarded-Host": "voice-demo.example.com",
                },
            )
            self.assertEqual(res_proxy.status_code, 200)
            self.assertEqual(
                res_proxy.json()["wsUrl"],
                "wss://voice-demo.example.com/ws-client",
            )

    async def test_websocket_params_and_browser_protobuf_wire_compatibility(self) -> None:
        import json

        from pipecat.frames.frames import (
            InputAudioRawFrame,
            InputTransportMessageFrame,
            InterruptionFrame,
            OutputAudioRawFrame,
            OutputTransportMessageUrgentFrame,
        )
        from pipecat.serializers.protobuf import ProtobufFrameSerializer

        cfg = CascadeConfig(api_key="unit-test-dummy-key")
        ws_params = build_websocket_params(cfg)
        self.assertTrue(ws_params.audio_in_enabled)
        self.assertTrue(ws_params.audio_out_enabled)
        self.assertEqual(ws_params.audio_in_sample_rate, 16000)
        self.assertEqual(ws_params.audio_out_sample_rate, 24000)
        self.assertFalse(ws_params.add_wav_header)
        self.assertIsInstance(ws_params.serializer, ProtobufFrameSerializer)
        self.assertIsInstance(ws_params.allowed_origins, list)

        serializer = ws_params.serializer
        assert serializer is not None

        def _varint(n: int) -> bytes:
            out = bytearray()
            while n >= 0x80:
                out.append((n & 0x7F) | 0x80)
                n >>= 7
            out.append(n)
            return bytes(out)

        # 1) Encode 20ms 16kHz mono PCM frame using the exact browser JS layout
        pcm_20ms = b"\x01\x00\xff\xff" * 160  # 640 bytes (320 int16 samples)
        inner_audio = (
            b"\x1a" + _varint(len(pcm_20ms)) + pcm_20ms + b"\x20" + _varint(16000) + b"\x28\x01"
        )
        wire_audio = b"\x12" + _varint(len(inner_audio)) + inner_audio

        deserialized_audio = await serializer.deserialize(wire_audio)
        self.assertIsInstance(deserialized_audio, InputAudioRawFrame)
        assert isinstance(deserialized_audio, InputAudioRawFrame)
        self.assertEqual(deserialized_audio.audio, pcm_20ms)
        self.assertEqual(deserialized_audio.sample_rate, 16000)
        self.assertEqual(deserialized_audio.num_channels, 1)

        # 2) Encode RTVI client-ready message using the exact browser JS layout
        ready_json = json.dumps(
            {
                "label": "rtvi-ai",
                "type": "client-ready",
                "id": "ready-01",
                "data": {
                    "version": "2.2.0",
                    "about": {"library": "pipecat-cascade-observatory-ws"},
                },
            }
        ).encode("utf-8")
        inner_msg = b"\x0a" + _varint(len(ready_json)) + ready_json
        wire_msg = b"\x22" + _varint(len(inner_msg)) + inner_msg

        deserialized_msg = await serializer.deserialize(wire_msg)
        self.assertIsInstance(deserialized_msg, InputTransportMessageFrame)
        assert isinstance(deserialized_msg, InputTransportMessageFrame)
        self.assertEqual(deserialized_msg.message["type"], "client-ready")
        self.assertEqual(deserialized_msg.message["data"]["version"], "2.2.0")

        # 3) Verify server -> browser serialization for audio, RTVI messages, and interruptions
        out_audio_bytes = await serializer.serialize(
            OutputAudioRawFrame(audio=b"\x02\x00" * 240, sample_rate=24000, num_channels=1)
        )
        self.assertIsInstance(out_audio_bytes, bytes)

        out_rtvi_bytes = await serializer.serialize(
            OutputTransportMessageUrgentFrame(
                message={"label": "rtvi-ai", "type": "bot-ready", "id": "ready-01", "data": {}}
            )
        )
        self.assertIsInstance(out_rtvi_bytes, bytes)

        out_intr_bytes = await serializer.serialize(InterruptionFrame())
        self.assertIsInstance(out_intr_bytes, bytes)


if __name__ == "__main__":
    unittest.main()
