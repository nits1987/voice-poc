# Pipecat Gemini STT → LLM → TTS Cascade Voice Bot & Latency Observatory

A real-time streaming cascade voice bot built on **Pipecat (`pipecat-ai >= 1.13`)** and **Pipecat Client JS (`@pipecat-ai/client-js`)** to evaluate end-to-end voice latency as well as per-block latency across three Gemini models:

1. **STT Block (`GeminiSTTService`)**: `gemini-3.5-transcribe-live` over Gemini Live API WebSockets (16kHz PCM in) paired with local **Silero VAD** Hybrid endpointing (`audio_stream_end=True`).
2. **LLM Block (`GoogleLLMService`)**: `gemini-3.5-flash-lite` with `thinking_level="minimal"` for low-latency token streaming.
3. **TTS Block (`GeminiTTSService`)**: `gemini-3.8-flash-lite-tts` via the `google-genai` Developer API (`use_genai=True`, 24kHz mono PCM out).

---

## Architecture & Latency Instrumentation

```text
Mic / 16kHz PCM ─▶ [InputTransport] ─▶ [GeminiSTTService: gemini-3.5-transcribe-live]
                                                │  ▲
                            16kHz audio + text  │  │ VADUserStoppedSpeakingFrame (audio_stream_end)
                                                ▼  │
                                       [LLMUserAggregator] (Silero VAD + SmartTurn v3 ONNX)
                                                │
                                                ▼
                                       [GoogleLLMService: gemini-3.5-flash-lite]
                                                │
                                                ▼ (token stream -> sentence aggregator)
                                       [GeminiTTSService: gemini-3.8-flash-lite-tts]
                                                │
                                                ▼ (24kHz audio chunks)
                     Speaker / RTVI ◀─ [OutputTransport] ─▶ [LLMAssistantAggregator]
```

### What Gets Measured on Every Turn
`CascadeLatencyMonitor` combines Pipecat's `UserBotLatencyObserver` (critical-path timeline from the exact instant the user stops speaking to the instant `BotStartedSpeakingFrame` leaves the output transport) with per-service `MetricsFrame` telemetry:

| Metric | Critical-Path Waterfall Key | Per-Service Metric | Definition |
| :--- | :--- | :--- | :--- |
| **Overall E2E Perceived** | Sum of all stages | `e2e_from_user_silence_ms` | Actual end of user speech (`VAD stop - vad_stop_secs`) to first bot audio frame leaving output transport |
| **Net Cascade (ex-VAD)** | E2E minus `endpointing_wait` | `e2e_from_vad_stop_ms` | Latency strictly after Silero VAD confirms silence |
| **0. VAD Endpointing Wait** | `endpointing_wait` | `vad_stop_secs` | Silence window required by Silero VAD (`0.20s` default) |
| **1. Live STT** | `transcription` (`stt_finalize_ms`) | `stt_ttfb_ms`, `stt_utterance_processing_ms` | Time after user silence until `gemini-3.5-transcribe-live` emits finalized `TranscriptionFrame` |
| **2. Turn Detection** | `turn_detection` | `smart_turn_e2e_ms`, `smart_turn_probability` | Local SmartTurn v3 ONNX end-of-turn classification time after transcript arrives |
| **3. LLM Inference** | `llm_inference` | `llm_ttfb_ms`, `llm_ttfat_ms`, `llm_total_processing_ms` | `gemini-3.5-flash-lite` time-to-first-token, time-to-first-actionable-text, and full stream duration |
| **4. Sentence Aggregation** | `sentence_aggregation` | `sentence_aggregation_ms` | Time buffering streamed LLM tokens until the first speakable sentence boundary |
| **5. TTS Synthesis** | `speech_synthesis` | `tts_ttfb_ms`, `tts_ttfa_ms`, `tts_total_processing_ms` | `gemini-3.8-flash-lite-tts` time-to-first-byte, time-to-first-audible-audio, and full stream synthesis |
| **6. Output & Pipeline** | `output_transport`, `pipeline` | `output_transport_ms`, `pipeline_overhead_ms` | Transport audio buffer framing + async frame-queue handoff overhead |

All completed turns are persisted to `logs/latency_metrics.jsonl` (with initial greetings isolated from `user_turn` P50/P95/Mean aggregates).

---

## Quick Start

### 1. Add your Gemini API key
Edit `.env` and paste your Gemini Developer API key:

```ini
GOOGLE_API_KEY=AIza...
```

### 2. Start the Interactive Voice Bot + Latency Observatory UI
```bash
cd /usr/local/google/home/nitindhir/Documents/paytm-demo
uv run bot.py --host 0.0.0.0 --port 7860
```

Open:
- **Latency Observatory Dashboard (Pipecat JS Client + Waterfall UI + No-Mic Spoken Turn Simulator)**: `http://localhost:7860/dashboard`
- **Official Pipecat Prebuilt WebRTC UI**: `http://localhost:7860/client/`

> **Cloudtop Microphone Tip**: Browser `getUserMedia()` requires HTTPS or `localhost`. When running on Cloudtop (`nd-dev01.c.googlers.com`), forward port `7860` from your laptop:
> ```bash
> ssh -L 7860:localhost:7860 nd-dev01.c.googlers.com
> ```
> Then open `http://localhost:7860/dashboard` in Chrome on your laptop for live WebRTC microphone & speaker audio. You can also test full spoken cascade turns directly on `http://nd-dev01.c.googlers.com:7860/dashboard` without a microphone using the **Spoken Turn Simulator** panel.

### 3. Run the Headless Spoken Audio Benchmark CLI
Synthesizes real 16kHz spoken questions, streams them in real-time 20ms frames through the full cascade (`Silero VAD -> Live STT -> SmartTurn v3 -> Flash-Lite LLM -> Flash-Lite TTS`), and prints per-turn waterfalls plus a P50/P95/Mean summary table:

```bash
uv run benchmark_cascade.py --turns 3 --save-wav-dir logs/wavs
```

Run custom spoken questions or compare SmartTurn v3 vs pure VAD silence endpointing:

```bash
uv run benchmark_cascade.py \
  -q "How do geostationary satellites stay above the same spot on Earth?" \
  -q "Why do leaves change color in autumn?" \
  --turn-stop-mode vad \
  --vad-stop-secs 0.2
```

### 4. Run Unit & Route Verification Tests
```bash
uv run python -m unittest discover -s tests -v
```
