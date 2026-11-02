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

### Observed Benchmark Result: `< 1s` End-to-End Bot Speech Start
In live interactive voice sessions, **overall end-to-end latency from user silence to bot speech start (`e2e_from_user_silence_ms`) consistently lands under 1 second (`< 1,000 ms`)**. Three streaming design choices keep the cascade sub-second:
- **Continuous Live STT WebSocket**: Audio streams in 20ms 16kHz frames to `gemini-3.5-transcribe-live` *while* the user is speaking, so once local Silero VAD (`0.20s`) + SmartTurn v3 ONNX detect turn completion, finalized transcription arrives almost immediately.
- **Minimal-Thinking Token Streaming**: `gemini-3.5-flash-lite` runs with `thinking_level="minimal"` and streams tokens into the sentence aggregator as soon as the first clause forms—without waiting for full LLM completion.
- **Streaming Audio Chunks**: `gemini-3.8-flash-lite-tts` synthesizes and emits 24kHz PCM audio chunks incrementally, starting playback as soon as the first audio chunk arrives.

---

## Quick Start & Testing Workflow

### 1. Add your Gemini API key
Create or edit `.env` in the repository root and add your Gemini Developer API key:

```ini
GOOGLE_API_KEY=AIza...
```

### 2. Start the Server
```bash
uv run bot.py
```
*(Binds to `0.0.0.0:7860` by default; pass `--port <PORT>` or `--host <HOST>` to override.)*

### 3. Test the Voice Flow (`/client/`) & Inspect Metrics (`/dashboard`)
Use the two web routes side-by-side when testing:

1. **Test the Live Voice Flow — `/client/` (`http://localhost:7860/client/`)**:
   - Opens the **Pipecat Prebuilt Client UI** to run and test live multi-turn voice conversations with the bot (microphone input, live transcripts, audio playback, and transport selection—use **WebSocket** when over an SSH tunnel/Cloudtop, or **WebRTC** on direct localhost).
2. **Check Latency Metrics & Waterfalls — `/dashboard` (`http://localhost:7860/dashboard`)**:
   - Opens the **Latency Observatory Dashboard** to inspect **P50 / P95 KPI cards**, **per-turn critical-path waterfalls**, and **per-block STT / LLM / TTS latency breakdowns** (`TTFB`, `TTFAT`, `TTFA`, token counts, and JSON export).
   - Every turn spoken in `/client/` is recorded by `CascadeLatencyMonitor` into the shared `GLOBAL_LATENCY_STORE` and automatically appears on `/dashboard` in real time (you can also test directly from `/dashboard` via **Connect & Talk** or the no-mic **Spoken Turn Simulator**).

> **Cloudtop / Remote Host Tip**: Browser microphone capture (`navigator.mediaDevices.getUserMedia`) requires HTTPS or `localhost`. When running the server on a remote dev box / Cloudtop:
> ```bash
> ssh -N -L 7860:localhost:7860 <your-cloudtop-host>
> ```
> Then open `http://localhost:7860/client/` to talk to the bot and `http://localhost:7860/dashboard` to monitor per-turn and aggregate latency metrics (or allowlist `http://<your-cloudtop-host>:7860` under `chrome://flags/#unsafely-treat-insecure-origin-as-secure`).

### 4. Run the Headless Spoken Audio Benchmark CLI
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

### 5. Run Unit & Route Verification Tests
```bash
uv run python -m unittest discover -s tests -v
```
