# Pipecat Gemini STT → LLM → TTS Cascade Voice Bot (with 8kHz Telephony Dialer Converter)

A real-time streaming cascade voice bot built on **Pipecat (`pipecat-ai >= 1.13`)** and **Pipecat Client JS (`@pipecat-ai/client-js`)** to evaluate end-to-end voice latency and per-block latency across three Gemini models, with built-in **8kHz Telephony Dialer Conversion** (`teledemo` branch):

1. **8kHz Telephony Input Converter (`Telephony8kInputConverter`)**: Accepts **8,000 Hz** telephone audio (`G.711 μ-law`, `G.711 A-law`, or `16-bit Linear PCM`) from PSTN/SIP/Cloud dialers and upsamples it in real time via `SOXRStreamAudioResampler` to **16,000 Hz PCM16LE**.
2. **STT Block (`GeminiSTTService`)**: `gemini-3.5-transcribe-live` over Gemini Live API WebSockets (**16kHz PCM** in) paired with local **Silero VAD** Hybrid endpointing (`audio_stream_end=True`) and **SmartTurn v3 ONNX**.
3. **LLM Block (`GoogleLLMService`)**: `gemini-3.5-flash-lite` with `thinking_level="minimal"` for low-latency token streaming.
4. **TTS Block (`GeminiTTSService`)**: `gemini-3.8-flash-lite-tts` via the `google-genai` Developer API (`use_genai=True`, **24kHz mono PCM** native output).
5. **8kHz Telephony Output Converter (`Telephony8kOutputConverter`)**: Downsamples **24,000 Hz** Gemini TTS audio frames to **8,000 Hz** (flushing the SOXR filter tail on `TTSStoppedFrame`) and encodes to `G.711 μ-law`, `G.711 A-law`, or `8kHz PCM16LE` for the dialer output transport.

---

## Architecture & Sample-Rate Conversion Flow (`teledemo`)

```text
8kHz Dialer (Twilio / Exotel / Plivo / Telnyx / SIP /ws-dialer: μ-law, A-law, or PCM16)
        │
        ▼ (8,000 Hz)
[InputTransport / DialerSerializer] ──▶ [Telephony8kInputConverter: 8kHz -> 16kHz SOXR]
                                                        │
                                                        ▼ (16,000 Hz PCM16LE)
                                        [GeminiSTTService: gemini-3.5-transcribe-live]
                                                        │  ▲
                                    16kHz audio + text  │  │ VADUserStoppedSpeakingFrame
                                                        ▼  │
                                        [LLMUserAggregator] (Silero VAD + SmartTurn v3 ONNX @ 16kHz)
                                                        │
                                                        ▼
                                        [GoogleLLMService: gemini-3.5-flash-lite]
                                                        │
                                                        ▼ (token stream -> sentence aggregator)
                                        [GeminiTTSService: gemini-3.8-flash-lite-tts (24,000 Hz)]
                                                        │
                                                        ▼ (24kHz -> 8kHz SOXR + tail flush)
                                        [Telephony8kOutputConverter: 24kHz -> 8kHz]
                                                        │
                                                        ▼ (8,000 Hz PCM16 / μ-law / A-law)
8kHz Dialer / Speaker / RTVI ◀───────── [OutputTransport] ──▶ [LLMAssistantAggregator]
```

### Supported 8kHz Telephony Dialers & Endpoints

| Dialer / Protocol | WebSocket Endpoint | Wire Format & Codec | Notes |
| :--- | :--- | :--- | :--- |
| **Twilio Media Streams** | `ws://<host>:7860/ws` (TwiML: `POST /twiml`) | JSON base64 · `8kHz G.711 μ-law (PCMU)` | Auto-detected by `parse_telephony_websocket` |
| **Exotel Voicebot Applet** | `ws://<host>:7860/ws` | JSON base64 · `8kHz 16-bit Linear PCM` | Auto-detected by `parse_telephony_websocket` |
| **Plivo AudioStream** | `ws://<host>:7860/ws` | JSON base64 · `8kHz G.711 μ-law (PCMU)` | Auto-detected by `parse_telephony_websocket` |
| **Telnyx Media Streaming** | `ws://<host>:7860/ws` | JSON base64 · `8kHz PCMU / PCMA` | Auto-detected by `parse_telephony_websocket` |
| **Universal 8kHz Dialer** | `ws://<host>:7860/ws-dialer?codec=ulaw&rate=8000` | JSON base64 **or** Raw Binary (`ulaw`, `alaw`, `pcm16`) | Works with Asterisk AudioSocket, FreeSWITCH, Ozonetel, Genesys, custom PBXs |

---

## Quick Start & Testing Workflow

### 1. Add your Gemini API key
Create or edit `.env` in the repository root and add your Gemini Developer API key:

```ini
GOOGLE_API_KEY=AIza...
TELEPHONY_8K_ENABLED=true
DIALER_SAMPLE_RATE=8000
DIALER_ENCODING=ulaw
```

### 2. Start the Server
```bash
uv run bot.py
```
*(Binds to `0.0.0.0:7860` by default; pass `--port <PORT>` or `--host <HOST>` to override.)*

### 3. Test 8kHz Telephony Mode in the Dashboard (`/dashboard`)
Open `http://localhost:7860/dashboard`:
1. **Live 8kHz Mic Simulation**: Select **`WebSocket 8kHz Telephony Sim (8kHz → 16kHz)`** in the transport dropdown and click **Connect & Talk** to stream 8kHz mic frames through `Telephony8kInputConverter`.
2. **No-Mic 8kHz Dialer Simulator**: Select **`8kHz Dialer · G.711 μ-law`**, **`G.711 A-law`**, or **`Linear PCM16`** in the Spoken Turn Simulator and click **Run Turn** to synthesize an 8kHz telephone caller turn and play back the 8kHz converted bot response WAV.

### 4. Run the Headless 8kHz Telephony Benchmark CLI
```bash
uv run benchmark_cascade.py --telephony-8k --dialer-encoding ulaw --turns 3 --save-wav-dir logs/wavs
```

Or compare against wideband (`16kHz in / 24kHz out`):
```bash
uv run benchmark_cascade.py --wideband --turns 3
```

### 5. Run Unit & Telephony Converter Tests
```bash
uv run python -m unittest discover -s tests -v
```
