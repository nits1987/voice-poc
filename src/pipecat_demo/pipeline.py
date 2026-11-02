"""Pipecat STT -> LLM -> TTS cascade pipeline builder and worker runner."""

from __future__ import annotations

import asyncio
import uuid

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.google.gemini_live.stt import GeminiSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.google.tts import GeminiTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from pipecat_demo.config import CascadeConfig
from pipecat_demo.latency import CascadeLatencyMonitor, LatencyStore

GREETING_PROMPT = (
    "Greet the user warmly in one short sentence and invite them to ask any general question."
)


def build_transport_params(cfg: CascadeConfig) -> TransportParams:
    """Build audio-only TransportParams matching the Gemini STT (16kHz) and TTS (24kHz) rates."""
    return TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=cfg.audio_in_sample_rate,
        audio_out_enabled=True,
        audio_out_sample_rate=cfg.audio_out_sample_rate,
    )


def build_websocket_params(cfg: CascadeConfig) -> FastAPIWebsocketParams:
    """Build FastAPI WebSocket transport params with Protobuf RTVI + raw PCM framing."""
    return FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_in_sample_rate=cfg.audio_in_sample_rate,
        audio_out_enabled=True,
        audio_out_sample_rate=cfg.audio_out_sample_rate,
        add_wav_header=False,
        serializer=ProtobufFrameSerializer(),
    )


def build_user_aggregator_params(cfg: CascadeConfig) -> LLMUserAggregatorParams:
    """Configure Silero VAD + SmartTurn v3 (or pure VAD timeout) turn endpointing."""
    vad_analyzer = SileroVADAnalyzer(params=VADParams(stop_secs=cfg.vad_stop_secs))
    if cfg.turn_stop_mode == "vad":
        strategies = UserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.05)]
        )
    else:
        # Default UserTurnStrategies uses bundled LocalSmartTurnAnalyzerV3 ONNX model
        # gated on finalized transcription frames from GeminiSTTService.
        strategies = UserTurnStrategies()

    return LLMUserAggregatorParams(
        vad_analyzer=vad_analyzer,
        user_turn_strategies=strategies,
    )


def create_cascade_services(
    cfg: CascadeConfig,
) -> tuple[GeminiSTTService, GoogleLLMService, GeminiTTSService]:
    """Instantiate the three Gemini cascade blocks without deprecated parameters."""
    api_key = cfg.require_api_key()

    # Block 1: Streaming Live Transcription over WebSocket with Hybrid Silero VAD finalization
    stt = GeminiSTTService(
        api_key=api_key,
        settings=GeminiSTTService.Settings(
            model=cfg.stt_model,
            mode=cfg.stt_mode,
        ),
    )

    # Block 2: Low-latency streaming text LLM (minimal thinking, default sampling params)
    llm = GoogleLLMService(
        api_key=api_key,
        settings=GoogleLLMService.Settings(
            model=cfg.llm_model,
            system_instruction=cfg.llm_system_instruction,
            thinking=GoogleLLMService.ThinkingConfig(thinking_level=cfg.llm_thinking_level),
        ),
    )

    # Block 3: Streaming speech synthesis via Gemini Developer API (24kHz mono PCM)
    tts = GeminiTTSService(
        api_key=api_key,
        use_genai=True,
        settings=GeminiTTSService.Settings(
            model=cfg.tts_model,
            voice=cfg.tts_voice,
            language=Language.EN_US,
            prompt=cfg.tts_style_prompt,
        ),
    )

    return stt, llm, tts


def create_cascade_session(
    transport: BaseTransport,
    cfg: CascadeConfig,
    *,
    store: LatencyStore,
    session_id: str | None = None,
    greet_on_connect: bool | None = None,
    emit_rtvi_messages: bool = True,
    log_reports: bool = True,
    cancel_on_idle_timeout: bool = True,
) -> tuple[PipelineWorker, CascadeLatencyMonitor]:
    """Assemble the instrumented Pipecat cascade pipeline and attach latency observers."""
    resolved_session_id = session_id or uuid.uuid4().hex[:12]
    should_greet = cfg.greet_on_connect if greet_on_connect is None else greet_on_connect

    stt, llm, tts = create_cascade_services(cfg)
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=build_user_aggregator_params(cfg),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    monitor = CascadeLatencyMonitor(
        session_id=resolved_session_id,
        store=store,
        stt_model=cfg.stt_model,
        llm_model=cfg.llm_model,
        tts_model=cfg.tts_model,
        emit_rtvi_messages=emit_rtvi_messages,
        log_reports=log_reports,
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=cfg.audio_in_sample_rate,
            audio_out_sample_rate=cfg.audio_out_sample_rate,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=monitor.observers(),
        cancel_on_idle_timeout=cancel_on_idle_timeout,
    )
    monitor.attach_worker(worker)

    greeted = False
    disconnected = False
    greeting_fallback_task: asyncio.Task[None] | None = None

    async def _trigger_greeting_once() -> None:
        nonlocal greeted
        if not should_greet or greeted or disconnected:
            return
        greeted = True
        await worker.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": GREETING_PROMPT}],
                run_llm=True,
            )
        )

    if worker.rtvi is not None:

        @worker.rtvi.event_handler("on_client_ready")
        async def _on_client_ready(_rtvi: object) -> None:
            await _trigger_greeting_once()

    @transport.event_handler("on_client_connected")
    async def _on_client_connected(_transport: BaseTransport, _client: object) -> None:
        nonlocal greeting_fallback_task
        logger.info(
            f"[{resolved_session_id}] Client connected "
            f"(STT={cfg.stt_model} -> LLM={cfg.llm_model} -> TTS={cfg.tts_model})"
        )
        if should_greet and not greeted:
            if worker.rtvi is None:
                await _trigger_greeting_once()
            else:

                async def _delayed_greeting() -> None:
                    await asyncio.sleep(0.75)
                    await _trigger_greeting_once()

                greeting_fallback_task = asyncio.create_task(_delayed_greeting())

    @transport.event_handler("on_client_disconnected")
    async def _on_client_disconnected(_transport: BaseTransport, _client: object) -> None:
        nonlocal disconnected
        disconnected = True
        if greeting_fallback_task is not None and not greeting_fallback_task.done():
            greeting_fallback_task.cancel()
        logger.info(f"[{resolved_session_id}] Client disconnected; flushing latency metrics.")
        await monitor.flush_open_turn()
        await worker.cancel()

    @worker.event_handler("on_pipeline_finished")
    async def _on_pipeline_finished(_worker: PipelineWorker, _frame: object) -> None:
        await monitor.flush_open_turn()

    return worker, monitor


async def run_cascade_worker(worker: PipelineWorker, *, handle_sigint: bool = False) -> None:
    """Run a single cascade PipelineWorker to completion on a dedicated WorkerRunner."""
    runner = WorkerRunner(handle_sigint=handle_sigint)
    await runner.add_workers(worker)
    await runner.run()
