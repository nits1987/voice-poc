"""Per-block and end-to-end latency instrumentation for the Pipecat cascade."""

from __future__ import annotations

import asyncio
import math
import statistics
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextAssistantTurnFrame,
    LLMContextUserTurnFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    STTUsageMetricsData,
    TextAggregationMetricsData,
    TTFAMetricsData,
    TTFATMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
    TurnMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.user_bot_latency_observer import (
    LatencyBreakdown,
    MeasuredFrom,
    UserBotLatencyObserver,
)
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pydantic import BaseModel, Field


def _ms(seconds: float | None) -> float | None:
    if seconds is None:
        return None
    return round(seconds * 1000.0, 2)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 2)
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return round(sorted_vals[low], 2)
    weight = rank - low
    return round(sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight, 2)


class WaterfallStage(BaseModel):
    """Non-overlapping stage on the critical user-silence -> bot-speaking timeline."""

    key: str
    label: str
    owner: str
    owner_kind: str
    offset_ms: float
    duration_ms: float
    share_pct: float


class TurnLatencyRecord(BaseModel):
    """Complete per-block and end-to-end latency record for one turn."""

    turn_id: str
    session_id: str
    turn_index: int
    turn_type: Literal["user_turn", "greeting"] = "user_turn"
    timestamp_utc: str
    completed: bool = False
    interrupted: bool = False

    user_transcript: str = ""
    assistant_transcript: str = ""

    # End-to-end latency metrics (ms)
    e2e_from_user_silence_ms: float = 0.0
    e2e_from_vad_stop_ms: float = 0.0
    user_turn_release_ms: float | None = None
    bot_speech_duration_ms: float | None = None

    # Critical-path waterfall contributions (ms) — sum to e2e_from_user_silence_ms
    endpointing_wait_ms: float = 0.0
    stt_finalize_ms: float = 0.0
    turn_detection_ms: float = 0.0
    llm_inference_span_ms: float = 0.0
    sentence_aggregation_ms: float = 0.0
    tts_synthesis_span_ms: float = 0.0
    output_transport_ms: float = 0.0
    pipeline_overhead_ms: float = 0.0

    # Per-service reported block metrics (ms)
    stt_model: str | None = None
    stt_ttfb_ms: float | None = None
    stt_utterance_processing_ms: float | None = None
    stt_audio_seconds: float | None = None

    smart_turn_e2e_ms: float | None = None
    smart_turn_probability: float | None = None

    llm_model: str | None = None
    llm_ttfb_ms: float | None = None
    llm_ttfat_ms: float | None = None
    llm_thinking_ms: float | None = None
    llm_total_processing_ms: float | None = None
    llm_prompt_tokens: int | None = None
    llm_completion_tokens: int | None = None
    llm_reasoning_tokens: int | None = None

    tts_model: str | None = None
    tts_ttfb_ms: float | None = None
    tts_ttfa_ms: float | None = None
    tts_leading_silence_ms: float | None = None
    tts_total_processing_ms: float | None = None
    tts_characters: int | None = None

    waterfall: list[WaterfallStage] = Field(default_factory=list)


class MetricStats(BaseModel):
    """Aggregate summary statistics in milliseconds across user turns."""

    count: int = 0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    @classmethod
    def from_values(cls, values: list[float | None]) -> MetricStats:
        clean = sorted(v for v in values if v is not None and v >= 0)
        if not clean:
            return cls()
        return cls(
            count=len(clean),
            mean_ms=round(statistics.fmean(clean), 2),
            p50_ms=_percentile(clean, 50.0),
            p95_ms=_percentile(clean, 95.0),
            min_ms=round(clean[0], 2),
            max_ms=round(clean[-1], 2),
        )


class LatencySummary(BaseModel):
    """Aggregated P50/P95/Mean statistics across all recorded user turns."""

    user_turn_count: int = 0
    greeting_count: int = 0
    e2e_from_user_silence: MetricStats = Field(default_factory=MetricStats)
    e2e_from_vad_stop: MetricStats = Field(default_factory=MetricStats)
    endpointing_wait: MetricStats = Field(default_factory=MetricStats)
    stt_finalize: MetricStats = Field(default_factory=MetricStats)
    stt_ttfb: MetricStats = Field(default_factory=MetricStats)
    turn_detection: MetricStats = Field(default_factory=MetricStats)
    llm_ttfb: MetricStats = Field(default_factory=MetricStats)
    llm_ttfat: MetricStats = Field(default_factory=MetricStats)
    llm_total_processing: MetricStats = Field(default_factory=MetricStats)
    sentence_aggregation: MetricStats = Field(default_factory=MetricStats)
    tts_ttfb: MetricStats = Field(default_factory=MetricStats)
    tts_ttfa: MetricStats = Field(default_factory=MetricStats)
    tts_total_processing: MetricStats = Field(default_factory=MetricStats)
    output_transport: MetricStats = Field(default_factory=MetricStats)
    pipeline_overhead: MetricStats = Field(default_factory=MetricStats)


class LatencyStore:
    """Thread-safe store of turn latency records backed by an append-only JSONL log."""

    def __init__(self, jsonl_path: Path | None = None, *, load_existing: bool = True):
        self._lock = threading.Lock()
        self._turns: list[TurnLatencyRecord] = []
        self._by_id: dict[str, int] = {}
        self._jsonl_path = jsonl_path
        if jsonl_path is not None and load_existing and jsonl_path.exists():
            self._load_existing(jsonl_path)

    def _load_existing(self, path: Path) -> None:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                record = TurnLatencyRecord.model_validate_json(line)
                if record.turn_id in self._by_id:
                    idx = self._by_id[record.turn_id]
                    self._turns[idx] = record
                else:
                    self._by_id[record.turn_id] = len(self._turns)
                    self._turns.append(record)
        except Exception as exc:
            logger.warning(f"Could not load existing latency history from {path}: {exc}")

    def upsert(self, record: TurnLatencyRecord, *, persist: bool = False) -> LatencySummary:
        """Insert or update a turn record and optionally append to JSONL."""
        with self._lock:
            if record.turn_id in self._by_id:
                idx = self._by_id[record.turn_id]
                self._turns[idx] = record
            else:
                self._by_id[record.turn_id] = len(self._turns)
                self._turns.append(record)

            if persist and self._jsonl_path is not None:
                self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with self._jsonl_path.open("a", encoding="utf-8") as fh:
                    fh.write(record.model_dump_json() + "\n")

            return self._compute_summary_unlocked()

    def reset(self, *, clear_jsonl: bool = False) -> None:
        """Clear in-memory records (and optionally truncate the JSONL log)."""
        with self._lock:
            self._turns.clear()
            self._by_id.clear()
            if clear_jsonl and self._jsonl_path is not None and self._jsonl_path.exists():
                self._jsonl_path.write_text("", encoding="utf-8")

    def turns(self, *, session_id: str | None = None) -> list[TurnLatencyRecord]:
        with self._lock:
            if session_id is None:
                return list(self._turns)
            return [t for t in self._turns if t.session_id == session_id]

    def summary(self, *, session_id: str | None = None) -> LatencySummary:
        with self._lock:
            return self._compute_summary_unlocked(session_id=session_id)

    def snapshot(self, *, session_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            items = (
                list(self._turns)
                if session_id is None
                else [t for t in self._turns if t.session_id == session_id]
            )
            summary = self._compute_summary_unlocked(session_id=session_id)
            return {
                "summary": summary.model_dump(),
                "turns": [t.model_dump() for t in reversed(items)],
            }

    def _compute_summary_unlocked(self, *, session_id: str | None = None) -> LatencySummary:
        scoped = (
            self._turns
            if session_id is None
            else [t for t in self._turns if t.session_id == session_id]
        )
        user_turns = [t for t in scoped if t.turn_type == "user_turn"]
        greetings = [t for t in scoped if t.turn_type == "greeting"]
        return LatencySummary(
            user_turn_count=len(user_turns),
            greeting_count=len(greetings),
            e2e_from_user_silence=MetricStats.from_values(
                [t.e2e_from_user_silence_ms for t in user_turns]
            ),
            e2e_from_vad_stop=MetricStats.from_values([t.e2e_from_vad_stop_ms for t in user_turns]),
            endpointing_wait=MetricStats.from_values([t.endpointing_wait_ms for t in user_turns]),
            stt_finalize=MetricStats.from_values([t.stt_finalize_ms for t in user_turns]),
            stt_ttfb=MetricStats.from_values([t.stt_ttfb_ms for t in user_turns]),
            turn_detection=MetricStats.from_values([t.turn_detection_ms for t in user_turns]),
            llm_ttfb=MetricStats.from_values([t.llm_ttfb_ms for t in user_turns]),
            llm_ttfat=MetricStats.from_values([t.llm_ttfat_ms for t in user_turns]),
            llm_total_processing=MetricStats.from_values(
                [t.llm_total_processing_ms for t in user_turns]
            ),
            sentence_aggregation=MetricStats.from_values(
                [t.sentence_aggregation_ms for t in user_turns]
            ),
            tts_ttfb=MetricStats.from_values([t.tts_ttfb_ms for t in user_turns]),
            tts_ttfa=MetricStats.from_values([t.tts_ttfa_ms for t in user_turns]),
            tts_total_processing=MetricStats.from_values(
                [t.tts_total_processing_ms for t in user_turns]
            ),
            output_transport=MetricStats.from_values([t.output_transport_ms for t in user_turns]),
            pipeline_overhead=MetricStats.from_values([t.pipeline_overhead_ms for t in user_turns]),
        )


def format_turn_report(record: TurnLatencyRecord, summary: LatencySummary) -> str:
    """Format a readable terminal table for a single turn and running session percentiles."""
    title_kind = "USER TURN" if record.turn_type == "user_turn" else "INITIAL GREETING"
    lines = [
        "",
        "╭──────────────────────────────────────────────────────────────────────────────╮",
        f"│ CASCADE LATENCY REPORT · Turn #{record.turn_index:<3} ({title_kind}) · Session {record.session_id[:8]} │",
        "├──────────────────────────────────────────────────────────────────────────────┤",
    ]
    if record.user_transcript:
        lines.append(f"│ User : {record.user_transcript[:69]:<69} │")
    if record.assistant_transcript:
        lines.append(f"│ Bot  : {record.assistant_transcript[:69]:<69} │")
    lines.extend(
        [
            "├──────────────────────────────┬────────────┬──────────┬───────────────────────┤",
            "│ Critical-Path Stage          │ Latency ms │ Share %  │ Component / Owner     │",
            "├──────────────────────────────┼────────────┼──────────┼───────────────────────┤",
        ]
    )
    for stage in record.waterfall:
        lines.append(
            f"│ {stage.label[:28]:<28} │ {stage.duration_ms:>8.1f}ms │ {stage.share_pct:>6.1f}%  │ {stage.owner[:21]:<21} │"
        )
    lines.extend(
        [
            "├──────────────────────────────┼────────────┼──────────┼───────────────────────┤",
            f"│ {'E2E (from user silence)':<28} │ {record.e2e_from_user_silence_ms:>8.1f}ms │ {'100.0%':>7}  │ {'SileroVAD -> WebRTC':<21} │",
            f"│ {'Net Pipeline (ex-VAD wait)':<28} │ {record.e2e_from_vad_stop_ms:>8.1f}ms │ {'—':>7}  │ {'STT + LLM + TTS':<21} │",
            "├──────────────────────────────┴────────────┴──────────┴───────────────────────┤",
            "│ Per-Block Service Metrics (TTFB vs Total Stream Processing)                  │",
            f"│   • STT ({record.stt_model or 'gemini-3.5-transcribe-live'}): "
            f"finalize={record.stt_finalize_ms:.1f}ms | TTFB={record.stt_ttfb_ms or 0.0:.1f}ms"
            f" | stream={record.stt_utterance_processing_ms or 0.0:.1f}ms",
            f"│   • LLM ({record.llm_model or 'gemini-3.5-flash-lite'}): "
            f"TTFB={record.llm_ttfb_ms or 0.0:.1f}ms | TTFAT={record.llm_ttfat_ms or 0.0:.1f}ms"
            f" | total={record.llm_total_processing_ms or 0.0:.1f}ms",
            f"│   • TTS ({record.tts_model or 'gemini-3.8-flash-lite-tts'}): "
            f"sent_agg={record.sentence_aggregation_ms:.1f}ms | TTFB={record.tts_ttfb_ms or 0.0:.1f}ms"
            f" | TTFA={record.tts_ttfa_ms or 0.0:.1f}ms | total={record.tts_total_processing_ms or 0.0:.1f}ms",
        ]
    )
    if summary.user_turn_count > 0:
        lines.extend(
            [
                "├──────────────────────────────────────────────────────────────────────────────┤",
                f"│ Running Aggregates (n={summary.user_turn_count} user turns): "
                f"E2E P50={summary.e2e_from_user_silence.p50_ms:.1f}ms  "
                f"P95={summary.e2e_from_user_silence.p95_ms:.1f}ms  "
                f"Mean={summary.e2e_from_user_silence.mean_ms:.1f}ms",
                f"│   Block P50s -> STT Finalize: {summary.stt_finalize.p50_ms:.1f}ms | "
                f"LLM TTFB: {summary.llm_ttfb.p50_ms:.1f}ms | "
                f"TTS TTFB: {summary.tts_ttfb.p50_ms:.1f}ms",
            ]
        )
    lines.append("╰──────────────────────────────────────────────────────────────────────────────╯")
    return "\n".join(lines)


class _CascadeMetricsCollector(BaseObserver):
    """Collects per-service MetricsFrames, transcripts, and post-TTFB totals per turn."""

    def __init__(self, monitor: CascadeLatencyMonitor):
        super().__init__(observe_every_push=False)
        self._monitor = monitor

    async def on_push_frame(self, data: FramePushed) -> None:
        if data.direction != FrameDirection.DOWNSTREAM:
            return
        await self._monitor.handle_downstream_frame(data.frame)


class CascadeLatencyMonitor:
    """Coordinates Pipecat's UserBotLatencyObserver with per-block service metrics."""

    def __init__(
        self,
        *,
        session_id: str,
        store: LatencyStore,
        stt_model: str,
        llm_model: str,
        tts_model: str,
        emit_rtvi_messages: bool = True,
        log_reports: bool = True,
        time_source: Any = time.time,
    ):
        self.session_id = session_id
        self.store = store
        self.stt_model = stt_model
        self.llm_model = llm_model
        self.tts_model = tts_model
        self._emit_rtvi = emit_rtvi_messages
        self._log_reports = log_reports
        self._now = time_source

        self._worker: PipelineWorker | None = None
        self._turn_counter = 0
        self._completed_turns: dict[int, TurnLatencyRecord] = {}
        self._turn_waiters: dict[int, asyncio.Future[TurnLatencyRecord]] = {}

        self.user_bot_observer = UserBotLatencyObserver(
            min_contribution_secs=0.0,
            time_source=time_source,
        )
        self.metrics_observer = _CascadeMetricsCollector(self)

        @self.user_bot_observer.event_handler("on_latency_breakdown")
        async def _on_breakdown(_observer: UserBotLatencyObserver, breakdown: LatencyBreakdown):
            await self._handle_latency_breakdown(breakdown)

        self._reset_cycle_metrics()

    def _reset_cycle_metrics(self) -> None:
        self._pending_user_transcript_parts: list[str] = []
        self._pending_assistant_text_parts: list[str] = []
        self._stt_ttfb_ms: float | None = None
        self._stt_processing_ms: float | None = None
        self._stt_audio_secs: float = 0.0
        self._smart_turn_e2e_ms: float | None = None
        self._smart_turn_prob: float | None = None
        self._llm_ttfb_ms: float | None = None
        self._llm_ttfat_ms: float | None = None
        self._llm_thinking_ms: float | None = None
        self._llm_processing_ms: float | None = None
        self._llm_prompt_tokens: int | None = None
        self._llm_completion_tokens: int | None = None
        self._llm_reasoning_tokens: int | None = None
        self._sentence_agg_ms: float | None = None
        self._tts_ttfb_ms: float | None = None
        self._tts_ttfa_ms: float | None = None
        self._tts_leading_silence_ms: float | None = None
        self._tts_processing_ms: float | None = None
        self._tts_chars: int = 0
        self._active_record: TurnLatencyRecord | None = None
        self._bot_started_at: float | None = None
        self._bot_last_stopped_at: float | None = None
        self._bot_is_speaking: bool = False
        self._saw_llm_text: bool = False
        self._assistant_turn_finished: bool = False

    def observers(self) -> list[BaseObserver]:
        """Return the observers to pass to `PipelineWorker(..., observers=...)`."""
        return [self.metrics_observer, self.user_bot_observer]

    def attach_worker(self, worker: PipelineWorker) -> None:
        """Attach the owning PipelineWorker so RTVI latency frames can be queued."""
        self._worker = worker

    @staticmethod
    def _classify_processor(processor_name: str) -> Literal["stt", "llm", "tts", "other"]:
        lowered = processor_name.lower()
        if "stt" in lowered or "transcribe" in lowered:
            return "stt"
        if "tts" in lowered:
            return "tts"
        if "llm" in lowered:
            return "llm"
        return "other"

    async def handle_downstream_frame(self, frame: Any) -> None:
        if isinstance(frame, VADUserStartedSpeakingFrame):
            if self._active_record is not None and not self._active_record.completed:
                await self._finalize_active_record(interrupted=True)
            self._reset_cycle_metrics()
            return

        if isinstance(frame, InterruptionFrame):
            if self._active_record is not None and not self._active_record.completed:
                await self._finalize_active_record(interrupted=True)
            return

        if isinstance(frame, TranscriptionFrame):
            if frame.text.strip():
                self._pending_user_transcript_parts.append(frame.text.strip())
                if self._active_record is not None and not self._active_record.user_transcript:
                    self._active_record.user_transcript = " ".join(
                        self._pending_user_transcript_parts
                    )
            return

        if isinstance(frame, LLMContextUserTurnFrame):
            if frame.text.strip():
                self._pending_user_transcript_parts = [frame.text.strip()]
                if self._active_record is not None:
                    self._active_record.user_transcript = frame.text.strip()
            return

        if isinstance(frame, InterimTranscriptionFrame):
            return

        if isinstance(frame, LLMTextFrame):
            if frame.text:
                self._saw_llm_text = True
                self._pending_assistant_text_parts.append(frame.text)
                if self._active_record is not None:
                    self._active_record.assistant_transcript = "".join(
                        self._pending_assistant_text_parts
                    ).strip()
            return

        if isinstance(frame, LLMContextAssistantTurnFrame):
            self._assistant_turn_finished = True
            if frame.text.strip():
                self._pending_assistant_text_parts = [frame.text.strip()]
                if self._active_record is not None:
                    self._active_record.assistant_transcript = frame.text.strip()
            if (
                not self._bot_is_speaking
                and self._active_record is not None
                and not self._active_record.completed
            ):
                await self._finalize_active_record(interrupted=False)
            return

        if isinstance(frame, MetricsFrame):
            self._ingest_metrics_frame(frame)
            if self._active_record is not None:
                self._apply_latest_metrics_to_record(self._active_record)
            return

        if isinstance(frame, BotStartedSpeakingFrame):
            if self._bot_started_at is None:
                self._bot_started_at = self._now()
            self._bot_is_speaking = True
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            if self._active_record is not None:
                self._apply_latest_metrics_to_record(self._active_record)
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_is_speaking = False
            self._bot_last_stopped_at = self._now()
            if (
                (self._assistant_turn_finished or not self._saw_llm_text)
                and self._active_record is not None
                and not self._active_record.completed
            ):
                await self._finalize_active_record(interrupted=False)

    def _ingest_metrics_frame(self, frame: MetricsFrame) -> None:
        for item in frame.data:
            role = self._classify_processor(item.processor)
            if isinstance(item, TTFBMetricsData):
                val_ms = _ms(item.value)
                if role == "stt" and self._stt_ttfb_ms is None:
                    self._stt_ttfb_ms = val_ms
                elif role == "llm" and self._llm_ttfb_ms is None:
                    self._llm_ttfb_ms = val_ms
                elif role == "tts" and self._tts_ttfb_ms is None:
                    self._tts_ttfb_ms = val_ms
            elif isinstance(item, TTFATMetricsData):
                if self._llm_ttfat_ms is None:
                    self._llm_ttfat_ms = _ms(item.ttfat)
                    self._llm_thinking_ms = _ms(item.thinking_time)
            elif isinstance(item, TTFAMetricsData):
                if self._tts_ttfa_ms is None:
                    self._tts_ttfa_ms = _ms(item.ttfa)
                    self._tts_leading_silence_ms = _ms(item.leading_silence)
            elif isinstance(item, TextAggregationMetricsData):
                if self._sentence_agg_ms is None:
                    self._sentence_agg_ms = _ms(item.value)
            elif isinstance(item, ProcessingMetricsData):
                val_ms = _ms(item.value) or 0.0
                if role == "stt":
                    self._stt_processing_ms = val_ms
                elif role == "llm":
                    self._llm_processing_ms = round(
                        (self._llm_processing_ms or 0.0) + val_ms, 2
                    )
                elif role == "tts":
                    self._tts_processing_ms = round(
                        (self._tts_processing_ms or 0.0) + val_ms, 2
                    )
            elif isinstance(item, LLMUsageMetricsData):
                self._llm_prompt_tokens = item.value.prompt_tokens
                self._llm_completion_tokens = (
                    (self._llm_completion_tokens or 0) + item.value.completion_tokens
                )
                if item.value.reasoning_tokens is not None:
                    self._llm_reasoning_tokens = (
                        (self._llm_reasoning_tokens or 0) + item.value.reasoning_tokens
                    )
            elif isinstance(item, TTSUsageMetricsData):
                self._tts_chars += item.value
            elif isinstance(item, STTUsageMetricsData):
                self._stt_audio_secs = round(self._stt_audio_secs + item.value.audio_seconds, 3)
            elif isinstance(item, TurnMetricsData):
                self._smart_turn_e2e_ms = round(item.e2e_processing_time_ms, 2)
                self._smart_turn_prob = round(item.probability, 4)

    def _apply_latest_metrics_to_record(self, rec: TurnLatencyRecord) -> None:
        if self._stt_ttfb_ms is not None:
            rec.stt_ttfb_ms = self._stt_ttfb_ms
        if self._stt_processing_ms is not None:
            rec.stt_utterance_processing_ms = self._stt_processing_ms
        if self._stt_audio_secs > 0:
            rec.stt_audio_seconds = self._stt_audio_secs
        if self._smart_turn_e2e_ms is not None:
            rec.smart_turn_e2e_ms = self._smart_turn_e2e_ms
        if self._smart_turn_prob is not None:
            rec.smart_turn_probability = self._smart_turn_prob
        if self._llm_ttfb_ms is not None:
            rec.llm_ttfb_ms = self._llm_ttfb_ms
        if self._llm_ttfat_ms is not None:
            rec.llm_ttfat_ms = self._llm_ttfat_ms
        if self._llm_thinking_ms is not None:
            rec.llm_thinking_ms = self._llm_thinking_ms
        if self._llm_processing_ms is not None:
            rec.llm_total_processing_ms = self._llm_processing_ms
        if self._llm_prompt_tokens is not None:
            rec.llm_prompt_tokens = self._llm_prompt_tokens
        if self._llm_completion_tokens is not None:
            rec.llm_completion_tokens = self._llm_completion_tokens
        if self._llm_reasoning_tokens is not None:
            rec.llm_reasoning_tokens = self._llm_reasoning_tokens
        if self._tts_ttfb_ms is not None:
            rec.tts_ttfb_ms = self._tts_ttfb_ms
        if self._tts_ttfa_ms is not None:
            rec.tts_ttfa_ms = self._tts_ttfa_ms
        if self._tts_leading_silence_ms is not None:
            rec.tts_leading_silence_ms = self._tts_leading_silence_ms
        if self._tts_processing_ms is not None:
            rec.tts_total_processing_ms = self._tts_processing_ms
        if self._tts_chars > 0:
            rec.tts_characters = self._tts_chars
        if self._pending_assistant_text_parts:
            rec.assistant_transcript = "".join(self._pending_assistant_text_parts).strip()

    async def _handle_latency_breakdown(self, breakdown: LatencyBreakdown) -> None:
        self._turn_counter += 1
        turn_type: Literal["user_turn", "greeting"] = (
            "greeting" if breakdown.measured_from == MeasuredFrom.CLIENT_CONNECTED else "user_turn"
        )
        total_ms = _ms(breakdown.total_secs) or 0.0
        anchor_time = (
            breakdown.contributions[0].start_time if breakdown.contributions else self._now()
        )

        by_key: dict[str, float] = {}
        waterfall: list[WaterfallStage] = []
        running_offset_ms = 0.0
        for contrib in breakdown.contributions:
            dur_ms = _ms(contrib.duration_secs) or 0.0
            by_key[contrib.key] = round(by_key.get(contrib.key, 0.0) + dur_ms, 2)
            offset_ms = (
                running_offset_ms
                if contrib.key == "pipeline"
                else max(0.0, round((contrib.start_time - anchor_time) * 1000.0, 2))
            )
            running_offset_ms = round(max(running_offset_ms, offset_ms + dur_ms), 2)
            share_pct = round((dur_ms / total_ms) * 100.0, 1) if total_ms > 0 else 0.0
            waterfall.append(
                WaterfallStage(
                    key=contrib.key,
                    label=contrib.label,
                    owner=contrib.owner,
                    owner_kind=str(contrib.owner_kind),
                    offset_ms=offset_ms,
                    duration_ms=dur_ms,
                    share_pct=share_pct,
                )
            )

        # Fallback: also inspect breakdown.ttfb in case any TTFB arrived in same frame batch
        for ttfb_entry in breakdown.ttfb:
            role = self._classify_processor(ttfb_entry.processor)
            val_ms = _ms(ttfb_entry.duration_secs)
            if role == "stt" and self._stt_ttfb_ms is None:
                self._stt_ttfb_ms = val_ms
            elif role == "llm" and self._llm_ttfb_ms is None:
                self._llm_ttfb_ms = val_ms
            elif role == "tts" and self._tts_ttfb_ms is None:
                self._tts_ttfb_ms = val_ms

        if breakdown.text_aggregation and self._sentence_agg_ms is None:
            self._sentence_agg_ms = _ms(breakdown.text_aggregation.duration_secs)

        endpointing_ms = by_key.get("endpointing_wait", 0.0)
        sentence_agg_ms = by_key.get("sentence_aggregation", self._sentence_agg_ms or 0.0)

        record = TurnLatencyRecord(
            turn_id=f"{self.session_id}-t{self._turn_counter}",
            session_id=self.session_id,
            turn_index=self._turn_counter,
            turn_type=turn_type,
            timestamp_utc=datetime.now(UTC).isoformat(),
            completed=False,
            user_transcript=" ".join(self._pending_user_transcript_parts).strip(),
            assistant_transcript="".join(self._pending_assistant_text_parts).strip(),
            e2e_from_user_silence_ms=total_ms,
            e2e_from_vad_stop_ms=round(max(0.0, total_ms - endpointing_ms), 2),
            user_turn_release_ms=_ms(breakdown.user_turn_secs),
            endpointing_wait_ms=endpointing_ms,
            stt_finalize_ms=by_key.get("transcription", 0.0),
            turn_detection_ms=by_key.get("turn_detection", 0.0),
            llm_inference_span_ms=by_key.get("llm_inference", 0.0),
            sentence_aggregation_ms=sentence_agg_ms,
            tts_synthesis_span_ms=by_key.get("speech_synthesis", 0.0),
            output_transport_ms=by_key.get("output_transport", 0.0),
            pipeline_overhead_ms=by_key.get("pipeline", 0.0),
            stt_model=self.stt_model,
            llm_model=self.llm_model,
            tts_model=self.tts_model,
            waterfall=waterfall,
        )
        self._apply_latest_metrics_to_record(record)
        self._active_record = record

        summary = self.store.upsert(record, persist=False)
        if self._log_reports:
            logger.info(format_turn_report(record, summary))
        await self._push_rtvi_event("cascade_latency", record, summary)
        if (
            not self._bot_is_speaking
            and self._bot_last_stopped_at is not None
            and (self._assistant_turn_finished or not self._saw_llm_text)
        ):
            await self._finalize_active_record(interrupted=False)

    async def _finalize_active_record(self, *, interrupted: bool) -> None:
        rec = self._active_record
        if rec is None or rec.completed:
            return
        self._apply_latest_metrics_to_record(rec)
        if self._bot_started_at is not None:
            stop_ts = self._bot_last_stopped_at or self._now()
            rec.bot_speech_duration_ms = _ms(max(0.0, stop_ts - self._bot_started_at))
        rec.completed = True
        rec.interrupted = interrupted
        summary = self.store.upsert(rec, persist=True)
        self._completed_turns[rec.turn_index] = rec
        waiter = self._turn_waiters.pop(rec.turn_index, None)
        if waiter is not None and not waiter.done():
            waiter.set_result(rec)
        await self._push_rtvi_event("cascade_latency_update", rec, summary)
        self._active_record = None
        self._bot_started_at = None
        self._bot_last_stopped_at = None

    async def wait_for_turn(self, turn_index: int, *, timeout: float = 45.0) -> TurnLatencyRecord:
        """Await finalization of the given 1-based turn index."""
        if turn_index in self._completed_turns:
            return self._completed_turns[turn_index]
        fut = self._turn_waiters.get(turn_index)
        if fut is None:
            fut = asyncio.get_running_loop().create_future()
            self._turn_waiters[turn_index] = fut
        return await asyncio.wait_for(fut, timeout=timeout)

    async def flush_open_turn(self) -> None:
        """Persist any in-progress turn record when the pipeline shuts down."""
        await self.user_bot_observer.cleanup()
        if self._active_record is not None and not self._active_record.completed:
            await self._finalize_active_record(interrupted=False)

    async def _push_rtvi_event(
        self, event_type: str, record: TurnLatencyRecord, summary: LatencySummary
    ) -> None:
        if not self._emit_rtvi or self._worker is None:
            return
        try:
            await self._worker.queue_frame(
                RTVIServerMessageFrame(
                    data={
                        "type": event_type,
                        "turn": record.model_dump(),
                        "summary": summary.model_dump(),
                    }
                )
            )
        except Exception as exc:
            logger.debug(f"Skipping RTVI latency push on closing worker: {exc}")
