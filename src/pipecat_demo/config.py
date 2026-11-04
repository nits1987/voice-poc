"""Configuration loader for the Gemini STT -> LLM -> TTS cascade voice bot."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from pipecat_demo.telephony import (
    STT_PIPELINE_SAMPLE_RATE,
    TELEPHONY_SAMPLE_RATE,
    TTS_NATIVE_SAMPLE_RATE,
    DialerEncoding,
    normalize_dialer_encoding,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
LOGS_DIR = PROJECT_ROOT / "logs"

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a concise, knowledgeable voice assistant answering general questions. "
    "Keep every answer clear, accurate, and natural to hear aloud in 1 to 3 short sentences. "
    "Never use markdown, bullet points, asterisks, URLs, or code blocks because your text "
    "is synthesized directly into speech."
)

ThinkingLevel = Literal["minimal", "low", "medium", "high"]
TurnStopMode = Literal["smart_turn", "vad"]


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CascadeConfig:
    """Runtime settings for the three-block Gemini voice cascade + 8kHz telephony converter."""

    api_key: str
    stt_model: str = "gemini-3.5-transcribe-live"
    stt_mode: str = "SMART"
    llm_model: str = "gemini-3.5-flash-lite"
    llm_thinking_level: ThinkingLevel = "minimal"
    llm_system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION
    tts_model: str = "gemini-3.8-flash-lite-tts"
    tts_voice: str = "Kore"
    tts_style_prompt: str = "Warm, clear, natural conversational voice at a brisk, helpful pace."
    audio_in_sample_rate: int = STT_PIPELINE_SAMPLE_RATE
    audio_out_sample_rate: int = TTS_NATIVE_SAMPLE_RATE
    telephony_8k_enabled: bool = True
    dialer_sample_rate: int = TELEPHONY_SAMPLE_RATE
    dialer_encoding: DialerEncoding = "ulaw"
    vad_stop_secs: float = 0.2
    turn_stop_mode: TurnStopMode = "smart_turn"
    greet_on_connect: bool = True
    metrics_jsonl_path: Path = LOGS_DIR / "latency_metrics.jsonl"

    @property
    def has_api_key(self) -> bool:
        """Return True if a non-placeholder Gemini API key is configured."""
        cleaned = self.api_key.strip()
        if not cleaned:
            return False
        placeholders = {"your_api_key_here", "your-api-key", "changeme", "<your-api-key>"}
        return cleaned.lower() not in placeholders

    def require_api_key(self) -> str:
        """Return the configured API key or raise a clear RuntimeError."""
        if not self.has_api_key:
            raise RuntimeError(
                f"GOOGLE_API_KEY is not set. Add your Gemini API key to {ENV_PATH} "
                "or export GOOGLE_API_KEY in your shell before starting a voice session."
            )
        return self.api_key.strip()

    def with_telephony_8k(
        self,
        *,
        enabled: bool = True,
        encoding: str | None = None,
        dialer_sample_rate: int | None = None,
    ) -> CascadeConfig:
        """Return a copy configured for 8kHz telephony output (or wideband 24kHz when disabled)."""
        resolved_rate = dialer_sample_rate or self.dialer_sample_rate or TELEPHONY_SAMPLE_RATE
        resolved_enc = (
            normalize_dialer_encoding(encoding, default=self.dialer_encoding)
            if encoding is not None
            else self.dialer_encoding
        )
        return dataclasses.replace(
            self,
            telephony_8k_enabled=enabled,
            dialer_sample_rate=resolved_rate,
            dialer_encoding=resolved_enc,
            audio_out_sample_rate=resolved_rate if enabled else TTS_NATIVE_SAMPLE_RATE,
        )

    def public_summary(self) -> dict[str, object]:
        """Return safe configuration metadata for the Web UI and JSONL logs."""
        return {
            "api_key_configured": self.has_api_key,
            "env_file": str(ENV_PATH),
            "stt": {
                "model": self.stt_model,
                "mode": self.stt_mode,
                "sample_rate_hz": self.audio_in_sample_rate,
            },
            "llm": {
                "model": self.llm_model,
                "thinking_level": self.llm_thinking_level,
            },
            "tts": {
                "model": self.tts_model,
                "voice": self.tts_voice,
                "native_sample_rate_hz": TTS_NATIVE_SAMPLE_RATE,
                "sample_rate_hz": self.audio_out_sample_rate,
            },
            "telephony": {
                "enabled": self.telephony_8k_enabled,
                "dialer_sample_rate_hz": self.dialer_sample_rate,
                "dialer_encoding": self.dialer_encoding,
                "input_converter": f"{self.dialer_sample_rate}Hz -> {self.audio_in_sample_rate}Hz (SOXR)",
                "output_converter": f"{TTS_NATIVE_SAMPLE_RATE}Hz -> {self.dialer_sample_rate}Hz (SOXR)",
                "supported_dialers": ["twilio", "exotel", "plivo", "telnyx", "ws-dialer"],
                "supported_codecs": ["ulaw", "alaw", "pcm16"],
            },
            "turn_detection": {
                "vad_stop_secs": self.vad_stop_secs,
                "turn_stop_mode": self.turn_stop_mode,
                "greet_on_connect": self.greet_on_connect,
            },
        }


def load_config(*, override_env: bool = True) -> CascadeConfig:
    """Load cascade configuration from `.env` and process environment variables."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=override_env)
    else:
        load_dotenv(override=override_env)

    api_key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    thinking_raw = (os.getenv("GEMINI_LLM_THINKING_LEVEL") or "minimal").strip().lower()
    thinking_level: ThinkingLevel = (
        thinking_raw if thinking_raw in {"minimal", "low", "medium", "high"} else "minimal"  # type: ignore[assignment]
    )
    turn_raw = (os.getenv("TURN_STOP_MODE") or "smart_turn").strip().lower()
    turn_mode: TurnStopMode = "vad" if turn_raw == "vad" else "smart_turn"

    try:
        vad_stop_secs = float(os.getenv("VAD_STOP_SECS", "0.2"))
    except ValueError:
        vad_stop_secs = 0.2

    try:
        dialer_sample_rate = int(os.getenv("DIALER_SAMPLE_RATE", str(TELEPHONY_SAMPLE_RATE)))
    except ValueError:
        dialer_sample_rate = TELEPHONY_SAMPLE_RATE

    telephony_8k_enabled = _parse_bool(os.getenv("TELEPHONY_8K_ENABLED"), default=True)
    dialer_encoding = normalize_dialer_encoding(os.getenv("DIALER_ENCODING"), default="ulaw")

    return CascadeConfig(
        api_key=api_key,
        stt_model=(os.getenv("GEMINI_STT_MODEL") or "gemini-3.5-transcribe-live").strip(),
        stt_mode=(os.getenv("GEMINI_STT_MODE") or "SMART").strip().upper(),
        llm_model=(os.getenv("GEMINI_LLM_MODEL") or "gemini-3.5-flash-lite").strip(),
        llm_thinking_level=thinking_level,
        llm_system_instruction=(
            os.getenv("GEMINI_SYSTEM_INSTRUCTION") or DEFAULT_SYSTEM_INSTRUCTION
        ).strip(),
        tts_model=(os.getenv("GEMINI_TTS_MODEL") or "gemini-3.8-flash-lite-tts").strip(),
        tts_voice=(os.getenv("GEMINI_TTS_VOICE") or "Kore").strip(),
        tts_style_prompt=(
            os.getenv("GEMINI_TTS_STYLE")
            or "Warm, clear, natural conversational voice at a brisk, helpful pace."
        ).strip(),
        telephony_8k_enabled=telephony_8k_enabled,
        dialer_sample_rate=dialer_sample_rate,
        dialer_encoding=dialer_encoding,
        vad_stop_secs=vad_stop_secs,
        turn_stop_mode=turn_mode,
        greet_on_connect=_parse_bool(os.getenv("GREET_ON_CONNECT"), default=True),
    )
