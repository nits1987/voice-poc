"""8kHz Telephony & Dialer audio converter, resampler, and WebSocket serializers.

Bridges 8kHz PSTN / SIP / Cloud telephony dialers (Twilio, Exotel, Plivo, Telnyx,
Asterisk, FreeSWITCH, Genesys, Ozonetel, and custom WebSocket dialers) with the
wideband Gemini cascade pipeline:

- Inbound caller audio:  8kHz (G.711 μ-law / A-law / PCM16) -> 16kHz PCM16LE
                         for Silero VAD + SmartTurn v3 ONNX + Gemini Live STT
                         (`gemini-3.5-transcribe-live`).
- Outbound bot audio:    24kHz PCM16LE from Gemini TTS (`gemini-3.8-flash-lite-tts`)
                         -> 8kHz (PCM16LE / G.711 μ-law / G.711 A-law) for the
                         telephony dialer output transport.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger
from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler
from pipecat.audio.utils import (
    alaw_to_pcm,
    create_stream_resampler,
    pcm_to_alaw,
    pcm_to_ulaw,
    ulaw_to_pcm,
)
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.serializers.base_serializer import FrameSerializer

TELEPHONY_SAMPLE_RATE: int = 8000
STT_PIPELINE_SAMPLE_RATE: int = 16000
TTS_NATIVE_SAMPLE_RATE: int = 24000

DialerEncoding = Literal["ulaw", "alaw", "pcm16"]


def normalize_dialer_encoding(
    raw: str | None,
    *,
    default: DialerEncoding = "ulaw",
) -> DialerEncoding:
    """Normalize a codec/encoding string into 'ulaw', 'alaw', or 'pcm16'."""
    if not raw:
        return default
    cleaned = raw.strip().lower().replace("-", "").replace("_", "").replace("audio/", "")
    if cleaned in {"ulaw", "mulaw", "pcmu", "g711u", "g711ulaw", "xmulaw"}:
        return "ulaw"
    if cleaned in {"alaw", "pcma", "g711a", "g711alaw", "xalaw"}:
        return "alaw"
    if cleaned in {"pcm16", "pcm", "linear16", "slin", "l16", "s16le", "pcm16le", "raw"}:
        return "pcm16"
    return default


async def resample_pcm16(
    pcm_bytes: bytes,
    in_rate: int,
    out_rate: int,
    *,
    resampler: BaseAudioResampler | None = None,
    flush: bool | None = None,
) -> bytes:
    """Resample 16-bit little-endian mono PCM between `in_rate` and `out_rate`."""
    if not pcm_bytes:
        return b""
    if len(pcm_bytes) % 2 == 1:
        pcm_bytes = pcm_bytes[:-1]
    if in_rate == out_rate:
        return pcm_bytes

    owns_resampler = resampler is None
    active_resampler = resampler or create_stream_resampler()
    should_flush = owns_resampler if flush is None else flush

    converted = await active_resampler.resample(pcm_bytes, in_rate, out_rate)
    if should_flush:
        converted += await active_resampler.flush()
    return converted


async def convert_8k_to_16k_pcm(
    audio_bytes: bytes,
    *,
    encoding: DialerEncoding = "pcm16",
    in_rate: int = TELEPHONY_SAMPLE_RATE,
    out_rate: int = STT_PIPELINE_SAMPLE_RATE,
    resampler: BaseAudioResampler | None = None,
    flush: bool | None = None,
) -> bytes:
    """Convert 8kHz dialer audio (`pcm16`, `ulaw`, or `alaw`) into 16kHz PCM16LE for Gemini STT."""
    if not audio_bytes:
        return b""

    owns_resampler = resampler is None
    active_resampler = resampler or create_stream_resampler()
    should_flush = owns_resampler if flush is None else flush
    norm_enc = normalize_dialer_encoding(encoding)

    if norm_enc == "ulaw":
        pcm_out = await ulaw_to_pcm(audio_bytes, in_rate, out_rate, active_resampler) or b""
        if should_flush and in_rate != out_rate:
            pcm_out += await active_resampler.flush()
        return pcm_out

    if norm_enc == "alaw":
        pcm_out = await alaw_to_pcm(audio_bytes, in_rate, out_rate, active_resampler) or b""
        if should_flush and in_rate != out_rate:
            pcm_out += await active_resampler.flush()
        return pcm_out

    return await resample_pcm16(
        audio_bytes,
        in_rate,
        out_rate,
        resampler=active_resampler,
        flush=should_flush,
    )


async def convert_24k_to_8k_pcm(
    pcm24k_bytes: bytes,
    *,
    in_rate: int = TTS_NATIVE_SAMPLE_RATE,
    out_rate: int = TELEPHONY_SAMPLE_RATE,
    resampler: BaseAudioResampler | None = None,
    flush: bool | None = None,
) -> bytes:
    """Downsample 24kHz Gemini TTS PCM16LE audio to 8kHz PCM16LE for telephony dialers."""
    return await resample_pcm16(
        pcm24k_bytes,
        in_rate,
        out_rate,
        resampler=resampler,
        flush=flush,
    )


async def encode_dialer_audio_from_pcm(
    pcm_bytes: bytes,
    *,
    in_rate: int = TTS_NATIVE_SAMPLE_RATE,
    out_rate: int = TELEPHONY_SAMPLE_RATE,
    encoding: DialerEncoding = "ulaw",
    resampler: BaseAudioResampler | None = None,
    flush: bool | None = None,
) -> bytes:
    """Resample PCM16LE from `in_rate` to `out_rate` (8kHz) and encode to `ulaw`, `alaw`, or `pcm16`."""
    if not pcm_bytes:
        return b""
    if len(pcm_bytes) % 2 == 1:
        pcm_bytes = pcm_bytes[:-1]

    owns_resampler = resampler is None
    active_resampler = resampler or create_stream_resampler()
    should_flush = owns_resampler if flush is None else flush
    norm_enc = normalize_dialer_encoding(encoding)

    pcm_8k = await resample_pcm16(
        pcm_bytes,
        in_rate,
        out_rate,
        resampler=active_resampler,
        flush=should_flush,
    )
    if not pcm_8k:
        return b""

    if norm_enc == "ulaw":
        # pcm_8k is already at out_rate, so pass matching in/out rates to encode directly
        return await pcm_to_ulaw(pcm_8k, out_rate, out_rate, active_resampler) or b""
    if norm_enc == "alaw":
        return await pcm_to_alaw(pcm_8k, out_rate, out_rate, active_resampler) or b""
    return pcm_8k


async def simulate_pstn_8k_roundtrip(
    pcm16k_bytes: bytes,
    *,
    encoding: DialerEncoding = "ulaw",
    dialer_rate: int = TELEPHONY_SAMPLE_RATE,
) -> bytes:
    """Convert 16kHz PCM speech into 8kHz PSTN-quantized PCM16LE (simulating an 8kHz dialer line).

    Downsamples 16kHz -> 8kHz, applies G.711 μ-law or A-law companding when configured,
    and decodes back to 8kHz 16-bit PCM so it can be streamed as 8kHz `InputAudioRawFrame`s
    through `Telephony8kInputConverter`.
    """
    encoded_8k = await encode_dialer_audio_from_pcm(
        pcm16k_bytes,
        in_rate=STT_PIPELINE_SAMPLE_RATE,
        out_rate=dialer_rate,
        encoding=encoding,
        flush=True,
    )
    return await convert_8k_to_16k_pcm(
        encoded_8k,
        encoding=encoding,
        in_rate=dialer_rate,
        out_rate=dialer_rate,
        flush=True,
    )


@dataclass
class ConverterTelemetry:
    """Runtime counters for the 8kHz <-> 16kHz/24kHz pipeline converters."""

    frames_converted: int = 0
    frames_passthrough: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    last_source_sample_rate: int | None = None
    target_sample_rate: int = 0


class Telephony8kInputConverter(FrameProcessor):
    """Pipeline processor that upsamples 8kHz dialer input frames to 16kHz for Gemini STT + VAD.

    Placed immediately after `transport.input()` and before `GeminiSTTService`:
    - If an `InputAudioRawFrame` arrives at 8000 Hz (or any rate != 16000 Hz), it is
      resampled streamingly via `SOXRStreamAudioResampler` to `16000 Hz`.
    - If a serializer (`TwilioFrameSerializer`, `ExotelFrameSerializer`, etc.) has already
      upsampled the frame to `16000 Hz`, the frame passes through untouched without
      double-resampling.
    """

    def __init__(
        self,
        *,
        target_sample_rate: int = STT_PIPELINE_SAMPLE_RATE,
        name: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(name=name or "Telephony8kInputConverter", **kwargs)
        self._target_sample_rate = target_sample_rate
        # Use low-latency streaming resampler (quality="QQ", clear_after_secs=None)
        # so every 20ms 8kHz frame (160 samples) immediately emits a 20ms 16kHz frame
        # without the ~128ms (1,024-sample) lookahead block delay of VHQ.
        self._resampler = create_stream_resampler(quality="QQ", clear_after_secs=None)
        self.telemetry = ConverterTelemetry(target_sample_rate=target_sample_rate)

    @property
    def target_sample_rate(self) -> int:
        return self._target_sample_rate

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            src_rate = frame.sample_rate or TELEPHONY_SAMPLE_RATE
            self.telemetry.last_source_sample_rate = src_rate
            self.telemetry.bytes_in += len(frame.audio)

            if src_rate != self._target_sample_rate and frame.audio:
                upsampled = await self._resampler.resample(
                    frame.audio,
                    src_rate,
                    self._target_sample_rate,
                )
                self.telemetry.frames_converted += 1
                self.telemetry.bytes_out += len(upsampled)
                if upsampled:
                    converted_frame = InputAudioRawFrame(
                        audio=upsampled,
                        sample_rate=self._target_sample_rate,
                        num_channels=frame.num_channels,
                    )
                    await self.push_frame(converted_frame, direction)
                return

            self.telemetry.frames_passthrough += 1
            self.telemetry.bytes_out += len(frame.audio)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            self._resampler = create_stream_resampler(quality="QQ", clear_after_secs=None)

        await self.push_frame(frame, direction)


class Telephony8kOutputConverter(FrameProcessor):
    """Pipeline processor that downsamples 24kHz Gemini TTS audio frames to 8kHz for dialers.

    Placed immediately after `GeminiTTSService` (24kHz native) and before `transport.output()`:
    - When `enabled=True`, downsamples `24000 Hz` `TTSAudioRawFrame` / `OutputAudioRawFrame`
      to `8000 Hz` using `SOXRStreamAudioResampler`.
    - Flushes the SOXR filter tail on `TTSStoppedFrame` / `EndFrame` so trailing consonants
      are never clipped.
    - Resets resampler state on `InterruptionFrame` / `CancelFrame` so barge-in leaves no
      stale audio samples.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        target_sample_rate: int = TELEPHONY_SAMPLE_RATE,
        name: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(name=name or "Telephony8kOutputConverter", **kwargs)
        self._enabled = enabled
        self._target_sample_rate = target_sample_rate
        self._resampler = create_stream_resampler(quality="QQ", clear_after_secs=None)
        self._last_num_channels: int = 1
        self._last_destination: str | None = None
        self._last_interruptible: bool = True
        self._last_context_id: str | None = None
        self._has_pending_tail: bool = False
        self.telemetry = ConverterTelemetry(target_sample_rate=target_sample_rate)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def target_sample_rate(self) -> int:
        return self._target_sample_rate

    def _build_output_frame(
        self,
        audio_bytes: bytes,
        template: OutputAudioRawFrame | None = None,
    ) -> OutputAudioRawFrame:
        num_channels = template.num_channels if template is not None else self._last_num_channels
        destination = (
            template.transport_destination if template is not None else self._last_destination
        )
        interruptible = (
            template.interruptible if template is not None else self._last_interruptible
        )
        context_id = (
            getattr(template, "context_id", None) if template is not None else self._last_context_id
        )

        if isinstance(template, TTSAudioRawFrame) or template is None:
            out_frame: OutputAudioRawFrame = TTSAudioRawFrame(
                audio=audio_bytes,
                sample_rate=self._target_sample_rate,
                num_channels=num_channels,
                context_id=context_id,
            )
        else:
            out_frame = OutputAudioRawFrame(
                audio=audio_bytes,
                sample_rate=self._target_sample_rate,
                num_channels=num_channels,
            )
        out_frame.transport_destination = destination
        out_frame.interruptible = interruptible
        return out_frame

    async def _flush_tail(self, direction: FrameDirection) -> None:
        if not self._enabled or not self._has_pending_tail:
            return
        self._has_pending_tail = False
        tail = await self._resampler.flush()
        if tail:
            if len(tail) % 2 == 1:
                tail = tail + b"\x00"
            self.telemetry.bytes_out += len(tail)
            await self.push_frame(self._build_output_frame(tail), direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if not self._enabled:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, OutputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            src_rate = frame.sample_rate or TTS_NATIVE_SAMPLE_RATE
            self.telemetry.last_source_sample_rate = src_rate
            self.telemetry.bytes_in += len(frame.audio)

            self._last_num_channels = frame.num_channels
            self._last_destination = frame.transport_destination
            self._last_interruptible = frame.interruptible
            self._last_context_id = getattr(frame, "context_id", None)

            if src_rate != self._target_sample_rate and frame.audio:
                downsampled = await self._resampler.resample(
                    frame.audio,
                    src_rate,
                    self._target_sample_rate,
                )
                self._has_pending_tail = True
                self.telemetry.frames_converted += 1
                self.telemetry.bytes_out += len(downsampled)
                if downsampled:
                    await self.push_frame(self._build_output_frame(downsampled, frame), direction)
                return

            self.telemetry.frames_passthrough += 1
            self.telemetry.bytes_out += len(frame.audio)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TTSStoppedFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._flush_tail(direction)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, EndFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._flush_tail(direction)
            self._resampler = create_stream_resampler(quality="QQ", clear_after_secs=None)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (InterruptionFrame, CancelFrame)):
            self._has_pending_tail = False
            self._resampler = create_stream_resampler(quality="QQ", clear_after_secs=None)
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)


class Dialer8kFrameSerializer(FrameSerializer):
    """Universal 8kHz telephony WebSocket serializer (`ulaw`, `alaw`, or `pcm16`).

    Supports both:
    1. **JSON Media Envelopes** (`{"event": "media", "media": {"payload": "<base64>"}}`,
       `{"event": "start", ...}`, `{"event": "dtmf", ...}`, `{"event": "clear"}`)
       used by cloud telephony dialers and SIP WebSocket gateways.
    2. **Raw Binary Audio Frames** (raw 8kHz `ulaw`, `alaw`, or `pcm16` bytes) used by
       Asterisk AudioSocket, FreeSWITCH `mod_audio_stream`, and custom dialer bridges.
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for Dialer8kFrameSerializer."""

        dialer_sample_rate: int = TELEPHONY_SAMPLE_RATE
        sample_rate: int | None = None
        encoding: DialerEncoding = "ulaw"
        wire_format: Literal["auto", "json", "binary"] = "auto"

    def __init__(
        self,
        stream_sid: str = "dialer-8k",
        call_sid: str | None = None,
        params: InputParams | None = None,
    ):
        resolved_params = params or Dialer8kFrameSerializer.InputParams()
        super().__init__(resolved_params)
        self._params: Dialer8kFrameSerializer.InputParams = resolved_params
        self._stream_sid = stream_sid
        self._call_sid = call_sid
        self._dialer_sample_rate = self._params.dialer_sample_rate
        self._encoding: DialerEncoding = normalize_dialer_encoding(self._params.encoding)
        self._sample_rate = STT_PIPELINE_SAMPLE_RATE
        self._detected_wire_format: Literal["json", "binary"] = (
            "binary" if self._params.wire_format == "binary" else "json"
        )
        self._input_resampler = create_stream_resampler(
            quality="QQ",
            clear_after_secs=None,
        )
        self._output_resampler = create_stream_resampler(
            quality="QQ",
            clear_after_secs=None,
        )

    @property
    def encoding(self) -> DialerEncoding:
        return self._encoding

    @property
    def dialer_sample_rate(self) -> int:
        return self._dialer_sample_rate

    async def setup(self, setup: FrameProcessorSetup) -> None:
        self._sample_rate = self._params.sample_rate or setup.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, InterruptionFrame):
            if self._detected_wire_format == "binary":
                return None
            return json.dumps(
                {
                    "event": "clear",
                    "streamSid": self._stream_sid,
                    "stream_sid": self._stream_sid,
                }
            )

        if isinstance(frame, AudioRawFrame):
            encoded = await encode_dialer_audio_from_pcm(
                frame.audio,
                in_rate=frame.sample_rate or TTS_NATIVE_SAMPLE_RATE,
                out_rate=self._dialer_sample_rate,
                encoding=self._encoding,
                resampler=self._output_resampler,
                flush=False,
            )
            if not encoded:
                return None

            if self._detected_wire_format == "binary":
                return encoded

            payload_b64 = base64.b64encode(encoded).decode("ascii")
            return json.dumps(
                {
                    "event": "media",
                    "streamSid": self._stream_sid,
                    "stream_sid": self._stream_sid,
                    "media": {
                        "payload": payload_b64,
                        "codec": self._encoding,
                        "sampleRate": self._dialer_sample_rate,
                    },
                }
            )

        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame) or self._detected_wire_format == "binary":
                return None
            return json.dumps(frame.message)

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes):
            if self._params.wire_format == "auto":
                self._detected_wire_format = "binary"
            pcm_16k = await convert_8k_to_16k_pcm(
                data,
                encoding=self._encoding,
                in_rate=self._dialer_sample_rate,
                out_rate=self._sample_rate,
                resampler=self._input_resampler,
                flush=False,
            )
            if not pcm_16k:
                return None
            return InputAudioRawFrame(
                audio=pcm_16k,
                sample_rate=self._sample_rate,
                num_channels=1,
            )

        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            logger.warning(f"Dialer8kFrameSerializer ignored non-JSON text frame: {data[:120]!r}")
            return None

        if not isinstance(message, dict):
            return None

        if self._params.wire_format == "auto":
            self._detected_wire_format = "json"

        event = str(message.get("event") or message.get("type") or "").lower()
        if event in {"start", "connected"}:
            start_obj = message.get("start") if isinstance(message.get("start"), dict) else message
            stream_sid = (
                start_obj.get("streamSid")
                or start_obj.get("stream_sid")
                or start_obj.get("streamId")
                or message.get("stream_id")
            )
            if isinstance(stream_sid, str) and stream_sid:
                self._stream_sid = stream_sid
            call_sid = (
                start_obj.get("callSid")
                or start_obj.get("call_sid")
                or start_obj.get("callId")
                or start_obj.get("call_control_id")
            )
            if isinstance(call_sid, str) and call_sid:
                self._call_sid = call_sid

            media_fmt = (
                start_obj.get("mediaFormat")
                or start_obj.get("media_format")
                or message.get("mediaFormat")
                or {}
            )
            if isinstance(media_fmt, dict):
                raw_enc = media_fmt.get("encoding") or media_fmt.get("codec")
                if isinstance(raw_enc, str) and raw_enc:
                    self._encoding = normalize_dialer_encoding(raw_enc, default=self._encoding)
                raw_rate = media_fmt.get("sampleRate") or media_fmt.get("sample_rate")
                if isinstance(raw_rate, int) and raw_rate > 0:
                    self._dialer_sample_rate = raw_rate
            return None

        if event in {"media", "audio"}:
            media_obj = message.get("media") if isinstance(message.get("media"), dict) else message
            payload_b64 = (
                media_obj.get("payload")
                or media_obj.get("chunk")
                or message.get("payload")
                or message.get("audio")
            )
            if not isinstance(payload_b64, str) or not payload_b64:
                return None
            try:
                raw_audio = base64.b64decode(payload_b64)
            except Exception:
                return None

            chunk_enc = self._encoding
            if isinstance(media_obj.get("codec"), str):
                chunk_enc = normalize_dialer_encoding(media_obj["codec"], default=self._encoding)
            chunk_rate = self._dialer_sample_rate
            if isinstance(media_obj.get("sampleRate"), int) and media_obj["sampleRate"] > 0:
                chunk_rate = int(media_obj["sampleRate"])

            pcm_16k = await convert_8k_to_16k_pcm(
                raw_audio,
                encoding=chunk_enc,
                in_rate=chunk_rate,
                out_rate=self._sample_rate,
                resampler=self._input_resampler,
                flush=False,
            )
            if not pcm_16k:
                return None
            return InputAudioRawFrame(
                audio=pcm_16k,
                sample_rate=self._sample_rate,
                num_channels=1,
            )

        if event == "dtmf":
            dtmf_obj = message.get("dtmf") if isinstance(message.get("dtmf"), dict) else message
            digit = dtmf_obj.get("digit")
            if digit is not None:
                try:
                    return InputDTMFFrame(KeypadEntry(str(digit)))
                except ValueError:
                    return None

        return None


def build_provider_serializer(
    transport_type: str,
    call_data: dict[str, Any] | Any,
    *,
    default_encoding: DialerEncoding = "ulaw",
    dialer_sample_rate: int = TELEPHONY_SAMPLE_RATE,
    pipeline_in_sample_rate: int = STT_PIPELINE_SAMPLE_RATE,
) -> FrameSerializer:
    """Create a telephony FrameSerializer for Twilio, Exotel, Plivo, Telnyx, or generic dialers.

    Automatically disables `auto_hang_up` when REST API credentials are not set in `.env`
    so local/webhook media stream connections never fail with credential validation errors.
    """
    provider = (transport_type or "dialer").strip().lower()
    stream_id = (
        (call_data.get("stream_id") if hasattr(call_data, "get") else None)
        or getattr(call_data, "stream_id", None)
        or "stream-8k"
    )
    call_id = (
        (call_data.get("call_id") if hasattr(call_data, "get") else None)
        or getattr(call_data, "call_id", None)
        or "call-8k"
    )

    if provider == "twilio":
        from pipecat.serializers.twilio import TwilioFrameSerializer

        account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
        auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
        auto_hang_up = bool(account_sid and auth_token and call_id)
        return TwilioFrameSerializer(
            stream_sid=str(stream_id),
            call_sid=str(call_id) if call_id else None,
            account_sid=account_sid or None,
            auth_token=auth_token or None,
            params=TwilioFrameSerializer.InputParams(
                twilio_sample_rate=dialer_sample_rate,
                sample_rate=pipeline_in_sample_rate,
                auto_hang_up=auto_hang_up,
            ),
        )

    if provider == "exotel":
        from pipecat.serializers.exotel import ExotelFrameSerializer

        return ExotelFrameSerializer(
            stream_sid=str(stream_id),
            call_sid=str(call_id) if call_id else None,
            params=ExotelFrameSerializer.InputParams(
                exotel_sample_rate=dialer_sample_rate,
                sample_rate=pipeline_in_sample_rate,
            ),
        )

    if provider == "plivo":
        from pipecat.serializers.plivo import PlivoFrameSerializer

        auth_id = os.getenv("PLIVO_AUTH_ID", "").strip()
        auth_token = os.getenv("PLIVO_AUTH_TOKEN", "").strip()
        auto_hang_up = bool(auth_id and auth_token and call_id)
        return PlivoFrameSerializer(
            stream_id=str(stream_id),
            call_id=str(call_id) if call_id else None,
            auth_id=auth_id or None,
            auth_token=auth_token or None,
            params=PlivoFrameSerializer.InputParams(
                plivo_sample_rate=dialer_sample_rate,
                sample_rate=pipeline_in_sample_rate,
                auto_hang_up=auto_hang_up,
            ),
        )

    if provider == "telnyx":
        from pipecat.serializers.telnyx import TelnyxFrameSerializer

        api_key = os.getenv("TELNYX_API_KEY", "").strip()
        outbound_enc = (
            (call_data.get("outbound_encoding") if hasattr(call_data, "get") else None)
            or getattr(call_data, "outbound_encoding", None)
            or "PCMU"
        )
        auto_hang_up = bool(api_key and call_id)
        return TelnyxFrameSerializer(
            stream_id=str(stream_id),
            call_control_id=str(call_id) if call_id else None,
            outbound_encoding=str(outbound_enc),
            inbound_encoding="PCMU",
            api_key=api_key or None,
            params=TelnyxFrameSerializer.InputParams(
                telnyx_sample_rate=dialer_sample_rate,
                sample_rate=pipeline_in_sample_rate,
                auto_hang_up=auto_hang_up,
            ),
        )

    return Dialer8kFrameSerializer(
        stream_sid=str(stream_id),
        call_sid=str(call_id) if call_id else None,
        params=Dialer8kFrameSerializer.InputParams(
            dialer_sample_rate=dialer_sample_rate,
            sample_rate=pipeline_in_sample_rate,
            encoding=default_encoding,
        ),
    )
