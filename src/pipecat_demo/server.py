"""FastAPI + Pipecat Runner server for the Gemini STT -> LLM -> TTS cascade voice bot."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket
from fastapi.responses import FileResponse, RedirectResponse
from loguru import logger
from pipecat.runner.run import app, main as pipecat_runner_main
from pipecat.runner.types import RunnerArguments, WebSocketRunnerArguments
from pipecat.runner.utils import create_transport, parse_telephony_websocket
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport
from pydantic import BaseModel, Field

from pipecat_demo.benchmark import run_simulated_voice_turn
from pipecat_demo.config import CascadeConfig, TurnStopMode, load_config
from pipecat_demo.latency import LatencyStore
from pipecat_demo.pipeline import (
    build_telephony_websocket_params,
    build_transport_params,
    build_websocket_params,
    create_cascade_session,
    run_cascade_worker,
)
from pipecat_demo.telephony import (
    TELEPHONY_SAMPLE_RATE,
    Dialer8kFrameSerializer,
    build_provider_serializer,
    normalize_dialer_encoding,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"

_initial_cfg = load_config()
GLOBAL_LATENCY_STORE = LatencyStore(jsonl_path=_initial_cfg.metrics_jsonl_path, load_existing=True)


class SimulateTurnRequest(BaseModel):
    """Request body for running a full spoken turn without a browser microphone."""

    question: str = Field(
        min_length=2,
        max_length=500,
        description="General question to synthesize into speech and stream into the cascade.",
    )
    turn_stop_mode: Literal["smart_turn", "vad"] | None = Field(
        default=None,
        description="Optional per-turn endpointing override.",
    )
    telephony_8k: bool | None = Field(
        default=None,
        description="When true, simulate an 8kHz telephony dialer (8kHz in -> 16kHz STT -> 24kHz TTS -> 8kHz out).",
    )
    dialer_encoding: Literal["ulaw", "alaw", "pcm16"] | None = Field(
        default=None,
        description="8kHz dialer codec to simulate when telephony_8k is enabled ('ulaw', 'alaw', or 'pcm16').",
    )


def rewrite_ws_url_for_request(ws_url: str, request: Request) -> str:
    """Rewrite Pipecat Runner's `wsUrl` to match the caller's actual scheme and Host header."""
    parsed = urlsplit(ws_url)
    origin_raw = request.headers.get("origin", "").strip()
    referer_raw = request.headers.get("referer", "").strip()
    origin_split = urlsplit(origin_raw) if origin_raw else None
    referer_split = urlsplit(referer_raw) if referer_raw else None

    forwarded_proto = (
        request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
        or (origin_split.scheme.lower() if origin_split and origin_split.scheme else "")
        or (referer_split.scheme.lower() if referer_split and referer_split.scheme else "")
        or request.url.scheme.lower()
    )
    ws_scheme = "wss" if forwarded_proto in ("https", "wss") else "ws"
    forwarded_host = (
        request.headers.get("x-forwarded-host", "").split(",")[0].strip()
        or (origin_split.netloc if origin_split and origin_split.netloc else "")
        or request.headers.get("host", "").strip()
        or request.url.netloc
        or parsed.netloc
    )
    path = parsed.path or "/ws-client"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{ws_scheme}://{forwarded_host}{path}{query}"


def register_observatory_routes(target_app: FastAPI, store: LatencyStore) -> None:
    """Attach the custom Latency Observatory UI, 8kHz telephony endpoints, and telemetry routes."""
    if getattr(target_app.state, "cascade_routes_registered", False):
        return
    target_app.state.cascade_routes_registered = True

    @target_app.middleware("http")
    async def normalize_start_ws_url(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        if request.method != "POST" or request.url.path != "/start" or response.status_code != 200:
            return response

        body_chunks = [chunk async for chunk in response.body_iterator]  # type: ignore[attr-defined]
        raw_body = b"".join(body_chunks)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception:
            headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
            return Response(
                content=raw_body,
                status_code=response.status_code,
                headers=headers,
                media_type=response.media_type,
            )

        if isinstance(payload, dict) and isinstance(payload.get("wsUrl"), str):
            payload["wsUrl"] = rewrite_ws_url_for_request(payload["wsUrl"], request)
            raw_body = json.dumps(payload).encode("utf-8")

        headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
        return Response(
            content=raw_body,
            status_code=response.status_code,
            headers=headers,
            media_type="application/json",
        )

    @target_app.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/dashboard")

    @target_app.get("/dashboard", include_in_schema=False)
    async def serve_dashboard() -> FileResponse:
        return FileResponse(INDEX_HTML_PATH, media_type="text/html")

    @target_app.get("/api/config")
    async def get_cascade_config() -> dict[str, object]:
        cfg = load_config()
        return cfg.public_summary()

    @target_app.get("/api/telephony")
    async def get_telephony_info(request: Request) -> dict[str, object]:
        cfg = load_config()
        ws_base = rewrite_ws_url_for_request("ws://localhost:7860/ws", request)
        ws_dialer = rewrite_ws_url_for_request("ws://localhost:7860/ws-dialer", request)
        return {
            "enabled": cfg.telephony_8k_enabled,
            "dialer_sample_rate_hz": cfg.dialer_sample_rate,
            "stt_pipeline_sample_rate_hz": cfg.audio_in_sample_rate,
            "tts_native_sample_rate_hz": 24000,
            "dialer_encoding": cfg.dialer_encoding,
            "endpoints": {
                "provider_auto_detect_ws": ws_base,
                "universal_dialer_8k_ws": f"{ws_dialer}?codec={cfg.dialer_encoding}&rate={cfg.dialer_sample_rate}",
                "twiml_webhook": str(request.base_url).rstrip("/") + "/twiml",
            },
            "supported_providers": ["twilio", "exotel", "plivo", "telnyx", "custom-8k-dialer"],
            "supported_codecs": ["ulaw", "alaw", "pcm16"],
        }

    @target_app.api_route("/twiml", methods=["GET", "POST"], include_in_schema=False)
    async def twiml_stream_webhook(request: Request) -> Response:
        ws_url = rewrite_ws_url_for_request("ws://localhost:7860/ws", request)
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<Response>\n"
            "  <Connect>\n"
            f'    <Stream url="{ws_url}" />\n'
            "  </Connect>\n"
            "</Response>"
        )
        return Response(content=xml, media_type="application/xml")

    @target_app.websocket("/ws-dialer")
    async def websocket_dialer_8k_endpoint(
        websocket: WebSocket,
        codec: str = Query(default=""),
        rate: int = Query(default=TELEPHONY_SAMPLE_RATE),
        wire: Literal["auto", "json", "binary"] = Query(default="auto"),
    ) -> None:
        """Direct 8kHz telephony WebSocket endpoint for SIP/PBX/custom dialer bridges."""
        cfg = load_config()
        cfg.require_api_key()
        await websocket.accept()

        encoding = normalize_dialer_encoding(codec or cfg.dialer_encoding, default=cfg.dialer_encoding)
        dialer_rate = rate if rate > 0 else cfg.dialer_sample_rate
        tele_cfg = cfg.with_telephony_8k(
            enabled=True,
            encoding=encoding,
            dialer_sample_rate=dialer_rate,
        )
        session_id = f"dialer-{uuid.uuid4().hex[:8]}"

        serializer = Dialer8kFrameSerializer(
            stream_sid=session_id,
            call_sid=session_id,
            params=Dialer8kFrameSerializer.InputParams(
                dialer_sample_rate=dialer_rate,
                sample_rate=tele_cfg.audio_in_sample_rate,
                encoding=encoding,
                wire_format=wire,
            ),
        )
        ws_params = build_telephony_websocket_params(tele_cfg, serializer=serializer)
        transport = FastAPIWebsocketTransport(websocket=websocket, params=ws_params)

        worker, _monitor = create_cascade_session(
            transport,
            tele_cfg,
            store=store,
            session_id=session_id,
            emit_rtvi_messages=False,
            cancel_on_idle_timeout=True,
        )
        await run_cascade_worker(worker, handle_sigint=False)

    @target_app.get("/api/latency")
    async def get_latency_snapshot(
        session_id: str | None = Query(default=None),
    ) -> dict[str, object]:
        return store.snapshot(session_id=session_id)

    @target_app.post("/api/latency/reset")
    async def reset_latency_snapshot(
        clear_jsonl: bool = Query(default=False),
    ) -> dict[str, object]:
        store.reset(clear_jsonl=clear_jsonl)
        return store.snapshot()

    @target_app.post("/api/simulate-turn")
    async def simulate_voice_turn(req: SimulateTurnRequest) -> dict[str, object]:
        cfg = load_config()
        if not cfg.has_api_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"GOOGLE_API_KEY is not set yet. Add your Gemini API key to {cfg.public_summary()['env_file']} "
                    "and click Run Turn again."
                ),
            )
        if req.turn_stop_mode is not None:
            mode: TurnStopMode = "vad" if req.turn_stop_mode == "vad" else "smart_turn"
            cfg = dataclasses.replace(cfg, turn_stop_mode=mode)

        try:
            return await run_simulated_voice_turn(
                req.question.strip(),
                cfg,
                store=store,
                telephony_8k=req.telephony_8k,
                dialer_encoding=req.dialer_encoding,
            )
        except Exception as exc:
            logger.exception("Simulated voice turn failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc


async def create_session_transport(
    runner_args: RunnerArguments,
    cfg: CascadeConfig,
) -> tuple[BaseTransport, CascadeConfig, bool]:
    """Build the appropriate transport and effective CascadeConfig for incoming runner_args.

    Returns `(transport, effective_cfg, is_telephony)`.
    """
    if isinstance(runner_args, WebSocketRunnerArguments) and runner_args.transport_type != "websocket":
        # Incoming telephony dialer connection on `/ws` (Twilio, Exotel, Plivo, Telnyx, or custom)
        transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
        runner_args.transport_type = transport_type
        runner_args.call_data = call_data

        tele_cfg = cfg.with_telephony_8k(enabled=True)
        serializer = build_provider_serializer(
            transport_type,
            call_data,
            default_encoding=tele_cfg.dialer_encoding,
            dialer_sample_rate=tele_cfg.dialer_sample_rate,
            pipeline_in_sample_rate=tele_cfg.audio_in_sample_rate,
        )
        ws_params = build_telephony_websocket_params(tele_cfg, serializer=serializer)
        transport = FastAPIWebsocketTransport(websocket=runner_args.websocket, params=ws_params)
        return transport, tele_cfg, True

    transport_factories = {
        "webrtc": lambda: build_transport_params(cfg),
        "websocket": lambda: build_websocket_params(cfg),
        "daily": lambda: build_transport_params(cfg),
        "twilio": lambda: build_telephony_websocket_params(cfg.with_telephony_8k(enabled=True)),
        "exotel": lambda: build_telephony_websocket_params(cfg.with_telephony_8k(enabled=True)),
        "plivo": lambda: build_telephony_websocket_params(cfg.with_telephony_8k(enabled=True)),
        "telnyx": lambda: build_telephony_websocket_params(cfg.with_telephony_8k(enabled=True)),
    }
    transport = await create_transport(runner_args, transport_factories)
    return transport, cfg, False


async def bot(runner_args: RunnerArguments) -> None:
    """Pipecat Runner entrypoint invoked for each incoming WebRTC / WebSocket / Telephony session."""
    cfg = load_config()
    cfg.require_api_key()

    transport, active_cfg, is_telephony = await create_session_transport(runner_args, cfg)

    session_id = getattr(runner_args, "session_id", None)
    worker, _monitor = create_cascade_session(
        transport,
        active_cfg,
        store=GLOBAL_LATENCY_STORE,
        session_id=session_id,
        emit_rtvi_messages=not is_telephony,
        cancel_on_idle_timeout=True,
    )
    await run_cascade_worker(
        worker,
        handle_sigint=bool(getattr(runner_args, "handle_sigint", False)),
    )


register_observatory_routes(app, GLOBAL_LATENCY_STORE)


def run() -> None:
    """Start the FastAPI + Pipecat Runner server (`pipecat-demo` / `python bot.py`)."""
    # Ensure pipecat.runner.run._get_bot_module() always resolves this module's `bot`
    # even when invoked via the console script entrypoint `uv run pipecat-demo`.
    current_module = sys.modules[__name__]
    sys.modules.setdefault("bot", current_module)
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "bot"):
        setattr(main_mod, "bot", bot)

    # Bind to 0.0.0.0 by default so both SSH localhost tunnels and Cloudtop
    # hostname proxies can reach the server unless --host is explicitly set.
    cli_args = sys.argv[1:]
    has_host_flag = any(a == "--host" or a.startswith("--host=") for a in cli_args)
    has_help_flag = any(a in ("-h", "--help") for a in cli_args)
    if not has_host_flag and not has_help_flag:
        sys.argv.extend(["--host", os.getenv("HOST", "0.0.0.0")])

    cfg = load_config()
    if not cfg.has_api_key:
        logger.warning(
            f"GOOGLE_API_KEY is currently empty. Add your key to {cfg.public_summary()['env_file']} "
            "before starting a voice turn."
        )

    pipecat_runner_main()


if __name__ == "__main__":
    run()
