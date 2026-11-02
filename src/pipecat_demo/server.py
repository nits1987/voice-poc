"""FastAPI + Pipecat Runner server for the Gemini STT -> LLM -> TTS cascade voice bot."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from loguru import logger
from pipecat.runner.run import app, main as pipecat_runner_main
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pydantic import BaseModel, Field

from pipecat_demo.benchmark import run_simulated_voice_turn
from pipecat_demo.config import TurnStopMode, load_config
from pipecat_demo.latency import LatencyStore
from pipecat_demo.pipeline import (
    build_transport_params,
    build_websocket_params,
    create_cascade_session,
    run_cascade_worker,
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
        description="General question to synthesize into 16kHz speech and stream into the cascade.",
    )
    turn_stop_mode: Literal["smart_turn", "vad"] | None = Field(
        default=None,
        description="Optional per-turn endpointing override.",
    )


def rewrite_ws_url_for_request(ws_url: str, request: Request) -> str:
    """Rewrite Pipecat Runner's `wsUrl` to match the caller's actual scheme and Host header."""
    parsed = urlsplit(ws_url)
    forwarded_proto = (
        request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
        or request.url.scheme.lower()
    )
    ws_scheme = "wss" if forwarded_proto in ("https", "wss") else "ws"
    forwarded_host = (
        request.headers.get("x-forwarded-host", "").split(",")[0].strip()
        or request.headers.get("host", "").strip()
        or request.url.netloc
        or parsed.netloc
    )
    path = parsed.path or "/ws-client"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{ws_scheme}://{forwarded_host}{path}{query}"


def register_observatory_routes(target_app: FastAPI, store: LatencyStore) -> None:
    """Attach the custom Latency Observatory UI and telemetry API routes to FastAPI."""
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
            return await run_simulated_voice_turn(req.question.strip(), cfg, store=store)
        except Exception as exc:
            logger.exception("Simulated voice turn failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc


async def bot(runner_args: RunnerArguments) -> None:
    """Pipecat Runner entrypoint invoked for each incoming WebRTC / WebSocket session."""
    cfg = load_config()
    cfg.require_api_key()

    transport_factories = {
        "webrtc": lambda: build_transport_params(cfg),
        "websocket": lambda: build_websocket_params(cfg),
        "daily": lambda: build_transport_params(cfg),
    }
    transport = await create_transport(runner_args, transport_factories)

    session_id = getattr(runner_args, "session_id", None)
    worker, _monitor = create_cascade_session(
        transport,
        cfg,
        store=GLOBAL_LATENCY_STORE,
        session_id=session_id,
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
